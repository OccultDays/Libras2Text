"""Train and evaluate a local classifier from LIBRAS-EQT-UECE landmarks.

The archive is intentionally read in place. It is not unpacked or committed.
The evaluation keeps all clips from one informant together to avoid leakage.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold



HAND_PREFIX = "Landmarks/Libras-EQT-UECE (Hand Landmarks)/"


def normalizar_mao(frame: np.ndarray) -> np.ndarray | None:
    """Normalize one 21-point hand in the same wrist-relative way as the app."""
    if frame.size != 63:
        return None
    hand = frame.reshape(21, 3)
    if not np.isfinite(hand).all() or np.linalg.norm(hand[0, :2]) < 0.02:
        return None
    if np.count_nonzero(np.linalg.norm(hand[:, :2], axis=1) > 0.02) < 18:
        return None
    adjusted = hand - hand[0]
    scale = float(np.linalg.norm(adjusted, axis=1).max())
    if scale <= 1e-7:
        return None
    return (adjusted / scale).reshape(63).astype(np.float32)


def label_from_path(path: str) -> str:
    folder = path.split("/")[-3]
    # Folder IDs identify catalog order and are not part of the spoken output.
    return re.sub(r"^\d+_", "", folder).strip()


def load_archive(path: str, lexical_only: bool = False):
    prefix = HAND_PREFIX
    features, labels, groups = [], [], []
    clip_count = 0
    dropped = defaultdict(int)
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith(".npy"):
                continue
            parts = name[len(prefix):].split("/")
            if len(parts) != 3:
                continue
            label, informant = label_from_path(name), parts[1]
            if lexical_only and " " in label:
                continue
            try:
                sequence = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            except (ValueError, OSError):
                dropped["invalid_file"] += 1
                continue
            if sequence.ndim != 2 or sequence.shape[1] != 126:
                dropped["unexpected_shape"] += 1
                continue
            hands = sequence.reshape(-1, 2, 21, 3)
            normalized = [[normalizar_mao(hand) for hand in pair] for pair in hands]
            # Select the hand that moves most across the clip. This approximates
            # the runtime's wrist association without assuming handedness order.
            scores = []
            for side in range(2):
                valid = [v for v in (frame[side] for frame in normalized) if v is not None]
                energy = float(np.mean([np.linalg.norm(b - a) for a, b in zip(valid, valid[1:])])) if len(valid) > 1 else 0
                scores.append((energy, side, valid))
            _, side, vectors = max(scores, key=lambda row: row[0])
            if len(vectors) < 10:
                dropped["too_few_valid_frames"] += 1
                continue
            clip_count += 1
            vectors = np.asarray(vectors, dtype=np.float32)
            # Match App.js: hand shape plus displacement from the oldest frame
            # in its rolling 10-frame buffer. Sample sparsely within each clip.
            for end in range(9, len(vectors), 5):
                start = max(0, end - 9)
                current = vectors[end]
                sample = np.concatenate([current, current - vectors[start]])
                features.append(sample)
                labels.append(label)
                groups.append(informant)
    if not features:
        raise RuntimeError("Nenhum clipe valido encontrado; confira o caminho/layout do ZIP.")
    return np.asarray(features), np.asarray(labels), np.asarray(groups), clip_count, dict(dropped)


def novo_modelo():
    # 30 shallow trees keep on-device inference and the bundled JSON modest.
    return RandomForestClassifier(
        n_estimators=15,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def avaliar(x, y, groups):
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    resultados = []
    for fold, (train, test) in enumerate(cv.split(x, y, groups), start=1):
        model = novo_modelo().fit(x[train], y[train])
        pred = model.predict(x[test])
        acc = accuracy_score(y[test], pred)
        macro = f1_score(y[test], pred, average="macro", zero_division=0)
        print(f"Informante teste {fold}: acuracia={acc:.3f}, macro-F1={macro:.3f}")
        resultados.append((acc, macro))
    return np.mean(resultados, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="Caminho para LIBRAS-EQT-UECE-Landmarks.zip")
    parser.add_argument("--lexical-only", action="store_true", help="Limita o piloto a classes com rótulo de uma palavra")
    args = parser.parse_args()

    x, y, groups, clips, dropped = load_archive(args.zip, args.lexical_only)
    print(f"Clipes validos: {clips}; amostras temporais: {len(y)}; classes: {len(np.unique(y))}; informantes: {len(np.unique(groups))}")
    print(f"Descartados: {dropped}")
    accuracy, macro = avaliar(x, y, groups)
    print(f"Media por informante: acuracia={accuracy:.3f}, macro-F1={macro:.3f}")

    model = novo_modelo().fit(x, y)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "servidor", "modelos_ia", "candidato_libras_eqt_uece.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Candidato de pesquisa salvo (nao instalado no app): {model_path}")


if __name__ == "__main__":
    main()
