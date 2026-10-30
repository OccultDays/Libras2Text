"""Evaluate a sequence classifier on the already downloaded LIBRAS landmarks.

Uses a per-class 3/1/1 signer-label split (train/validation/test), matching the
dataset's intended protocol. The local signer IDs are per class, not global.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PREFIX = "Landmarks/Libras-EQT-UECE (Hand Landmarks)/"
FRAMES = 16


def norm_hand(values):
    hand = values.reshape(21, 3)
    if not np.isfinite(hand).all() or np.linalg.norm(hand[0, :2]) < 0.02:
        return None
    if np.count_nonzero(np.linalg.norm(hand[:, :2], axis=1) > 0.02) < 18:
        return None
    relative = hand - hand[0]
    scale = np.linalg.norm(relative, axis=1).max()
    if scale <= 1e-7:
        return None
    return (relative / scale).astype(np.float32)


def resample(sequence):
    # Uniform temporal sampling retains sign order and dynamic motion while
    # making clips of different lengths comparable to a fixed-size model.
    positions = np.linspace(0, len(sequence) - 1, FRAMES)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    broadcast_shape = (len(weight),) + (1,) * (sequence.ndim - 1)
    weight = weight.reshape(broadcast_shape)
    sampled = sequence[lower] * (1.0 - weight) + sequence[upper] * weight
    return sampled.reshape(-1)


def load(zip_path, both_hands=False, include_position=False, max_classes=None):
    x, y, signer, clip_ids = [], [], [], []
    dropped = defaultdict(int)
    selected_labels = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.startswith(PREFIX) or not name.endswith(".npy"):
                continue
            parts = name[len(PREFIX):].split("/")
            if len(parts) != 3:
                continue
            folder, signer_name = parts[:2]
            label = re.sub(r"^\d+_", "", folder).strip()
            # A first release vocabulary of one-word lexical items; leave
            # sentence classes for a separate sequence-to-text model.
            if " " in label:
                continue
            if label not in selected_labels:
                if max_classes is not None and len(selected_labels) >= max_classes:
                    continue
                selected_labels.add(label)
            try:
                raw = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            except (ValueError, OSError):
                dropped["invalid_file"] += 1
                continue
            if raw.ndim != 2 or raw.shape[1] != 126:
                dropped["unexpected_shape"] += 1
                continue
            pairs = raw.reshape(-1, 2, 21, 3)
            normalized = [[norm_hand(hand) for hand in pair] for pair in pairs]
            candidates = []
            for side in range(2):
                seq = [frame[side] for frame in normalized if frame[side] is not None]
                energy = np.mean([np.linalg.norm(b - a) for a, b in zip(seq, seq[1:])]) if len(seq) > 1 else 0
                candidates.append((energy, side, seq))
            _, active_side, active_seq = max(candidates, key=lambda candidate: candidate[0])
            if len(active_seq) < 10:
                dropped["too_few_valid_frames"] += 1
                continue
            if both_hands:
                # Canonicalize the most active hand into slot 0 and retain the
                # companion hand in slot 1; zero means not tracked in a frame.
                companion_side = 1 - active_side
                frames = []
                for frame in normalized:
                    active = frame[active_side] if frame[active_side] is not None else np.zeros((21, 3), np.float32)
                    companion = frame[companion_side] if frame[companion_side] is not None else np.zeros((21, 3), np.float32)
                    frames.append(np.concatenate([active, companion], axis=0))
                seq = np.asarray(frames, dtype=np.float32)
            else:
                seq = np.asarray(active_seq)
                if include_position:
                    # Append the tracked hand wrist in camera-normalized image
                    # coordinates. The shape remains wrist-relative; location
                    # carries trajectory and articulation context.
                    # Keep the feature sequence aligned with accepted frames.
                    accepted_wrists = []
                    for raw_pair, pair in zip(pairs, normalized):
                        if pair[active_side] is not None:
                            accepted_wrists.append(raw_pair[active_side, 0])
                    seq = np.concatenate([seq.reshape(len(seq), -1), np.asarray(accepted_wrists, dtype=np.float32)], axis=1)
            x.append(resample(seq))
            y.append(label)
            signer.append(int(signer_name.rsplit(" ", 1)[-1]) - 1)
            clip_ids.append(name)
    return np.asarray(x), np.asarray(y), np.asarray(signer), clip_ids, dict(dropped)


def pipeline(c):
    # LinearSVC is fast and its multiclass coefficients can be exported for
    # direct on-device inference without shipping support vectors.
    return make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=10000))


def evaluate(x, y, signer):
    fold_results = []
    chosen_cs = []
    for fold in range(5):
        test_id = fold
        val_id = (fold + 1) % 5
        train = np.flatnonzero((signer != test_id) & (signer != val_id))
        val = np.flatnonzero(signer == val_id)
        test = np.flatnonzero(signer == test_id)
        best_c, best_val_f1 = None, -1
        for c in (0.1, 1.0, 10.0):
            model = pipeline(c).fit(x[train], y[train])
            score = f1_score(y[val], model.predict(x[val]), average="macro", zero_division=0)
            if score > best_val_f1:
                best_c, best_val_f1 = c, score
        # Refit on train + validation after selecting C; test signer remains unseen.
        fit = np.concatenate([train, val])
        model = pipeline(best_c).fit(x[fit], y[fit])
        pred = model.predict(x[test])
        acc = accuracy_score(y[test], pred)
        macro = f1_score(y[test], pred, average="macro", zero_division=0)
        confusions = Counter((actual, guess) for actual, guess in zip(y[test], pred) if actual != guess)
        worst = ", ".join(f"{actual}->{guess} ({count})" for (actual, guess), count in confusions.most_common(3))
        chosen_cs.append(best_c)
        fold_results.append((acc, macro))
        print(f"Fold {fold + 1} (teste por item: informante {test_id + 1}; C={best_c:g}): acuracia={acc:.3f}, macro-F1={macro:.3f}")
        if worst:
            print(f"  Confusoes mais comuns: {worst}")
    return np.mean(fold_results, axis=0), chosen_cs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="Caminho para LIBRAS-EQT-UECE-Landmarks.zip")
    parser.add_argument("--both-hands", action="store_true", help="Inclui a mao companheira no vetor temporal")
    parser.add_argument("--include-position", action="store_true", help="Acrescenta a trajetoria do pulso em coordenadas normalizadas da imagem")
    parser.add_argument("--max-classes", type=int, help="Limita o piloto as primeiras N classes para comparacoes rapidas")
    args = parser.parse_args()
    x, y, signer, _, dropped = load(args.zip, args.both_hands, args.include_position, args.max_classes)
    print(f"Clipes={len(y)}; classes={len(np.unique(y))}; features por clipe={x.shape[1]}; descartados={dropped}")
    (accuracy, macro), cs = evaluate(x, y, signer)
    print(f"Media teste: acuracia={accuracy:.3f}; macro-F1={macro:.3f}; C selecionado por fold={cs}")
    print("Candidato experimental; nao exportado nem ativado no app.")


if __name__ == "__main__":
    main()
