"""Train and export the experimental 20-class hand-sequence model for Android."""

from __future__ import annotations

import argparse
import json
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from avaliar_modelo_temporal_publico import load, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="Path to LIBRAS-EQT-UECE-Landmarks.zip")
    parser.add_argument("--output", help="Output JSON path; defaults to the app asset")
    args = parser.parse_args()

    x, y, _, _, dropped = load(args.zip, max_classes=20)
    # C=0.1 was selected in four of the five signer-held-out folds.
    scaler = StandardScaler().fit(x)
    classifier = pipeline(0.1).named_steps["linearsvc"].fit(scaler.transform(x), y)
    model = {
        "fonte_dados": "real_publico_experimental",
        "dataset": "LIBRAS-EQT-UECE Landmarks 1.1",
        "protocolo": "piloto das primeiras 20 classes lexicais; modelo final ajustado em todos os clipes dessas classes",
        "frames": 16,
        "pontos_por_frame": 21,
        "features_por_frame": 63,
        "quantidade_caracteristicas": int(x.shape[1]),
        "rotulos": [str(label) for label in classifier.classes_],
        "media": scaler.mean_.tolist(),
        "escala": scaler.scale_.tolist(),
        "coeficientes": classifier.coef_.tolist(),
        "interceptos": classifier.intercept_.tolist(),
        "confianca_calibrada": False,
    }
    output = args.output or os.path.join(os.path.dirname(__file__), "..", "aplicativo", "assets", "modelo_libras_temporal_piloto.json")
    output = os.path.abspath(output)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, separators=(",", ":"))
    print(f"Clipes={len(y)}; classes={len(classifier.classes_)}; features={x.shape[1]}; descartados={dropped}")
    print(f"Modelo experimental exportado para {output} ({os.path.getsize(output)} bytes); nao calibrado para traducao confiavel.")


if __name__ == "__main__":
    main()
