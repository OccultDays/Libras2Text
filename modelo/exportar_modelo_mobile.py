import json
import os
import joblib
import numpy as np


def exportar(fonte_dados="base_anatomica"):
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_modelo = os.path.join(raiz, "servidor", "modelos_ia", "modelo_libras.joblib")
    caminho_saida = os.path.join(raiz, "aplicativo", "assets", "modelo_libras_mobile.json")
    modelo = joblib.load(caminho_modelo)

    arvores = []
    for estimador in modelo.estimators_:
        arvore = estimador.tree_
        valores = arvore.value[:, 0, :]
        totais = valores.sum(axis=1, keepdims=True)
        probabilidades = np.divide(valores, totais, out=np.zeros_like(valores), where=totais != 0)
        arvores.append({
            "esquerda": arvore.children_left.tolist(),
            "direita": arvore.children_right.tolist(),
            "caracteristica": arvore.feature.tolist(),
            "limite": np.round(arvore.threshold, 6).tolist(),
            "probabilidades": np.round(probabilidades, 4).tolist(),
        })

    artefato = {
        "versao": 2,
        "fonte_dados": fonte_dados,
        "quantidade_caracteristicas": int(modelo.n_features_in_),
        "rotulos": [str(rotulo) for rotulo in modelo.classes_],
        "arvores": arvores,
    }
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(artefato, arquivo, separators=(",", ":"), ensure_ascii=False)

    print(f"Modelo mobile exportado: {caminho_saida}")
    print(f"Arvores: {len(arvores)} | caracteristicas: {modelo.n_features_in_}")


if __name__ == "__main__":
    exportar()
