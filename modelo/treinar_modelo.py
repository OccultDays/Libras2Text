import csv
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from exportar_modelo_mobile import exportar
from gerar_dataset_inicial import criar_dataset

TOTAL_CARACTERISTICAS = 110


def carregar_csv(caminho):
    rotulos = []
    caracteristicas = []
    with open(caminho, "r", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo)
        next(leitor, None)
        for linha in leitor:
            if not linha or len(linha) != TOTAL_CARACTERISTICAS + 1:
                continue
            rotulos.append(linha[0])
            caracteristicas.append([float(v) for v in linha[1:]])
    return rotulos, caracteristicas


def treinar():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_real = os.path.join(diretorio_atual, "dataset_libras_real.csv")
    caminho_base = os.path.join(diretorio_atual, "dataset_libras.csv")

    criar_dataset()
    rotulos_base, caract_base = carregar_csv(caminho_base)

    fonte_dados = "base_anatomica"
    rotulos = list(rotulos_base)
    caracteristicas = list(caract_base)

    if os.path.exists(caminho_real):
        rotulos_reais, caract_reais = carregar_csv(caminho_real)
        if rotulos_reais:
            classes_reais = set(rotulos_reais)
            filtrados = [
                (r, c) for r, c in zip(rotulos_base, caract_base)
                if r not in classes_reais
            ]
            rotulos = [r for r, _ in filtrados] + rotulos_reais
            caracteristicas = [c for _, c in filtrados] + caract_reais
            fonte_dados = "real"
            print(f"Integrando {len(rotulos_reais)} amostras reais coletadas ({len(classes_reais)} sinais).")

    x = np.array(caracteristicas, dtype=np.float64)
    y = np.array(rotulos)

    contagens = {rotulo: int(np.sum(y == rotulo)) for rotulo in np.unique(y)}
    insuficientes = [rotulo for rotulo, total in contagens.items() if total < 2]
    if insuficientes:
        raise ValueError(f"Grave pelo menos 2 exemplos para: {', '.join(insuficientes)}")

    fracao_teste = max(0.2, len(np.unique(y)) / len(y))
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=fracao_teste, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=75,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(x_treino, y_treino)

    previsoes = modelo.predict(x_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"\nAcuracia na validacao: {acuracia * 100:.2f}%\n")
    print(classification_report(y_teste, previsoes))

    modelo.fit(x, y)
    modelo.n_jobs = 1
    rotulos_unicos = sorted(list(set(y)))

    pasta_saida = os.path.join(diretorio_atual, "..", "servidor", "modelos_ia")
    os.makedirs(pasta_saida, exist_ok=True)

    caminho_modelo = os.path.join(pasta_saida, "modelo_libras.joblib")
    caminho_rotulos = os.path.join(pasta_saida, "rotulos.joblib")

    joblib.dump(modelo, caminho_modelo)
    joblib.dump(rotulos_unicos, caminho_rotulos)
    exportar(fonte_dados=fonte_dados)

    print(f"Modelo salvo em: {caminho_modelo}")
    print(f"Rotulos salvos em: {caminho_rotulos}")


if __name__ == "__main__":
    treinar()
