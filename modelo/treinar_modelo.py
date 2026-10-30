import os
import joblib
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from gerar_dataset_inicial import criar_dataset
from exportar_modelo_mobile import exportar

def treinar():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_real = os.path.join(diretorio_atual, "dataset_libras_real.csv")
    caminho_sintetico = os.path.join(diretorio_atual, "dataset_libras.csv")

    if os.path.exists(caminho_real):
        caminho_csv = caminho_real
        print(f"Treinando com exemplos reais coletados: {caminho_csv}")
    else:
        caminho_csv = caminho_sintetico
        if not os.path.exists(caminho_csv):
            criar_dataset()
        print("AVISO: dataset_libras_real.csv nao encontrado; usando dados sinteticos de demonstracao.")
        print("Esse modelo nao representa a variacao real dos sinais de Libras.")

    rotulos = []
    caracteristicas = []

    with open(caminho_csv, "r", encoding="utf-8") as f:
        leitor = csv.reader(f)
        _cabecalho = next(leitor)
        for linha in leitor:
            if not linha:
                continue
            rotulos.append(linha[0])
            caracteristicas.append([float(v) for v in linha[1:]])

    x = np.array(caracteristicas)
    y = np.array(rotulos)

    contagens = {rotulo: int(np.sum(y == rotulo)) for rotulo in np.unique(y)}
    insuficientes = [rotulo for rotulo, total in contagens.items() if total < 2]
    if insuficientes:
        raise ValueError(
            "Cada rotulo precisa de pelo menos 2 exemplos para treino e validacao. "
            f"Colete mais exemplos para: {', '.join(insuficientes)}"
        )

    if x.ndim != 2 or x.shape[1] != 204:
        raise ValueError(f"Esperadas 204 caracteristicas por exemplo, recebidas: {x.shape}")

    fracao_teste = max(0.2, len(np.unique(y)) / len(y))
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=fracao_teste, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(x_treino, y_treino)

    previsoes = modelo.predict(x_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"\nAcuracia no teste: {acuracia * 100:.2f}%\n")
    print(classification_report(y_teste, previsoes))

    rotulos_unicos = sorted(list(set(y)))

    pasta_saida = os.path.join(diretorio_atual, "..", "servidor", "modelos_ia")
    os.makedirs(pasta_saida, exist_ok=True)

    caminho_modelo = os.path.join(pasta_saida, "modelo_libras.joblib")
    caminho_rotulos = os.path.join(pasta_saida, "rotulos.joblib")

    joblib.dump(modelo, caminho_modelo)
    joblib.dump(rotulos_unicos, caminho_rotulos)
    fonte_dados = "real" if caminho_csv == caminho_real else "sintetico_demo"
    exportar(fonte_dados=fonte_dados)

    print(f"Modelo salvo em: {caminho_modelo}")
    print(f"Rotulos salvos em: {caminho_rotulos}")

if __name__ == "__main__":
    treinar()
