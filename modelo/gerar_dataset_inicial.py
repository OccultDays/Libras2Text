import os
import csv
import numpy as np

def obter_configuracao_sinal(sinal):
    dedos = {"polegar": 0.0, "indicador": 0.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
    abertura = 0.0
    pos_mao_inicio = [0.25, -0.15, 0.20]
    pos_mao_fim = [0.25, -0.15, 0.20]
    inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
    inclinacao_cabeca_fim = [0.0, 0.0, 0.0]

    if sinal == "OI":
        dedos = {"polegar": 0.8, "indicador": 1.0, "medio": 1.0, "anelar": 1.0, "minimo": 1.0}
        abertura = 0.6
        pos_mao_inicio = [0.28, 0.35, 0.15]
        pos_mao_fim = [0.42, 0.35, 0.15]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.05, 0.05, 0.0]

    elif sinal == "BOM DIA":
        dedos = {"polegar": 0.6, "indicador": 0.8, "medio": 0.8, "anelar": 0.8, "minimo": 0.8}
        pos_mao_inicio = [0.05, 0.20, 0.20]
        pos_mao_fim = [0.15, -0.10, 0.40]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.05, 0.0, 0.0]

    elif sinal == "OBRIGADO":
        dedos = {"polegar": 0.2, "indicador": 1.0, "medio": 1.0, "anelar": 1.0, "minimo": 1.0}
        pos_mao_inicio = [0.10, 0.42, 0.12]
        pos_mao_fim = [0.15, 0.05, 0.40]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.10, 0.0, 0.0]

    elif sinal == "POR FAVOR":
        dedos = {"polegar": 0.5, "indicador": 1.0, "medio": 1.0, "anelar": 1.0, "minimo": 1.0}
        pos_mao_inicio = [0.0, -0.05, 0.20]
        pos_mao_fim = [0.0, -0.05, 0.35]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.08, 0.0, 0.0]

    elif sinal == "SIM":
        dedos = {"polegar": 0.4, "indicador": 0.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        pos_mao_inicio = [0.20, 0.20, 0.20]
        pos_mao_fim = [0.20, 0.05, 0.25]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.15, 0.0, 0.0]

    elif sinal == "NAO":
        dedos = {"polegar": 0.0, "indicador": 1.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        pos_mao_inicio = [0.10, 0.25, 0.28]
        pos_mao_fim = [0.25, 0.25, 0.28]
        inclinacao_cabeca_inicio = [0.0, -0.10, 0.0]
        inclinacao_cabeca_fim = [0.0, 0.10, 0.0]

    elif sinal == "EU":
        dedos = {"polegar": 0.0, "indicador": 1.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        pos_mao_inicio = [0.0, 0.0, 0.25]
        pos_mao_fim = [0.0, -0.10, 0.05]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [-0.05, 0.0, 0.0]

    elif sinal == "VOCE":
        dedos = {"polegar": 0.0, "indicador": 1.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        pos_mao_inicio = [0.05, -0.05, 0.15]
        pos_mao_fim = [0.05, 0.05, 0.55]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.0, 0.0, 0.0]

    elif sinal == "CASA":
        dedos = {"polegar": 0.5, "indicador": 1.0, "medio": 1.0, "anelar": 1.0, "minimo": 1.0}
        pos_mao_inicio = [0.15, 0.10, 0.20]
        pos_mao_fim = [0.0, 0.0, 0.25]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.0, 0.0, 0.0]

    elif sinal == "AJUDA":
        dedos = {"polegar": 0.6, "indicador": 0.6, "medio": 0.6, "anelar": 0.6, "minimo": 0.6}
        pos_mao_inicio = [0.10, -0.12, 0.15]
        pos_mao_fim = [0.10, -0.05, 0.40]
        inclinacao_cabeca_inicio = [0.0, 0.0, 0.0]
        inclinacao_cabeca_fim = [0.05, 0.0, 0.0]

    elif sinal == "A":
        dedos = {"polegar": 0.5, "indicador": 0.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        pos_mao_inicio = pos_mao_fim = [0.25, 0.05, 0.20]
    elif sinal == "B":
        dedos = {"polegar": 0.0, "indicador": 1.0, "medio": 1.0, "anelar": 1.0, "minimo": 1.0}
        pos_mao_inicio = pos_mao_fim = [0.25, 0.08, 0.20]
    elif sinal == "C":
        dedos = {"polegar": 0.6, "indicador": 0.6, "medio": 0.6, "anelar": 0.6, "minimo": 0.6}
        abertura = 0.4
        pos_mao_inicio = pos_mao_fim = [0.25, 0.05, 0.20]
    elif sinal == "L":
        dedos = {"polegar": 1.0, "indicador": 1.0, "medio": 0.0, "anelar": 0.0, "minimo": 0.0}
        abertura = 0.8
        pos_mao_inicio = pos_mao_fim = [0.25, 0.10, 0.20]
    elif sinal == "V":
        dedos = {"polegar": 0.0, "indicador": 1.0, "medio": 1.0, "anelar": 0.0, "minimo": 0.0}
        abertura = 0.5
        pos_mao_inicio = pos_mao_fim = [0.25, 0.10, 0.20]

    return dedos, abertura, pos_mao_inicio, pos_mao_fim, inclinacao_cabeca_inicio, inclinacao_cabeca_fim

def construir_pose_frame(pos_mao, inclinacao, dedos, abertura):
    pontos_cabeca = np.array([
        [0.0, 0.35, 0.05],
        [-0.04, 0.38, 0.03],
        [0.04, 0.38, 0.03],
        [-0.10, 0.36, -0.05],
        [0.10, 0.36, -0.05],
        [-0.03, 0.30, 0.04],
        [0.03, 0.30, 0.04]
    ])
    pontos_cabeca[:, 1] += inclinacao[0]
    pontos_cabeca[:, 0] += inclinacao[1]

    ombro_esq = [-0.22, 0.0, 0.0]
    ombro_dir = [0.22, 0.0, 0.0]
    pulso_esq = [-0.20, -0.30, 0.10]
    pulso_dir = list(pos_mao)

    cotovelo_esq = [(-0.22 + pulso_esq[0]) / 2.0 - 0.08, -0.18, 0.05]
    cotovelo_dir = [(0.22 + pulso_dir[0]) / 2.0 + 0.08, (ombro_dir[1] + pulso_dir[1]) / 2.0 - 0.10, 0.08]

    pontos_bracos = np.array([
        ombro_esq, ombro_dir,
        cotovelo_esq, cotovelo_dir,
        pulso_esq, pulso_dir
    ])

    pontos_mao = np.zeros((21, 3))
    pontos_mao[0] = pulso_dir

    larguras_mcp = [-0.04, -0.015, 0.0, 0.015, 0.03]
    alturas_mcp = [0.03, 0.07, 0.08, 0.07, 0.06]

    cadeias = [
        ("polegar", [1, 2, 3, 4], 0),
        ("indicador", [5, 6, 7, 8], 1),
        ("medio", [9, 10, 11, 12], 2),
        ("anelar", [13, 14, 15, 16], 3),
        ("minimo", [17, 18, 19, 20], 4)
    ]

    for nome_dedo, indices, idx_dedo in cadeias:
        extensao = dedos[nome_dedo]
        x_base = pulso_dir[0] + larguras_mcp[idx_dedo]
        y_base = pulso_dir[1] + alturas_mcp[idx_dedo]

        desloc_abertura = 0.0
        if nome_dedo == "polegar":
            desloc_abertura = -0.04 * abertura
        elif nome_dedo == "minimo":
            desloc_abertura = 0.04 * abertura

        for etapa, idx_ponto in enumerate(indices):
            fator = (etapa + 1) * 0.025
            if extensao > 0.5:
                x = x_base + desloc_abertura * (etapa + 1) * 0.5
                y = y_base + fator * extensao
                z = pulso_dir[2] - 0.005 * (etapa + 1)
            else:
                x = x_base + desloc_abertura * 0.2
                y = y_base + (fator * 0.2)
                z = pulso_dir[2] + 0.01 * (etapa + 1)

            pontos_mao[idx_ponto] = [x, y, z]

    todos_pontos = np.vstack([pontos_cabeca, pontos_bracos, pontos_mao])
    centro = (todos_pontos[7] + todos_pontos[8]) / 2.0
    ajustado = todos_pontos - centro
    escala = np.max(np.linalg.norm(ajustado, axis=1))
    if escala > 0:
        ajustado = ajustado / escala

    return ajustado.flatten()

def gerar_amostra_temporal(sinal, ruido=0.015):
    dedos, abertura, pos_inicio, pos_fim, inc_inicio, inc_fim = obter_configuracao_sinal(sinal)

    # Gera quadro inicial e final do gesto
    frame_inicio = construir_pose_frame(pos_inicio, inc_inicio, dedos, abertura)
    frame_fim = construir_pose_frame(pos_fim, inc_fim, dedos, abertura)

    if ruido > 0:
        frame_inicio += np.random.normal(0, ruido, frame_inicio.shape)
        frame_fim += np.random.normal(0, ruido, frame_fim.shape)

    deslocamento = frame_fim - frame_inicio
    # Vetor de 204 valores: 102 do quadro atual/final + 102 do deslocamento temporal
    return np.concatenate([frame_fim, deslocamento])

def criar_dataset():
    sinais = [
        "OI", "BOM DIA", "OBRIGADO", "POR FAVOR",
        "SIM", "NAO", "EU", "VOCE", "CASA", "AJUDA",
        "A", "B", "C", "L", "V"
    ]
    amostras_por_sinal = 80
    caminho_csv = os.path.join(os.path.dirname(__file__), "dataset_libras.csv")

    linhas = []
    # 102 (posicao atual) + 102 (deslocamento) = 204 colunas
    cabecalho = ["rotulo"] + [f"coord_{i}" for i in range(204)]
    linhas.append(cabecalho)

    for sinal in sinais:
        for _ in range(amostras_por_sinal):
            vetor = gerar_amostra_temporal(sinal, ruido=0.015)
            linha = [sinal] + list(vetor)
            linhas.append(linha)

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerows(linhas)

    print(f"Dataset temporal criado com {len(linhas)-1} amostras (204 dimensoes) em {caminho_csv}")

if __name__ == "__main__":
    criar_dataset()
