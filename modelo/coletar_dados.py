from collections import deque
import csv
import os
import sys
import time
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "servidor"))
from extrator_pontos import ExtratorPontos, montar_vetor_temporal

DURACAO_GESTO_SEGUNDOS = 1.4

# Classes adicionais preparadas para coleta de exemplos reais.
# J e Z são mantidas como rótulos únicos, embora envolvam movimento.
ROTULOS_EXPANSAO = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "OI", "BOM DIA", "OBRIGADO", "POR FAVOR", "SIM", "NAO", "EU", "VOCE", "CASA", "AJUDA",
    "TUDO BEM", "TCHAU", "BOM", "AGUA", "COMER", "DESCULPA",
    "QUERER", "ONDE", "NOME", "POR QUE",
]

CONEXOES_MAO = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 17),
]


def desenhar_maos(quadro, pontos_maos):
    altura, largura = quadro.shape[:2]
    for mao in pontos_maos:
        if len(mao) != 21:
            continue
        for a, b in CONEXOES_MAO:
            p1 = (int(mao[a][0] * largura), int(mao[a][1] * altura))
            p2 = (int(mao[b][0] * largura), int(mao[b][1] * altura))
            cv2.line(quadro, p1, p2, (40, 200, 60), 2)
        for p in mao:
            centro = (int(p[0] * largura), int(p[1] * altura))
            cv2.circle(quadro, centro, 4, (255, 255, 255), -1)
            cv2.circle(quadro, centro, 4, (30, 140, 45), 1)


def coletar():
    extrator = ExtratorPontos()
    caminho_csv = os.path.join(os.path.dirname(__file__), "dataset_libras_real.csv")
    print("Rotulos novos preparados para coleta:")
    print(", ".join(ROTULOS_EXPANSAO))
    sinal = input("Digite o rotulo do sinal (ex: TUDO BEM, G, J): ").strip().upper()
    if not sinal:
        print("Rotulo invalido.")
        return

    if sinal not in ROTULOS_EXPANSAO:
        print("Use um dos rotulos da lista para evitar classes duplicadas ou erros de digitacao.")
        return

    quantidade = int(input("Quantos exemplos gravar? (recomendado: 30): ") or "30")
    if quantidade < 2:
        print("Grave pelo menos 2 exemplos por rotulo.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    existe_arquivo = os.path.exists(caminho_csv)
    cabecalho = ["rotulo"] + [f"coord_{i}" for i in range(110)]
    sequencias = deque(maxlen=9)
    gravadas = 0

    print(
        "Grave o sinal conforme a Libras usada na sua comunidade. "
        "Pressione ESPACO para gravar cada exemplo e Q para sair.\n"
    )

    with open(caminho_csv, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        if not existe_arquivo:
            escritor.writerow(cabecalho)

        while gravadas < quantidade:
            sucesso, quadro = camera.read()
            if not sucesso:
                break

            resultado = extrator.extrair_marcos(quadro)
            pronto = bool(resultado and resultado["tem_mao"] and resultado["quadro"] is not None)
            if resultado and resultado["pontos_maos"]:
                desenhar_maos(quadro, resultado["pontos_maos"])

            instrucao = "Mao detectada - ESPACO grava" if pronto else "Mostre a mao inteira na camera"
            cor = (0, 200, 0) if pronto else (0, 0, 220)
            cv2.putText(quadro, instrucao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)
            cv2.putText(
                quadro,
                f"{sinal}: {gravadas}/{quantidade}",
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Coleta de Libras", quadro)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla != ord(" "):
                continue

            sequencias.clear()
            inicio = time.monotonic()
            sair = False
            while time.monotonic() - inicio < DURACAO_GESTO_SEGUNDOS:
                sucesso, quadro_gesto = camera.read()
                if not sucesso:
                    break

                amostra = extrator.extrair_marcos(quadro_gesto)
                if amostra and amostra["pontos_maos"]:
                    desenhar_maos(quadro_gesto, amostra["pontos_maos"])
                if amostra and amostra["tem_mao"] and amostra["quadro"] is not None:
                    q = amostra["quadro"]
                    if len(sequencias) >= 2 and (q["instante"] - sequencias[-2]["instante"]) < 0.11:
                        sequencias[-1] = q
                    else:
                        sequencias.append(q)

                cv2.putText(
                    quadro_gesto,
                    f"Gravando {sinal}...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 200, 0),
                    2,
                )
                cv2.imshow("Coleta de Libras", quadro_gesto)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sair = True
                    break

            if sair:
                break

            if len(sequencias) < 2:
                print("Poucos quadros validos. Tente novamente com a mao bem visivel.")
                continue

            vetor_final = montar_vetor_temporal(list(sequencias))
            escritor.writerow([sinal] + [round(float(v), 6) for v in vetor_final])
            arquivo.flush()
            gravadas += 1
            print(f"Exemplo {gravadas}/{quantidade} salvo para {sinal} ({len(sequencias)} quadros).")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nColeta concluida em: {caminho_csv}")
    print("Treine o modelo com: python modelo/treinar_modelo.py")


if __name__ == "__main__":
    coletar()
