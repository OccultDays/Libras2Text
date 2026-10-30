"""Collect real temporal Libras examples for training the classifier."""
import csv
import os
import sys
import time
from collections import deque

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "servidor"))
from extrator_pontos import ExtratorPontos


DURACAO_GESTO_SEGUNDOS = 1.5
QUADROS_HISTORICO = 10


def coletar():
    extrator = ExtratorPontos()
    caminho_csv = os.path.join(os.path.dirname(__file__), "dataset_libras_real.csv")
    sinal = input("Digite o rotulo do sinal (ex: OI, SIM, A): ").strip().upper()
    if not sinal:
        print("Rotulo invalido.")
        return

    quantidade = int(input("Quantos exemplos gravar? (recomendado: 30 ou mais): ") or "30")
    if quantidade < 2:
        print("Grave pelo menos 2 exemplos por rotulo.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    existe_arquivo = os.path.exists(caminho_csv)
    cabecalho = ["rotulo"] + [f"coord_{i}" for i in range(204)]
    sequencias = deque(maxlen=QUADROS_HISTORICO)
    gravadas = 0

    print("Enquadre cabeca, bracos e maos. Pressione ESPACO para iniciar cada exemplo.")
    print("Execute o sinal durante 1,5 s; mantenha o quadro visivel. Pressione Q para sair.\n")

    with open(caminho_csv, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        if not existe_arquivo:
            escritor.writerow(cabecalho)

        while gravadas < quantidade:
            sucesso, quadro = camera.read()
            if not sucesso:
                break

            resultado = extrator.extrair_marcos(quadro)
            pronto = bool(resultado and resultado["tem_pose"] and resultado["tem_mao"])
            instrucao = "Cabeca e mao detectadas - ESPACO grava" if pronto else "Enquadre cabeca, bracos e pelo menos uma mao"
            cor = (0, 220, 0) if pronto else (0, 0, 255)
            cv2.putText(quadro, instrucao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)
            cv2.putText(quadro, f"{sinal}: {gravadas}/{quantidade}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Coleta de exemplos reais de Libras", quadro)
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
                if amostra is not None and amostra["tem_pose"] and amostra["tem_mao"]:
                    sequencias.append(amostra["vetor"])

                cv2.putText(quadro_gesto, f"Gravando {sinal}...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
                cv2.imshow("Coleta de exemplos reais de Libras", quadro_gesto)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sair = True
                    break

            if sair:
                break

            if len(sequencias) < 2:
                print("Poucos quadros validos. Ajuste o enquadramento e tente novamente.")
                continue

            quadro_final = sequencias[-1]
            deslocamento = quadro_final - sequencias[0]
            escritor.writerow([sinal] + quadro_final.tolist() + deslocamento.tolist())
            arquivo.flush()
            gravadas += 1
            print(f"Exemplo real {gravadas}/{quantidade} salvo para {sinal} ({len(sequencias)} quadros).")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nColeta finalizada. Dados salvos em: {caminho_csv}")
    print("Repita para cada rotulo e treine com: python modelo/treinar_modelo.py")


if __name__ == "__main__":
    coletar()
