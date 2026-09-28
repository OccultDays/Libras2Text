import os
import sys
import csv
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "servidor"))
from extrator_pontos import ExtratorPontos

def coletar():
    extrator = ExtratorPontos()
    caminho_csv = os.path.join(os.path.dirname(__file__), "dataset_libras.csv")

    sinal = input("Digite o nome do sinal ou palavra para gravar: ").strip().upper()
    if not sinal:
        print("Sinal invalido.")
        return

    quantidade = int(input("Quantidade de amostras para coletar (ex: 50): ") or "50")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    amostras_coletadas = 0
    print("\nPosicione-se de modo que cabeca, bracos e maos fiquem visiveis.")
    print("Pressione 'ESPACO' para gravar um frame ou 'Q' para sair.\n")

    existe_arquivo = os.path.exists(caminho_csv)

    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if not existe_arquivo:
            cabecalho = ["rotulo"] + [f"coord_{i}" for i in range(102)]
            escritor.writerow(cabecalho)

        while amostras_coletadas < quantidade:
            sucesso, frame = cap.read()
            if not sucesso:
                break

            marcos = extrator.extrair_marcos(frame)
            status_texto = f"Sinal: {sinal} | Coletadas: {amostras_coletadas}/{quantidade}"

            if marcos is not None:
                cv2.putText(frame, "Corpo detectado! Pressione ESPACO", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Posicione cabeca e maos no quadro", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, status_texto, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Coletor de Dados de Libras (Cabeca, Bracos e Maos)", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord(" "):
                if marcos is not None:
                    escritor.writerow([sinal] + list(marcos))
                    f.flush()
                    amostras_coletadas += 1
                    print(f"Amostra {amostras_coletadas}/{quantidade} gravada!")
            elif tecla == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Coleta finalizada!")

if __name__ == "__main__":
    coletar()
