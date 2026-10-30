import base64
import cv2
import numpy as np

class ProcessadorImagem:
    @staticmethod
    def decodificar_bytes(bytes_imagem):
        if not bytes_imagem:
            return None

        try:
            vetor_np = np.frombuffer(bytes_imagem, dtype=np.uint8)
            return cv2.imdecode(vetor_np, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def decodificar_base64(texto_base64):
        if not texto_base64:
            return None

        if "," in texto_base64:
            texto_base64 = texto_base64.split(",", 1)[1]

        try:
            bytes_imagem = base64.b64decode(texto_base64)
            return ProcessadorImagem.decodificar_bytes(bytes_imagem)
        except Exception:
            return None
