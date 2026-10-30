import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

class ExtratorPontos:
    def __init__(self, caminho_modelo_mao=None, caminho_modelo_pose=None):
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        pasta_modelos = os.path.join(diretorio_atual, "modelos_ia")

        if caminho_modelo_mao is None:
            caminho_modelo_mao = os.path.join(pasta_modelos, "marcador_maos.task")
        if caminho_modelo_pose is None:
            caminho_modelo_pose = os.path.join(pasta_modelos, "marcador_pose.task")

        self.caminho_modelo_mao = caminho_modelo_mao
        self.caminho_modelo_pose = caminho_modelo_pose
        self._ultimo_timestamp_ms = 0
        self._garantir_modelos()

        with open(self.caminho_modelo_mao, "rb") as arquivo_mao:
            conteudo_mao = arquivo_mao.read()

        opcoes_mao = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=conteudo_mao),
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector_mao = vision.HandLandmarker.create_from_options(opcoes_mao)

        with open(self.caminho_modelo_pose, "rb") as arquivo_pose:
            conteudo_pose = arquivo_pose.read()

        opcoes_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=conteudo_pose),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.55,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector_pose = vision.PoseLandmarker.create_from_options(opcoes_pose)

    def _garantir_modelos(self):
        pasta = os.path.dirname(self.caminho_modelo_mao)
        os.makedirs(pasta, exist_ok=True)

        if not os.path.exists(self.caminho_modelo_mao):
            url_mao = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url_mao, self.caminho_modelo_mao)

        if not os.path.exists(self.caminho_modelo_pose):
            url_pose = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url_pose, self.caminho_modelo_pose)

    def extrair_marcos(self, imagem_bgr):
        # Landmark quality is still good at this size, while inference is much
        # faster than running both models on the phone's full-resolution photo.
        altura, largura = imagem_bgr.shape[:2]
        maior_lado = max(altura, largura)
        if maior_lado > 720:
            escala = 720 / maior_lado
            imagem_bgr = cv2.resize(
                imagem_bgr,
                (round(largura * escala), round(altura * escala)),
                interpolation=cv2.INTER_AREA
            )

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)

        # VIDEO mode retains landmarks between consecutive frames, improving
        # stability when the signer moves quickly.
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self._ultimo_timestamp_ms + 1)
        self._ultimo_timestamp_ms = timestamp_ms
        resultado_pose = self.detector_pose.detect_for_video(imagem_mp, timestamp_ms)
        resultado_mao = self.detector_mao.detect_for_video(imagem_mp, timestamp_ms)

        tem_pose = bool(resultado_pose.pose_landmarks)
        tem_mao = bool(resultado_mao.hand_landmarks)

        if not tem_pose and not tem_mao:
            return None

        pontos_brutos_pose = []
        pontos_brutos_mao = []
        pontos_brutos_maos = []

        # 1. Cabeca (7 marcos da pose: nariz, olhos, orelhas, boca)
        indices_cabeca = [0, 2, 5, 7, 8, 9, 10]
        pontos_cabeca = np.zeros((7, 3))
        if tem_pose:
            marcos_pose = resultado_pose.pose_landmarks[0]
            for i, idx in enumerate(indices_cabeca):
                p = marcos_pose[idx]
                pontos_cabeca[i] = [p.x, p.y, p.z]

        # 2. Bracos e Tronco (6 marcos da pose: ombros, cotovelos, pulsos)
        indices_bracos = [11, 12, 13, 14, 15, 16]
        pontos_bracos = np.zeros((6, 3))
        if tem_pose:
            marcos_pose = resultado_pose.pose_landmarks[0]
            for i, idx in enumerate(indices_bracos):
                p = marcos_pose[idx]
                pontos_bracos[i] = [p.x, p.y, p.z]
            for idx in (indices_cabeca + indices_bracos):
                p = marcos_pose[idx]
                pontos_brutos_pose.append([round(p.x, 4), round(p.y, 4)])

        # 3. Mao (21 marcos dos dedos)
        pontos_mao = np.zeros((21, 3))
        if tem_mao:
            # The existing classifier was trained on one 21-point hand. Keep
            # that feature shape and choose the detected hand nearest a pose
            # wrist; expose every detected hand separately for the overlay.
            maos = resultado_mao.hand_landmarks
            for marcos in maos:
                pontos_brutos_maos.append([[round(p.x, 4), round(p.y, 4)] for p in marcos])
            if tem_pose:
                pulsos = [pontos_bracos[4], pontos_bracos[5]]
                marcos_mao = min(maos, key=lambda m: min(
                    np.linalg.norm(np.array([m[0].x, m[0].y]) - pulso[:2]) for pulso in pulsos
                ))
            else:
                marcos_mao = maos[0]
            for i, p in enumerate(marcos_mao):
                pontos_mao[i] = [p.x, p.y, p.z]
                pontos_brutos_mao.append([round(p.x, 4), round(p.y, 4)])
        elif tem_pose:
            pulso_dir = pontos_bracos[5]
            pontos_mao[:] = pulso_dir

        todos_pontos = np.vstack([pontos_cabeca, pontos_bracos, pontos_mao])
        marcos_normalizados = self.normalizar_coordenadas(todos_pontos, tem_pose, tem_mao)

        return {
            "vetor": marcos_normalizados,
            "pontos_pose": pontos_brutos_pose,
            "pontos_mao": pontos_brutos_mao,
            "pontos_maos": pontos_brutos_maos,
            "tem_pose": tem_pose,
            "tem_mao": tem_mao
        }

    def normalizar_coordenadas(self, coordenadas, tem_pose, tem_mao):
        # Ponto de referencia: centro dos ombros (se pose disponivel) ou pulso da mao
        if tem_pose:
            ombro_esq = coordenadas[7]
            ombro_dir = coordenadas[8]
            centro = (ombro_esq + ombro_dir) / 2.0
        else:
            centro = coordenadas[13]

        ajustado = coordenadas - centro
        distancias = np.linalg.norm(ajustado, axis=1)
        escala_maxima = np.max(distancias)
        if escala_maxima > 0:
            ajustado = ajustado / escala_maxima

        return ajustado.flatten()
