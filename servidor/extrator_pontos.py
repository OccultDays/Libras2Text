import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
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
        self._garantir_modelos()

        opcoes_mao = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.caminho_modelo_mao),
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.detector_mao = vision.HandLandmarker.create_from_options(opcoes_mao)

        opcoes_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.caminho_modelo_pose),
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.4
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
        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)

        resultado_pose = self.detector_pose.detect(imagem_mp)
        resultado_mao = self.detector_mao.detect(imagem_mp)

        tem_pose = bool(resultado_pose.pose_landmarks)
        tem_mao = bool(resultado_mao.hand_landmarks)

        if not tem_pose and not tem_mao:
            return None

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

        # 3. Mao (21 marcos dos dedos)
        pontos_mao = np.zeros((21, 3))
        if tem_mao:
            marcos_mao = resultado_mao.hand_landmarks[0]
            for i, p in enumerate(marcos_mao):
                pontos_mao[i] = [p.x, p.y, p.z]
        elif tem_pose:
            # Fallback usando o pulso detectado na pose caso a mao nao seja segmentada individualmente
            pulso_dir = pontos_bracos[5]
            pontos_mao[:] = pulso_dir

        todos_pontos = np.vstack([pontos_cabeca, pontos_bracos, pontos_mao])
        return self.normalizar_coordenadas(todos_pontos, tem_pose, tem_mao)

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
