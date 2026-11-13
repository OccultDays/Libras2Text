import math
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, float(valor)))


def calcular_vetor_quadro(pontos_mao_3d, largura_quadro=480, altura_quadro=640, pontos_pose=None, outra_mao_3d=None):
    if pontos_mao_3d is None or len(pontos_mao_3d) != 21:
        return None

    maior_lado = float(max(largura_quadro, altura_quadro, 1))
    escala_x = float(largura_quadro) / maior_lado
    escala_y = float(altura_quadro) / maior_lado

    mao = np.zeros((21, 3), dtype=np.float64)
    for i, p in enumerate(pontos_mao_3d):
        mao[i, 0] = float(p[0]) * escala_x
        mao[i, 1] = float(p[1]) * escala_y
        mao[i, 2] = 0.0

    if not np.isfinite(mao).all():
        return None

    pulso = mao[0].copy()
    pontos_ref = [
        (float(mao[0, 0]), float(mao[0, 1])),
        (float(mao[8, 0]), float(mao[8, 1])),
        (float(mao[20, 0]), float(mao[20, 1])),
        (float(mao[9, 0]), float(mao[9, 1])),
    ]
    relativos = mao - pulso
    tamanho_palma = float(
        np.linalg.norm(relativos[5])
        + np.linalg.norm(relativos[9])
        + np.linalg.norm(relativos[13])
        + np.linalg.norm(relativos[17])
    ) / 4.0
    escala_mao = tamanho_palma / 0.45
    if escala_mao < 0.02:
        return None

    h = relativos / escala_mao
    det_palma = h[5, 0] * h[17, 1] - h[5, 1] * h[17, 0]
    if det_palma < 0.0:
        h[:, 0] *= -1.0

    valores = h.flatten().tolist()

    pontas = [4, 8, 12, 16, 20]
    bases = [2, 5, 9, 13, 17]
    cadeias = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    for p in pontas:
        valores.append(float(np.linalg.norm(h[p])))

    for b, p in zip(bases, pontas):
        valores.append(float(np.linalg.norm(h[p] - h[b])))

    for a, b, c, d in cadeias:
        v1 = h[b] - h[a]
        v2 = h[d] - h[c]
        norma = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        valores.append(float(np.dot(v1, v2) / norma))

    for p in [8, 12, 16, 20]:
        valores.append(float(np.linalg.norm(h[4] - h[p])))

    valores.append(float(np.linalg.norm(h[8] - h[12])))
    valores.append(float(np.linalg.norm(h[12] - h[16])))
    valores.append(float(np.linalg.norm(h[16] - h[20])))

    for b in [5, 9, 13]:
        valores.append(float(np.linalg.norm(h[4] - h[b])))

    if pontos_pose is not None and len(pontos_pose) >= 9:
        nariz_x = float(pontos_pose[0][0]) * escala_x
        nariz_y = float(pontos_pose[0][1]) * escala_y
        ombro_esq_x = float(pontos_pose[7][0]) * escala_x
        ombro_esq_y = float(pontos_pose[7][1]) * escala_y
        ombro_dir_x = float(pontos_pose[8][0]) * escala_x
        ombro_dir_y = float(pontos_pose[8][1]) * escala_y
        largura_ombros = max(math.hypot(ombro_esq_x - ombro_dir_x, ombro_esq_y - ombro_dir_y), 0.12)

        pos_pulso_y = _limitar((pulso[1] - nariz_y) / largura_ombros, -2.0, 2.5)
        pos_indicador_y = _limitar((mao[8, 1] - nariz_y) / largura_ombros, -2.0, 2.5)
        dist_centro_x = _limitar(abs(pulso[0] - nariz_x) / largura_ombros, 0.0, 2.5)
        tem_corpo = 1.0
    else:
        pos_pulso_y = _limitar((float(pontos_mao_3d[0][1]) - 0.30) * 2.0, -2.0, 2.5)
        pos_indicador_y = _limitar((float(pontos_mao_3d[8][1]) - 0.30) * 2.0, -2.0, 2.5)
        dist_centro_x = _limitar(abs(float(pontos_mao_3d[0][0]) - 0.50) * 2.5, 0.0, 2.5)
        tem_corpo = 0.0

    valores.extend([pos_pulso_y, pos_indicador_y, dist_centro_x, tem_corpo])

    tem_outra_mao = 0.0
    dist_pulsos = 4.0
    dist_pontas_ind = 4.0
    dist_pontas_med = 4.0
    dif_altura_pulsos = 0.0
    cos_palmas = 0.0

    if outra_mao_3d is not None and len(outra_mao_3d) == 21:
        outra = np.zeros((21, 3), dtype=np.float64)
        for i, p in enumerate(outra_mao_3d):
            outra[i, 0] = float(p[0]) * escala_x
            outra[i, 1] = float(p[1]) * escala_y
            outra[i, 2] = 0.0
        if np.isfinite(outra).all():
            tem_outra_mao = 1.0
            dist_pulsos = _limitar(math.hypot(mao[0, 0] - outra[0, 0], mao[0, 1] - outra[0, 1]) / escala_mao, 0.0, 4.0)
            dist_pontas_ind = _limitar(math.hypot(mao[8, 0] - outra[8, 0], mao[8, 1] - outra[8, 1]) / escala_mao, 0.0, 4.0)
            dist_pontas_med = _limitar(math.hypot(mao[12, 0] - outra[12, 0], mao[12, 1] - outra[12, 1]) / escala_mao, 0.0, 4.0)
            dif_altura_pulsos = _limitar((outra[0, 1] - mao[0, 1]) / escala_mao, -3.0, 3.0)
            dir_a_x = mao[9, 0] - mao[0, 0]
            dir_a_y = mao[9, 1] - mao[0, 1]
            dir_b_x = outra[9, 0] - outra[0, 0]
            dir_b_y = outra[9, 1] - outra[0, 1]
            norma_ab = math.hypot(dir_a_x, dir_a_y) * math.hypot(dir_b_x, dir_b_y) + 1e-6
            cos_palmas = _limitar((dir_a_x * dir_b_x + dir_a_y * dir_b_y) / norma_ab, -1.0, 1.0)

    valores.extend([
        tem_outra_mao,
        dist_pulsos,
        dist_pontas_ind,
        dist_pontas_med,
        dif_altura_pulsos,
        cos_palmas,
    ])

    return {
        "vetor": np.array(valores, dtype=np.float64),
        "pontos_ref": pontos_ref,
        "centro": (float(pontos_mao_3d[0][0]), float(pontos_mao_3d[0][1])),
        "escala_mao": escala_mao,
        "instante": time.monotonic(),
    }


def montar_vetor_temporal(historico):
    atual = historico[-1]
    primeiro = historico[0]
    escala = max(float(atual["escala_mao"]), 0.05)

    max_abs_dx = 0.0
    max_amp_x = 0.0
    max_amp_y = 0.0
    max_energia_x = 0.0
    max_energia_y = 0.0

    for r in range(4):
        xs = [float(item["pontos_ref"][r][0]) for item in historico]
        ys = [float(item["pontos_ref"][r][1]) for item in historico]

        abs_dx_r = abs(xs[-1] - xs[0]) / escala
        amp_x_r = (max(xs) - min(xs)) / escala
        amp_y_r = (max(ys) - min(ys)) / escala

        energia_rx = 0.0
        energia_ry = 0.0
        for k in range(1, len(historico)):
            energia_rx += abs(xs[k] - xs[k - 1]) / escala
            energia_ry += abs(ys[k] - ys[k - 1]) / escala

        if abs_dx_r > max_abs_dx:
            max_abs_dx = abs_dx_r
        if amp_x_r > max_amp_x:
            max_amp_x = amp_x_r
        if amp_y_r > max_amp_y:
            max_amp_y = amp_y_r
        if energia_rx > max_energia_x:
            max_energia_x = energia_rx
        if energia_ry > max_energia_y:
            max_energia_y = energia_ry

    dy_pulso = (float(atual["pontos_ref"][0][1]) - float(primeiro["pontos_ref"][0][1])) / escala
    dy_base = (float(atual["pontos_ref"][3][1]) - float(primeiro["pontos_ref"][3][1])) / escala
    dy_bruto = dy_pulso if abs(dy_pulso) >= abs(dy_base) else dy_base

    abs_dx = _limitar(max_abs_dx, 0.0, 3.0)
    dy = _limitar(dy_bruto, -3.0, 3.0)
    amp_x = _limitar(max_amp_x, 0.0, 4.0)
    amp_y = _limitar(max_amp_y, 0.0, 4.0)
    energia_x = min(max_energia_x, 4.0)
    energia_y = min(max_energia_y, 4.0)
    energia_total = min(math.hypot(energia_x, energia_y), 5.0)

    vetor_base = atual["vetor"].copy()
    n_hist = len(historico)
    if len(vetor_base) >= 98 and vetor_base[92] < 0.5 and n_hist >= 2:
        for idx_ant in range(n_hist - 2, max(-1, n_hist - 4), -1):
            anterior = historico[idx_ant]
            if len(anterior["vetor"]) >= 98 and anterior["vetor"][92] > 0.5:
                vetor_base[92:98] = anterior["vetor"][92:98]
                break

    delta_dedos = (atual["vetor"][63:68] - primeiro["vetor"][63:68]).tolist()
    movimento = np.array(
        [abs_dx, dy, amp_x, amp_y, energia_x, energia_y, energia_total] + delta_dedos,
        dtype=np.float64,
    )
    return np.concatenate([vetor_base, movimento])


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
        self._contador_pose = 0
        self._ultima_pose = []
        self._instante_ultima_pose = 0.0
        self._garantir_modelos()

        with open(self.caminho_modelo_mao, "rb") as arquivo_mao:
            conteudo_mao = arquivo_mao.read()

        opcoes_mao = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=conteudo_mao),
            num_hands=2,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector_mao = vision.HandLandmarker.create_from_options(opcoes_mao)

        with open(self.caminho_modelo_pose, "rb") as arquivo_pose:
            conteudo_pose = arquivo_pose.read()

        opcoes_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=conteudo_pose),
            num_poses=1,
            min_pose_detection_confidence=0.45,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector_pose = vision.PoseLandmarker.create_from_options(opcoes_pose)

    def _garantir_modelos(self):
        pasta = os.path.dirname(self.caminho_modelo_mao)
        os.makedirs(pasta, exist_ok=True)

        caminho_mao_app = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            "aplicativo",
            "modules",
            "libras-native",
            "android",
            "src",
            "main",
            "assets",
            "marcador_maos.task",
        ))
        caminho_pose_app = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            "aplicativo",
            "modules",
            "libras-native",
            "android",
            "src",
            "main",
            "assets",
            "marcador_pose.task",
        ))

        if not os.path.exists(self.caminho_modelo_mao):
            if os.path.exists(caminho_mao_app):
                with open(caminho_mao_app, "rb") as orig, open(self.caminho_modelo_mao, "wb") as dest:
                    dest.write(orig.read())
            else:
                url_mao = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url_mao, self.caminho_modelo_mao)

        if not os.path.exists(self.caminho_modelo_pose):
            if os.path.exists(caminho_pose_app):
                with open(caminho_pose_app, "rb") as orig, open(self.caminho_modelo_pose, "wb") as dest:
                    dest.write(orig.read())
            else:
                url_pose = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url_pose, self.caminho_modelo_pose)

    def extrair_marcos(self, imagem_bgr):
        altura, largura = imagem_bgr.shape[:2]
        maior_lado = max(altura, largura)
        if maior_lado > 480:
            escala = 480 / maior_lado
            largura = round(largura * escala)
            altura = round(altura * escala)
            imagem_bgr = cv2.resize(imagem_bgr, (largura, altura), interpolation=cv2.INTER_AREA)

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)

        resultado_mao = self.detector_mao.detect(imagem_mp)
        tem_mao = bool(resultado_mao.hand_landmarks)

        if not tem_mao:
            return {
                "quadro": None,
                "quadros_maos": [],
                "pontos_pose": [],
                "pontos_mao": [],
                "pontos_mao_3d": [],
                "pontos_maos": [],
                "pontos_maos_3d": [],
                "largura_quadro": largura,
                "altura_quadro": altura,
                "tem_pose": False,
                "tem_mao": False,
            }

        agora = time.monotonic()
        atualizar_pose = (
            self._contador_pose % 4 == 0
            or (agora - self._instante_ultima_pose) > 1.0
        )
        self._contador_pose += 1

        if atualizar_pose:
            resultado_pose = self.detector_pose.detect(imagem_mp)
            self._instante_ultima_pose = agora
            if resultado_pose.pose_landmarks:
                marcos_pose = resultado_pose.pose_landmarks[0]
                indices_cabeca = [0, 2, 5, 7, 8, 9, 10]
                indices_bracos = [11, 12, 13, 14, 15, 16]
                self._ultima_pose = [
                    [round(marcos_pose[idx].x, 4), round(marcos_pose[idx].y, 4)]
                    for idx in indices_cabeca + indices_bracos
                ]
            else:
                self._ultima_pose = []

        pontos_brutos_pose = self._ultima_pose
        tem_pose = bool(pontos_brutos_pose)

        maos_ordenadas = sorted(resultado_mao.hand_landmarks, key=lambda m: m[0].y)
        pontos_brutos_maos = []
        pontos_maos_3d = []
        quadros_maos = []

        for marcos in maos_ordenadas:
            brutos_2d = [[round(p.x, 4), round(p.y, 4)] for p in marcos]
            brutos_3d = [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in marcos]
            pontos_brutos_maos.append(brutos_2d)
            pontos_maos_3d.append(brutos_3d)

        for idx, brutos_3d in enumerate(pontos_maos_3d):
            outra_3d = pontos_maos_3d[1 - idx] if len(pontos_maos_3d) == 2 else None
            q = calcular_vetor_quadro(brutos_3d, largura, altura, pontos_brutos_pose, outra_3d)
            if q is not None:
                quadros_maos.append(q)

        quadro_principal = quadros_maos[0] if quadros_maos else None
        pontos_brutos_mao = pontos_brutos_maos[0] if pontos_brutos_maos else []
        pontos_mao_3d = pontos_maos_3d[0] if pontos_maos_3d else []

        return {
            "quadro": quadro_principal,
            "quadros_maos": quadros_maos,
            "pontos_pose": pontos_brutos_pose,
            "pontos_mao": pontos_brutos_mao,
            "pontos_mao_3d": pontos_mao_3d,
            "pontos_maos": pontos_brutos_maos,
            "pontos_maos_3d": pontos_maos_3d,
            "largura_quadro": largura,
            "altura_quadro": altura,
            "tem_pose": tem_pose,
            "tem_mao": quadro_principal is not None,
        }
