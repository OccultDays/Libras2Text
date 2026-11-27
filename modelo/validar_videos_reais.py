import argparse
from collections import Counter, deque
import csv
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DIRETORIO_MODELO = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_MODELO)
PASTA_VIDEOS = os.path.join(DIRETORIO_MODELO, "videos_reais")
CAMINHO_MODELO_MOBILE = os.path.join(RAIZ_PROJETO, "aplicativo", "assets", "modelo_libras_mobile.json")
CAMINHO_DATASET = os.path.join(DIRETORIO_MODELO, "dataset_libras.csv")
CAMINHO_DATASET_VIDEOS = os.path.join(DIRETORIO_MODELO, "dataset_videos_reais.csv")

sys.path.append(os.path.join(RAIZ_PROJETO, "servidor"))
from extrator_pontos import (
    ExtratorPontos,
    calcular_vetor_quadro,
    filtrar_maos_ativas,
    montar_vetor_temporal,
)

JANELA_TEMPORAL_SEG = 1.1
INTERVALO_AMOSTRA_SEG = 0.11
JANELA_COMPOSICAO_SEG = 2.8

MAPA_SUBCLASSES = {
    "DIA_ROSTO": "DIA",
    "DIA_SOL": "DIA",
}

# Sinais compostos avaliados apenas por composicao (nao entram no treino direto)
SINAIS_COMPOSTOS_TESTE = {"BOM DIA"}

INES_BASE = "https://www.ines.gov.br/dicionario-de-libras/public/media/palavras/videos/"
VLIBRASIL_BASE = "https://huggingface.co/datasets/SchberGg/v-librasil-raw/resolve/main/videos/"

# Catalogo de videos reais verificados visualmente (INES, V-LIBRASIL/UFPE, UFV, SpreadTheSign)
CATALOGO_OFICIAL = {
    "BOM DIA": [
        ("ufv_bom_dia.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/01/01446-Bom-Dia.mp4"),
    ],
    "BOM": [
        ("ines_bom1.mp4", INES_BASE + "bom1Sm_Prog001.mp4"),
        ("sts_boa.mp4", "https://media.spreadthesign.com/video/mp4/14/147976.mp4"),
        ("vlibrasil_bem_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Bem_Articulador1.mp4")),
        ("vlibrasil_bem_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Bem_Articulador3.mp4")),
    ],
    "DIA_ROSTO": [
        ("ines_dia1.mp4", INES_BASE + "dia1Sm_Prog001.mp4"),
        ("vlibrasil_dia_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Dia_Articulador1.mp4")),
        ("vlibrasil_dia_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Dia_Articulador2.mp4")),
        ("vlibrasil_dia_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Dia_Articulador3.mp4")),
        ("ufv_dia.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/04/01516-Dia.mp4"),
    ],
    "DIA_SOL": [
        ("ines_dia3.mp4", INES_BASE + "dia3Sm_Prog001.mp4"),
        ("sts_dia.mp4", "https://media.spreadthesign.com/video/mp4/14/386820.mp4"),
    ],
    "CASA": [
        ("ines_casa.mp4", INES_BASE + "casaSm_Prog001.mp4"),
        ("vlibrasil_casa_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Casa_Articulador1.mp4")),
        ("vlibrasil_casa_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Casa_Articulador2.mp4")),
        ("vlibrasil_casa_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Casa_Articulador3.mp4")),
        ("ufv_casa.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/09/casa.mp4"),
        ("sts_casa.mp4", "https://media.spreadthesign.com/video/mp4/14/550420.mp4"),
    ],
    "POR FAVOR": [
        ("ines_por_favor.mp4", INES_BASE + "por_favorSm_Prog001.mp4"),
        ("vlibrasil_por_favor_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Por favor_Articulador1.mp4")),
        ("vlibrasil_por_favor_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Por favor_Articulador2.mp4")),
        ("vlibrasil_por_favor_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Por favor_Articulador3.mp4")),
        ("ufv_por_favor.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/01/01450-Por-favor-com-licen%C3%A7a.mp4"),
        ("sts_por_favor.mp4", "https://media.spreadthesign.com/video/mp4/14/417949.mp4"),
    ],
    "OBRIGADO": [
        ("ines_obrigado.mp4", INES_BASE + "obrigado1Sm_Prog001.mp4"),
        ("ines_agradecer.mp4", INES_BASE + "agradecerSm_Prog001.mp4"),
        ("vlibrasil_obrigado_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Obrigado_Articulador1.mp4")),
        ("vlibrasil_obrigado_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Obrigado_Articulador2.mp4")),
        ("vlibrasil_obrigado_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Obrigado_Articulador3.mp4")),
        ("ufv_obrigado.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/01/01448-Obrigado.mp4"),
        ("sts_obrigado.mp4", "https://media.spreadthesign.com/video/mp4/14/560823.mp4"),
    ],
    "OI": [
        ("ines_oi.mp4", INES_BASE + "oiSm_Prog001.mp4"),
        ("ines_ola.mp4", INES_BASE + "olaSm_Prog001.mp4"),
        ("vlibrasil_oi_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Oi_Articulador1.mp4")),
        ("vlibrasil_oi_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Oi_Articulador2.mp4")),
        ("vlibrasil_oi_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Oi_Articulador3.mp4")),
        ("sts_ola.mp4", "https://media.spreadthesign.com/video/mp4/14/555228.mp4"),
    ],
    "SIM": [
        ("ines_sim.mp4", INES_BASE + "simSm_Prog001.mp4"),
        ("vlibrasil_sim_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Sim_Articulador1.mp4")),
        ("vlibrasil_sim_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Sim_Articulador2.mp4")),
        ("vlibrasil_sim_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Sim_Articulador3.mp4")),
    ],
    "NAO": [
        ("ines_nao1.mp4", INES_BASE + "nao1Sm_Prog001.mp4"),
        ("ines_nao2.mp4", INES_BASE + "nao2Sm_Prog001.mp4"),
        ("vlibrasil_nao_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Não_Articulador1.mp4")),
        ("vlibrasil_nao_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Não_Articulador2.mp4")),
        ("vlibrasil_nao_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Não_Articulador3.mp4")),
        ("sts_nao_1.mp4", "https://media.spreadthesign.com/video/mp4/14/417934.mp4"),
    ],
    "EU": [
        ("ines_eu.mp4", INES_BASE + "euSm_Prog001.mp4"),
    ],
    "VOCE": [
        ("ines_voce.mp4", INES_BASE + "voceSm_Prog001.mp4"),
    ],
    "AJUDA": [
        ("ines_ajudar1.mp4", INES_BASE + "ajudar1Sm_Prog001.mp4"),
        ("ines_ajudar2.mp4", INES_BASE + "ajudar2Sm_Prog001.mp4"),
        ("vlibrasil_ajudar_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Ajudar_Articulador1.mp4")),
        ("vlibrasil_ajudar_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Ajudar_Articulador2.mp4")),
        ("vlibrasil_ajudar_3.mp4", VLIBRASIL_BASE + urllib.parse.quote("Ajudar_Articulador3.mp4")),
        ("ufv_ajudar.mp4", "https://sistemas.cead.ufv.br/capes/dicionario/wp-content/uploads/2018/05/709-ajudar.mp4"),
        ("sts_ajudar.mp4", "https://media.spreadthesign.com/video/mp4/14/486646.mp4"),
    ],
    "A": [
        ("ines_a.mp4", INES_BASE + "aSm_Prog001.mp4"),
        ("vlibrasil_a_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra A_Articulador1.mp4")),
        ("vlibrasil_a_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra A_Articulador2.mp4")),
    ],
    "B": [
        ("ines_b.mp4", INES_BASE + "bSm_Prog001.mp4"),
        ("vlibrasil_b_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra B_Articulador1.mp4")),
        ("vlibrasil_b_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra B_Articulador2.mp4")),
    ],
    "C": [("ines_c.mp4", INES_BASE + "cSm_Prog001.mp4")],
    "D": [
        ("ines_d.mp4", INES_BASE + "dSm_Prog001.mp4"),
        ("vlibrasil_d_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra D_Articulador1.mp4")),
        ("vlibrasil_d_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra D_Articulador2.mp4")),
    ],
    "E": [
        ("ines_e.mp4", INES_BASE + "eSm_Prog001.mp4"),
        ("vlibrasil_e_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra E_Articulador1.mp4")),
        ("vlibrasil_e_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra E_Articulador2.mp4")),
    ],
    "F": [
        ("ines_f.mp4", INES_BASE + "fSm_Prog001.mp4"),
        ("vlibrasil_f_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra F_Articulador1.mp4")),
        ("vlibrasil_f_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra F_Articulador2.mp4")),
    ],
    "I": [
        ("ines_i.mp4", INES_BASE + "iSm_Prog001.mp4"),
        ("vlibrasil_i_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra I_Articulador1.mp4")),
        ("vlibrasil_i_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra I_Articulador2.mp4")),
    ],
    "L": [
        ("ines_l.mp4", INES_BASE + "lSm_Prog001.mp4"),
        ("vlibrasil_l_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra L_Articulador1.mp4")),
        ("vlibrasil_l_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra L_Articulador2.mp4")),
    ],
    "O": [("ines_o.mp4", INES_BASE + "oSm_Prog001.mp4")],
    "U": [
        ("ines_u.mp4", INES_BASE + "uSm_Prog001.mp4"),
        ("vlibrasil_u_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra U_Articulador1.mp4")),
        ("vlibrasil_u_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra U_Articulador2.mp4")),
    ],
    "V": [
        ("ines_v.mp4", INES_BASE + "vSm_Prog001.mp4"),
        ("vlibrasil_v_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra V_Articulador1.mp4")),
        ("vlibrasil_v_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra V_Articulador2.mp4")),
    ],
    "W": [
        ("ines_w.mp4", INES_BASE + "wSm_Prog001.mp4"),
        ("vlibrasil_w_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra W_Articulador1.mp4")),
        ("vlibrasil_w_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra W_Articulador2.mp4")),
    ],
    "Y": [
        ("vlibrasil_y_1.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra Y_Articulador1.mp4")),
        ("vlibrasil_y_2.mp4", VLIBRASIL_BASE + urllib.parse.quote("Letra Y_Articulador2.mp4")),
    ],
}


def baixar_videos_oficiais(sinais_filtro=None):
    os.makedirs(PASTA_VIDEOS, exist_ok=True)
    for sinal, lista in CATALOGO_OFICIAL.items():
        if sinais_filtro and sinal not in sinais_filtro:
            continue
        pasta_sinal = os.path.join(PASTA_VIDEOS, sinal)
        os.makedirs(pasta_sinal, exist_ok=True)
        for nome_arq, url in lista:
            destino = os.path.join(pasta_sinal, nome_arq)
            if os.path.exists(destino) and os.path.getsize(destino) > 1000:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=25) as resp, open(destino, "wb") as out:
                    out.write(resp.read())
                print(f"[baixado] {sinal}/{nome_arq}")
            except Exception as erro:
                print(f"[falha] {sinal}/{nome_arq}: {erro}")


def carregar_modelo_mobile(caminho=CAMINHO_MODELO_MOBILE):
    with open(caminho, "r", encoding="utf-8") as arq:
        return json.load(arq)


def _agrupar_probabilidades(rotulos, probs):
    agrupadas = {}
    for rotulo, prob in zip(rotulos, probs):
        exib = MAPA_SUBCLASSES.get(str(rotulo), str(rotulo))
        agrupadas[exib] = agrupadas.get(exib, 0.0) + float(prob)
    ordenado = sorted(agrupadas.items(), key=lambda item: item[1], reverse=True)
    ranking = {k: v for k, v in ordenado[:5]}
    melhor_rot, melhor_prob = ordenado[0]
    if melhor_rot == "NEUTRO":
        return "", 0.0, ranking
    return melhor_rot, melhor_prob, ranking


def prever_modelo_mobile(modelo, caracteristicas):
    if len(caracteristicas) != modelo["quantidade_caracteristicas"]:
        return "", 0.0, {}
    rotulos = modelo["rotulos"]
    probs = np.zeros(len(rotulos), dtype=np.float64)
    for arvore in modelo["arvores"]:
        no = 0
        esq = arvore["esquerda"]
        dir_ = arvore["direita"]
        feat = arvore["caracteristica"]
        lim = arvore["limite"]
        while esq[no] != -1:
            if caracteristicas[feat[no]] <= lim[no]:
                no = esq[no]
            else:
                no = dir_[no]
        probs += np.array(arvore["probabilidades"][no], dtype=np.float64)
    probs /= max(len(modelo["arvores"]), 1)
    return _agrupar_probabilidades(rotulos, probs)


def prever_sklearn(modelo_rf, caracteristicas):
    probs = modelo_rf.predict_proba(np.asarray(caracteristicas, dtype=np.float64).reshape(1, -1))[0]
    return _agrupar_probabilidades(modelo_rf.classes_, probs)


def escolher_rotulo_votacao(votacoes):
    pesos_base = [0.60, 0.85, 1.30]
    pesos = pesos_base[-len(votacoes):]
    info = {}
    for (rotulo, conf), peso in zip(votacoes, pesos):
        if not rotulo:
            continue
        item = info.setdefault(rotulo, [0.0, 0.0, 0])
        item[0] += conf * peso
        item[1] += conf
        item[2] += 1
    if not info:
        return "", 0.0
    melhor_rotulo = max(info, key=lambda r: info[r][0])
    _, soma_conf, votos = info[melhor_rotulo]
    media_conf = soma_conf / votos
    aceito = (votos >= 2 and media_conf >= 0.38) or media_conf >= 0.50
    return (melhor_rotulo if aceito else ""), media_conf


def recortar_retrato_celular(quadro_bgr):
    alt, larg = quadro_bgr.shape[:2]
    larg_alvo = int(round(alt * 0.75))
    if larg > larg_alvo:
        inicio_x = (larg - larg_alvo) // 2
        quadro_bgr = quadro_bgr[:, inicio_x:inicio_x + larg_alvo]
    return cv2.resize(quadro_bgr, (240, 320), interpolation=cv2.INTER_AREA)


def extrair_sequencia_video(extrator, caminho_video, recorte_celular=False, espelhar=False, coletar_neutro=False):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 120.0:
        fps = 30.0

    extrator._contador_pose = 0
    extrator._ultima_pose = []
    extrator._instante_ultima_pose = 0.0

    trilhas = []
    falhas_mao = 0
    total_quadros = 0
    quadros_0_maos = 0
    quadros_1_mao = 0
    quadros_2_maos = 0
    quadros_com_pose = 0
    janelas = []
    vetores_neutros = []

    ultimo_tempo_pose = -10.0
    passo_quadro = max(1, int(round(fps / 12.0)))

    idx_quadro = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx_quadro += 1
        if (idx_quadro - 1) % passo_quadro != 0:
            continue

        total_quadros += 1
        tempo_video = idx_quadro / fps

        if espelhar:
            frame = cv2.flip(frame, 1)
        if recorte_celular:
            frame = recortar_retrato_celular(frame)

        if tempo_video - ultimo_tempo_pose >= 0.25:
            extrator._contador_pose = 0
            ultimo_tempo_pose = tempo_video

        res = extrator.extrair_marcos(frame)
        maos_brutas = res.get("pontos_maos_3d") or []
        pontos_pose = res.get("pontos_pose") or []
        larg = res["largura_quadro"]
        alt = res["altura_quadro"]

        if pontos_pose:
            quadros_com_pose += 1

        maos_ativas = filtrar_maos_ativas(maos_brutas, larg, alt, pontos_pose if pontos_pose else None)

        # Coleta quadros reais de maos em repouso na base para treinar a classe NEUTRO
        if coletar_neutro and maos_brutas and not maos_ativas and len(vetores_neutros) < 4:
            outra_rep = maos_brutas[1] if len(maos_brutas) == 2 else None
            q_rep = calcular_vetor_quadro(maos_brutas[0], larg, alt, pontos_pose if pontos_pose else None, outra_rep)
            if q_rep is not None:
                vetores_neutros.append(montar_vetor_temporal([q_rep]))

        if len(maos_ativas) == 0:
            quadros_0_maos += 1
        elif len(maos_ativas) == 1:
            quadros_1_mao += 1
        else:
            quadros_2_maos += 1

        quadros_maos = []
        for idx, mao in enumerate(maos_ativas):
            outra = maos_ativas[1 - idx] if len(maos_ativas) == 2 else None
            q = calcular_vetor_quadro(mao, larg, alt, pontos_pose if pontos_pose else None, outra)
            if q is not None:
                q["instante"] = tempo_video
                quadros_maos.append(q)

        if not quadros_maos:
            falhas_mao += 1
            if falhas_mao >= 5:
                trilhas.clear()
                janelas.append({"vazio": True})
            continue

        falhas_mao = 0
        trilhas = [t for t in trilhas if (tempo_video - t["ultimo_instante"]) <= JANELA_TEMPORAL_SEG]
        limite_dist = 0.85 if (len(quadros_maos) == 1 and len(trilhas) == 1) else 0.65

        pares = []
        for iq, q in enumerate(quadros_maos):
            for it, t in enumerate(trilhas):
                dist = math.hypot(q["centro"][0] - t["centro"][0], q["centro"][1] - t["centro"][1])
                if dist <= limite_dist:
                    pares.append((dist, iq, it))
        pares.sort(key=lambda item: item[0])

        usados_q = set()
        usados_t = set()
        trilhas_ativas = []
        for _, iq, it in pares:
            if iq in usados_q or it in usados_t:
                continue
            usados_q.add(iq)
            usados_t.add(it)
            q = quadros_maos[iq]
            trilha = trilhas[it]
            while trilha["quadros"] and (q["instante"] - trilha["quadros"][0]["instante"]) > JANELA_TEMPORAL_SEG:
                trilha["quadros"].popleft()
            if len(trilha["quadros"]) >= 2 and (q["instante"] - trilha["quadros"][-2]["instante"]) < INTERVALO_AMOSTRA_SEG:
                trilha["quadros"][-1] = q
            else:
                trilha["quadros"].append(q)
            trilha["centro"] = q["centro"]
            trilha["ultimo_instante"] = q["instante"]
            trilhas_ativas.append(trilha)

        for iq, q in enumerate(quadros_maos):
            if iq in usados_q:
                continue
            nova = {"centro": q["centro"], "ultimo_instante": q["instante"], "quadros": deque([q], maxlen=9)}
            trilhas.append(nova)
            trilhas_ativas.append(nova)

        trilhas.sort(key=lambda t: t["ultimo_instante"], reverse=True)
        trilhas = trilhas[:2]

        opcoes_trilhas = []
        for trilha in trilhas_ativas:
            vetor = montar_vetor_temporal(list(trilha["quadros"]))
            opcoes_trilhas.append({
                "vetor": vetor,
                "centro_y": trilha["centro"][1],
                "quadros": len(trilha["quadros"]),
            })
        janelas.append({"vazio": False, "tempo": tempo_video, "trilhas": opcoes_trilhas})

    cap.release()
    return {
        "video": os.path.basename(caminho_video),
        "quadros": total_quadros,
        "q0": quadros_0_maos,
        "q1": quadros_1_mao,
        "q2": quadros_2_maos,
        "q_pose": quadros_com_pose,
        "janelas": janelas,
        "vetores_neutros": vetores_neutros,
    }


def classificar_sequencia(seq, fn_prever):
    votacoes = deque(maxlen=3)
    estava_em_movimento = False
    predicoes_janela = []
    vetores_extraidos = []
    acumulador_gesto = {}
    ultimo_comp = {"rotulo": "", "tempo": -10.0}
    soma_probs_classe = Counter()
    total_avaliacoes = 0

    for item in seq["janelas"]:
        if item.get("vazio"):
            votacoes.clear()
            estava_em_movimento = False
            continue

        tempo_atual = item.get("tempo", 0.0)
        trilhas_ativas = item["trilhas"]
        menor_y = min(t["centro_y"] for t in trilhas_ativas)
        maior_y = max(t["centro_y"] for t in trilhas_ativas)

        avaliacoes = []
        for t in trilhas_ativas:
            vetor = t["vetor"]
            rotulo, conf, ranking = fn_prever(vetor)
            em_mov = float(vetor[104]) >= 0.28 or max(float(vetor[100]), float(vetor[101])) >= 0.24
            prioridade = conf
            if len(trilhas_ativas) > 1:
                if (maior_y - menor_y) > 0.14 and t["centro_y"] <= menor_y + 0.02:
                    prioridade += 0.08
                if em_mov:
                    prioridade += 0.14
            avaliacoes.append({
                "rotulo": rotulo,
                "conf": conf,
                "prioridade": prioridade,
                "em_mov": em_mov,
                "vetor": vetor,
                "ranking": ranking,
            })

        if len(avaliacoes) == 2 and avaliacoes[0]["rotulo"] and avaliacoes[0]["rotulo"] == avaliacoes[1]["rotulo"]:
            escolhida = max(avaliacoes, key=lambda a: a["conf"])
            rot_bruto = escolhida["rotulo"]
            conf_bruta = min(1.0, escolhida["conf"] + 0.05)
            em_mov_atual = avaliacoes[0]["em_mov"] or avaliacoes[1]["em_mov"]
        else:
            escolhida = max(avaliacoes, key=lambda a: a["prioridade"])
            rot_bruto = escolhida["rotulo"]
            conf_bruta = escolhida["conf"]
            em_mov_atual = escolhida["em_mov"]

        vetores_extraidos.append(escolhida["vetor"])
        for cls, p in escolhida["ranking"].items():
            if cls != "NEUTRO":
                soma_probs_classe[cls] += p
        total_avaliacoes += 1

        if em_mov_atual and not estava_em_movimento:
            votacoes = deque([v for v in votacoes if v[0] == rot_bruto], maxlen=3)
        estava_em_movimento = em_mov_atual

        votacoes.append((rot_bruto, conf_bruta))
        pred_base, media_conf = escolher_rotulo_votacao(votacoes)
        pred_exib = pred_base

        if pred_exib == "BOM":
            ultimo_comp = {"rotulo": "BOM", "tempo": tempo_atual}
        elif pred_exib in ("DIA", "D") and ultimo_comp["rotulo"] == "BOM" and (tempo_atual - ultimo_comp["tempo"]) <= JANELA_COMPOSICAO_SEG:
            pred_exib = "BOM DIA"

        if pred_exib:
            predicoes_janela.append((pred_exib, media_conf))

        if pred_base and media_conf >= 0.36:
            reg = acumulador_gesto.setdefault(pred_base, [0.0, 0.0, 0.0, 0, tempo_atual, tempo_atual])
            reg[0] += media_conf
            reg[1] += media_conf
            if media_conf > reg[2]:
                reg[2] = media_conf
            reg[3] += 1
            reg[5] = tempo_atual

    melhor_rotulo = ""
    melhor_conf = 0.0
    reg_bom = acumulador_gesto.get("BOM")
    r_dia = acumulador_gesto.get("DIA")
    r_d = acumulador_gesto.get("D")
    if r_dia and r_d:
        reg_dia = [
            r_dia[0] + r_d[0],
            r_dia[1] + r_d[1],
            max(r_dia[2], r_d[2]),
            r_dia[3] + r_d[3],
            min(r_dia[4], r_d[4]),
            max(r_dia[5], r_d[5]),
        ]
    else:
        reg_dia = r_dia or r_d

    if (
        reg_bom
        and reg_dia
        and reg_bom[4] <= reg_dia[5]
        and (reg_bom[3] >= 2 or reg_bom[2] >= 0.42)
        and (reg_dia[3] >= 2 or reg_dia[2] >= 0.42)
    ):
        melhor_rotulo = "BOM DIA"
        tot_cont = reg_bom[3] + reg_dia[3]
        melhor_conf = max((reg_bom[1] + reg_dia[1]) / tot_cont, max(reg_bom[2], reg_dia[2]) * 0.9)
    else:
        melhor_soma = -1.0
        for rotulo, (soma, soma_conf, max_conf, contagem, _, _) in acumulador_gesto.items():
            if contagem < 2 and max_conf < 0.48:
                continue
            if soma > melhor_soma:
                melhor_soma = soma
                melhor_rotulo = rotulo
                melhor_conf = max(soma_conf / contagem, max_conf * 0.9)

    contagem_pred = Counter(p for p, _ in predicoes_janela)
    pred_majoritaria = contagem_pred.most_common(1)[0][0] if contagem_pred else ""
    top_probs = [
        (cls, round(val / max(total_avaliacoes, 1), 3))
        for cls, val in soma_probs_classe.most_common(4)
    ]

    return {
        "video": seq["video"],
        "quadros": seq["quadros"],
        "q0": seq["q0"],
        "q1": seq["q1"],
        "q2": seq["q2"],
        "q_pose": seq["q_pose"],
        "pred_gesto_25s": melhor_rotulo,
        "conf_gesto_25s": round(melhor_conf, 3),
        "pred_majoritaria": pred_majoritaria,
        "votos_janelas": dict(contagem_pred.most_common(4)),
        "top_probs": top_probs,
        "vetores": vetores_extraidos,
    }


def avaliar_video(extrator, modelo, caminho_video, recorte_celular=False, espelhar=False):
    seq = extrair_sequencia_video(extrator, caminho_video, recorte_celular=recorte_celular, espelhar=espelhar)
    if seq is None:
        return None
    return classificar_sequencia(seq, lambda v: prever_modelo_mobile(modelo, v))


def extrair_dataset_videos(sinais_filtro=None):
    videos_por_sinal = listar_videos_locais(sinais_filtro)
    if not videos_por_sinal:
        print("Nenhum video encontrado para extracao.")
        return

    extrator = ExtratorPontos()
    cabecalho = ["rotulo", "origem_video"] + [f"coord_{i}" for i in range(110)]
    linhas = [cabecalho]
    neutros_total = 0

    for sinal, lista in videos_por_sinal.items():
        if sinal in SINAIS_COMPOSTOS_TESTE:
            print(f"[teste cego] {sinal:12s}: mantido fora do treino (avaliado por composicao)")
            continue
        total_sinal = 0
        for caminho in lista:
            origem = f"{sinal}/{os.path.basename(caminho)}"
            for recorte, espelhar in ((False, False), (False, True), (True, False)):
                seq = extrair_sequencia_video(
                    extrator,
                    caminho,
                    recorte_celular=recorte,
                    espelhar=espelhar,
                    coletar_neutro=(not espelhar and not recorte),
                )
                if seq is None:
                    continue

                if seq["vetores_neutros"] and neutros_total < 180:
                    for vn in seq["vetores_neutros"][:2]:
                        linhas.append(["NEUTRO", origem] + [round(float(x), 6) for x in vn])
                        neutros_total += 1

                janelas_validas = [j for j in seq["janelas"] if not j.get("vazio") and j["trilhas"]]
                if not janelas_validas:
                    continue

                n_val = len(janelas_validas)
                ini = int(n_val * 0.12) if n_val >= 10 else 0
                fim = int(n_val * 0.90) if n_val >= 10 else n_val
                selecionadas = janelas_validas[ini:max(fim, ini + 1)]

                if len(selecionadas) > 14:
                    indices = np.linspace(0, len(selecionadas) - 1, 14, dtype=int)
                    selecionadas = [selecionadas[i] for i in indices]

                for j in selecionadas:
                    for t in j["trilhas"]:
                        if t["quadros"] < 2 and n_val >= 4:
                            continue
                        v = t["vetor"]
                        linhas.append([sinal, origem] + [round(float(x), 6) for x in v])
                        total_sinal += 1
        print(f"[extraido] {sinal:12s}: {total_sinal} janelas reais ({len(lista)} videos)")

    print(f"[extraido] {'NEUTRO':12s}: {neutros_total} janelas reais de repouso")
    with open(CAMINHO_DATASET_VIDEOS, "w", newline="", encoding="utf-8") as arq:
        csv.writer(arq).writerows(linhas)
    print(f"Dataset de videos reais salvo em {CAMINHO_DATASET_VIDEOS} ({len(linhas) - 1} amostras).")


def carregar_dataset_sintetico(caminho_csv=CAMINHO_DATASET):
    rotulos, dados = [], []
    with open(caminho_csv, "r", encoding="utf-8") as arq:
        leitor = csv.reader(arq)
        next(leitor, None)
        for linha in leitor:
            if len(linha) != 111:
                continue
            rotulos.append(linha[0])
            dados.append([float(v) for v in linha[1:]])
    return np.array(dados, dtype=np.float64), np.array(rotulos)


def carregar_dataset_videos(caminho_csv=CAMINHO_DATASET_VIDEOS):
    rotulos, origens, dados = [], [], []
    if not os.path.exists(caminho_csv):
        return np.empty((0, 110)), np.array([]), np.array([])
    with open(caminho_csv, "r", encoding="utf-8") as arq:
        leitor = csv.reader(arq)
        next(leitor, None)
        for linha in leitor:
            if len(linha) != 112:
                continue
            rotulos.append(linha[0])
            origens.append(linha[1])
            dados.append([float(v) for v in linha[2:]])
    return np.array(dados, dtype=np.float64), np.array(rotulos), np.array(origens)


def montar_pesos_balanceados(rotulos_sint, rotulos_reais, peso_real_por_classe=120.0):
    pesos_sint = np.ones(len(rotulos_sint), dtype=np.float64)
    if len(rotulos_reais) == 0:
        return pesos_sint
    contagens = Counter(rotulos_reais.tolist())
    pesos_reais = []
    for r in rotulos_reais:
        alvo = 70.0 if r in ("DIA_ROSTO", "DIA_SOL", "NEUTRO") else peso_real_por_classe
        pesos_reais.append(min(3.0, alvo / max(contagens[r], 1)))
    return np.concatenate([pesos_sint, np.array(pesos_reais, dtype=np.float64)])


def validar_lovo(sinais_filtro=None, recorte_celular=False):
    x_sint, y_sint = carregar_dataset_sintetico()
    x_real, y_real, origens_real = carregar_dataset_videos()
    if len(x_real) == 0:
        print("Execute primeiro --extrair-treino para gerar dataset_videos_reais.csv.")
        return

    videos_por_sinal = listar_videos_locais(sinais_filtro)
    extrator = ExtratorPontos()

    print("\nExtraindo sequencias reais dos videos para validacao Leave-One-Video-Out (sem vazamento)...")
    sequencias = []
    for sinal, lista in videos_por_sinal.items():
        for caminho in lista:
            origem = f"{sinal}/{os.path.basename(caminho)}"
            seq = extrair_sequencia_video(extrator, caminho, recorte_celular=recorte_celular, espelhar=False)
            if seq is not None:
                sequencias.append((sinal, origem, seq))

    total = len(sequencias)
    ok_gesto = 0
    ok_maj = 0
    ok_alguma = 0
    por_sinal = {}

    for sinal, origem, seq in sequencias:
        esperado = MAPA_SUBCLASSES.get(sinal, sinal)
        mascara = origens_real != origem
        x_r = x_real[mascara]
        y_r = y_real[mascara]

        x_tr = np.vstack([x_sint, x_r]) if len(x_r) > 0 else x_sint
        y_tr = np.concatenate([y_sint, y_r]) if len(y_r) > 0 else y_sint
        pesos = montar_pesos_balanceados(y_sint, y_r)

        rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=17,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(x_tr, y_tr, sample_weight=pesos)
        rf.n_jobs = 1

        r = classificar_sequencia(seq, lambda v, m=rf: prever_sklearn(m, v))
        acertou_g = r["pred_gesto_25s"] == esperado
        acertou_m = r["pred_majoritaria"] == esperado
        acertou_a = esperado in r["votos_janelas"]
        ok_gesto += int(acertou_g)
        ok_maj += int(acertou_m)
        ok_alguma += int(acertou_a)

        est = por_sinal.setdefault(sinal, [0, 0, 0])
        est[0] += int(acertou_g)
        est[1] += int(acertou_m)
        est[2] += 1

        marca = "OK" if (acertou_g or acertou_m) else "ERRO"
        print(
            f"  [{marca:4s}] {origem:36s} | gesto={r['pred_gesto_25s'] or '-':10s} ({r['conf_gesto_25s']:.2f}) "
            f"| maj={r['pred_majoritaria'] or '-':10s} | top={r['top_probs'][:3]}"
        )

    modo = "240x320 (celular)" if recorte_celular else "quadro padrao (max 480px)"
    print(f"\n================ RESUMO LOVO (SEM VAZAMENTO) - {modo} ================")
    print(f"Videos avaliados (100% fora do treino): {total}")
    print(f"Acerto modo Gravar Gesto (2.5s): {ok_gesto}/{total} ({100.0 * ok_gesto / max(total, 1):.1f}%)")
    print(f"Acerto maioria das janelas     : {ok_maj}/{total} ({100.0 * ok_maj / max(total, 1):.1f}%)")
    print(f"Sinal apareceu em >=1 janela   : {ok_alguma}/{total} ({100.0 * ok_alguma / max(total, 1):.1f}%)")
    print("\nPor sinal (gesto / maioria / total):")
    for sinal in sorted(por_sinal.keys()):
        g, m, tot = por_sinal[sinal]
        print(f"  {sinal:12s}: gesto={g}/{tot} | maioria={m}/{tot}")


def listar_videos_locais(sinais_filtro=None):
    encontrados = {}
    if not os.path.isdir(PASTA_VIDEOS):
        return encontrados
    for nome in sorted(os.listdir(PASTA_VIDEOS)):
        caminho_sinal = os.path.join(PASTA_VIDEOS, nome)
        if not os.path.isdir(caminho_sinal):
            continue
        sinal = nome.upper()
        if sinais_filtro and sinal not in sinais_filtro:
            continue
        arqs = [
            os.path.join(caminho_sinal, f)
            for f in sorted(os.listdir(caminho_sinal))
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
        ]
        if arqs:
            encontrados[sinal] = arqs
    return encontrados


def main():
    parser = argparse.ArgumentParser(description="Validacao do modelo mobile usando videos reais no MediaPipe.")
    parser.add_argument("--baixar", action="store_true", help="Baixa os videos reais oficiais.")
    parser.add_argument("--extrair-treino", action="store_true", help="Extrai janelas reais dos videos para dataset_videos_reais.csv.")
    parser.add_argument("--validar-lovo", action="store_true", help="Executa validacao Leave-One-Video-Out sem vazamento.")
    parser.add_argument("--sinais", type=str, default="", help="Lista de sinais separados por virgula.")
    parser.add_argument("--recorte-celular", action="store_true", help="Simula camera frontal vertical 240x320.")
    parser.add_argument("--espelhar", action="store_true", help="Avalia com imagem espelhada horizontalmente.")
    args = parser.parse_args()

    sinais_filtro = None
    if args.sinais.strip():
        sinais_filtro = {s.strip().upper() for s in args.sinais.split(",") if s.strip()}

    if args.baixar or not os.path.isdir(PASTA_VIDEOS):
        baixar_videos_oficiais(sinais_filtro)

    if args.extrair_treino:
        extrair_dataset_videos(sinais_filtro)
        return

    if args.validar_lovo:
        validar_lovo(sinais_filtro=sinais_filtro, recorte_celular=args.recorte_celular)
        return

    videos_por_sinal = listar_videos_locais(sinais_filtro)
    if not videos_por_sinal:
        print(f"Nenhum video encontrado em {PASTA_VIDEOS}. Use --baixar ou coloque arquivos .mp4 em subpastas por sinal.")
        return

    extrator = ExtratorPontos()
    modelo = carregar_modelo_mobile()

    total_videos = 0
    acertos_gesto = 0
    acertos_maj = 0
    acertos_alguma_janela = 0
    resumo_sinais = {}

    for sinal, lista_videos in videos_por_sinal.items():
        esperado = MAPA_SUBCLASSES.get(sinal, sinal)
        print(f"\n================ {sinal} ({len(lista_videos)} videos) ================")
        ok_gesto_sinal = 0
        ok_maj_sinal = 0
        for caminho in lista_videos:
            r = avaliar_video(
                extrator,
                modelo,
                caminho,
                recorte_celular=args.recorte_celular,
                espelhar=args.espelhar,
            )
            if r is None:
                print(f"  [erro leitura] {os.path.basename(caminho)}")
                continue
            total_videos += 1
            acertou_g = r["pred_gesto_25s"] == esperado
            acertou_m = r["pred_majoritaria"] == esperado
            acertou_alguma = esperado in r["votos_janelas"]
            if acertou_g:
                acertos_gesto += 1
                ok_gesto_sinal += 1
            if acertou_m:
                acertos_maj += 1
                ok_maj_sinal += 1
            if acertou_alguma:
                acertos_alguma_janela += 1

            marca = "OK" if (acertou_g or acertou_m) else "ERRO"
            print(
                f"  [{marca:4s}] {r['video']:28s} | q={r['quadros']:2d} (0m:{r['q0']} 1m:{r['q1']} 2m:{r['q2']} pose:{r['q_pose']}) "
                f"| gesto={r['pred_gesto_25s'] or '-':10s} ({r['conf_gesto_25s']:.2f}) "
                f"| maj={r['pred_majoritaria'] or '-':10s} | top={r['top_probs'][:3]}"
            )
        resumo_sinais[sinal] = (ok_gesto_sinal, ok_maj_sinal, len(lista_videos))

    print("\n================ RESUMO GERAL ================")
    modo = "240x320 (celular)" if args.recorte_celular else "quadro padrao (max 480px)"
    print(f"Modo: {modo} | Espelhado: {args.espelhar}")
    print(f"Total de videos reais avaliados: {total_videos}")
    if total_videos > 0:
        print(f"Acerto modo Gravar Gesto (2.5s): {acertos_gesto}/{total_videos} ({100.0 * acertos_gesto / total_videos:.1f}%)")
        print(f"Acerto maioria das janelas     : {acertos_maj}/{total_videos} ({100.0 * acertos_maj / total_videos:.1f}%)")
        print(f"Sinal apareceu em >=1 janela   : {acertos_alguma_janela}/{total_videos} ({100.0 * acertos_alguma_janela / total_videos:.1f}%)")
        print("\nPor sinal (gesto / maioria / total):")
        for sinal, (g, m, tot) in resumo_sinais.items():
            print(f"  {sinal:12s}: gesto={g}/{tot} | maioria={m}/{tot}")


if __name__ == "__main__":
    main()
