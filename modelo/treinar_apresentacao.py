import argparse
from collections import Counter, deque
import csv
import hashlib
import io
import json
import math
import os
import pickletools
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DIRETORIO_MODELO = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_MODELO)
PASTA_VIDEOS_REAIS = os.path.join(DIRETORIO_MODELO, "videos_reais")
PASTA_VIDEOS_APRES = os.path.join(DIRETORIO_MODELO, "videos_apresentacao")
CAMINHO_DATASET_SINT = os.path.join(DIRETORIO_MODELO, "dataset_libras.csv")
CAMINHO_DATASET_APRES = os.path.join(DIRETORIO_MODELO, "dataset_apresentacao_reais.csv")
CAMINHO_MODELO_JOBLIB = os.path.join(RAIZ_PROJETO, "servidor", "modelos_ia", "modelo_apresentacao.joblib")
CAMINHO_ROTULOS_JOBLIB = os.path.join(RAIZ_PROJETO, "servidor", "modelos_ia", "rotulos_apresentacao.joblib")
CAMINHO_MODELO_MOBILE = os.path.join(RAIZ_PROJETO, "aplicativo", "assets", "modelo_apresentacao_mobile.json")

sys.path.append(os.path.join(RAIZ_PROJETO, "servidor"))
from extrator_pontos import (
    ExtratorPontos,
    calcular_vetor_quadro,
    filtrar_maos_ativas,
    montar_vetor_temporal_expandido,
)

TOTAL_CARACTERISTICAS = 132

JANELA_TEMPORAL_SEG = 1.1
INTERVALO_AMOSTRA_SEG = 0.11

INES_BASE = "https://www.ines.gov.br/dicionario-de-libras/public/media/palavras/videos/"
VLIB_BASE = "https://huggingface.co/datasets/SchberGg/v-librasil-raw/resolve/main/videos/"
UFSC_GLOSS = "https://glossario.libras.ufsc.br/media/sinal_videos/"
SIGNBANK = "https://videos.nals.cce.ufsc.br/SignBank/V%C3%ADdeos/"
MALTA = "https://huggingface.co/datasets/MALTA-Lab/MALTA_LIBRAS/resolve/main/malta_libras_complete/"

MAPA_SUBCLASSES = {
    "DIA_ROSTO": "DIA",
    "DIA_SOL": "DIA",
    "2_HORIZ": "2",
    "2_VERT": "2",
}

SINAIS_REUTILIZADOS = ("BOM", "DIA_ROSTO", "DIA_SOL")
REUTILIZAR_ARQUIVOS = {
    "2_VERT": [
        os.path.join(PASTA_VIDEOS_REAIS, "V", "vlibrasil_v_2.mp4"),
    ],
}
SINAIS_COMPOSTOS_TESTE = set()

# Catalogo oficial curado visualmente (zero duplicatas e zero variantes incompativeis)
CATALOGO_APRESENTACAO = {
    "LIBRAS": [
        ("sb_libras.mp4", SIGNBANK + "LIBRAS.mp4"),
        ("ufsc_libras_104.mp4", UFSC_GLOSS + "104-sinal-2019-09-14-05-38-05.mp4"),
        ("ines_lingua_sinais.mp4", INES_BASE + "lingua-de-sinaisSm_Prog001.mp4"),
        ("malta_usp_ls.pt", MALTA + "usp/lingua_de_sinais.pt"),
        ("malta_sts_ls.pt", MALTA + "spread_the_sign/" + urllib.parse.quote("língua de sinais.pt")),
        ("malta_acess2_ls.pt", MALTA + "acessibilidade_2/lingua-de-sinais.pt"),
    ],
    "2_HORIZ": [
        ("ines_dois1.mp4", INES_BASE + "dois1Sm_Prog001.mp4"),
        ("sts_dois.mp4", "https://media.spreadthesign.com/video/mp4/14/554120.mp4"),
        ("malta_ufv_dois.pt", MALTA + "ufv/1343-dois.pt"),
        ("malta_acess2_dois1.pt", MALTA + "acessibilidade_2/dois1.pt"),
    ],
    "2_VERT": [
        ("ines_dois2.mp4", INES_BASE + "dois2Sm_Prog001.mp4"),
        ("malta_acess2_dois2.pt", MALTA + "acessibilidade_2/dois2.pt"),
    ],
    "TEXTO": [
        ("sb_texto.mp4", SIGNBANK + "TEXTO.mp4"),
        ("sts_texto.mp4", "https://media.spreadthesign.com/video/mp4/14/552639.mp4"),
        ("ines_anotar1.mp4", INES_BASE + "anotar1Sm_Prog001.mp4"),
    ],
    "APLICATIVO": [
        ("ufsc_aplicativo.mp4", UFSC_GLOSS + "487-sinal-2019-09-16-21-03-54.mp4"),
        ("malta_ufv_aplicativo.pt", MALTA + "ufv/aplicativo.pt"),
    ],
    "TRADUZIR": [
        ("vlibrasil_traduzir_1.mp4", VLIB_BASE + urllib.parse.quote("Traduzir_Articulador1.mp4")),
        ("vlibrasil_traduzir_2.mp4", VLIB_BASE + urllib.parse.quote("Traduzir_Articulador2.mp4")),
        ("vlibrasil_traduzir_3.mp4", VLIB_BASE + urllib.parse.quote("Traduzir_Articulador3.mp4")),
        ("sts_tradutor.mp4", "https://media.spreadthesign.com/video/mp4/14/660460.mp4"),
        ("sb_traducao.mp4", SIGNBANK + urllib.parse.quote("TRADUÇÃO.mp4")),
        ("malta_usp_traducao.pt", MALTA + "usp/traducao.pt"),
        ("malta_ufsc_traduzir.pt", MALTA + "ufsc/TRADUZIR.pt"),
    ],
    "TEMPO": [
        ("vlibrasil_tempo_1.mp4", VLIB_BASE + urllib.parse.quote("Tempo_Articulador1.mp4")),
        ("vlibrasil_tempo_3.mp4", VLIB_BASE + urllib.parse.quote("Tempo_Articulador3.mp4")),
        ("ines_tempo1.mp4", INES_BASE + "tempoSm_Prog001.mp4"),
        ("ines_hora1.mp4", INES_BASE + "hora1Sm_Prog001.mp4"),
        ("sts_hora.mp4", "https://media.spreadthesign.com/video/mp4/14/551946.mp4"),
        ("vlibrasil_hora_1.mp4", VLIB_BASE + urllib.parse.quote("Hora_Articulador1.mp4")),
        ("vlibrasil_hora_2.mp4", VLIB_BASE + urllib.parse.quote("Hora_Articulador2.mp4")),
        ("vlibrasil_hora_3.mp4", VLIB_BASE + urllib.parse.quote("Hora_Articulador3.mp4")),
    ],
    "REAL": [
        ("vlibrasil_real_1.mp4", VLIB_BASE + urllib.parse.quote("Real_Articulador1.mp4")),
        ("vlibrasil_real_2.mp4", VLIB_BASE + urllib.parse.quote("Real_Articulador2.mp4")),
        ("vlibrasil_real_3.mp4", VLIB_BASE + urllib.parse.quote("Real_Articulador3.mp4")),
        ("ines_real2.mp4", INES_BASE + "real2Sm_Prog001.mp4"),
        ("ines_verdade.mp4", INES_BASE + "verdadeSm_Prog001.mp4"),
        ("sb_verdade.mp4", SIGNBANK + "VERDADE.mp4"),
        ("sts_verdade1.mp4", "https://media.spreadthesign.com/video/mp4/14/552936.mp4"),
        ("malta_acess2_real2.pt", MALTA + "acessibilidade_2/real2.pt"),
    ],
    "TRANSFORMAR": [
        ("ines_transformar.mp4", INES_BASE + "transformarSm_Prog001.mp4"),
        ("sts_transformar.mp4", "https://media.spreadthesign.com/video/mp4/14/481128.mp4"),
        ("malta_acess2_transformar.pt", MALTA + "acessibilidade_2/transformar.pt"),
    ],
    "INOVACAO": [
        ("sb_novidade.mp4", SIGNBANK + "NOVIDADE.mp4"),
        ("ines_novidade.mp4", INES_BASE + "novidadeSm_Prog001.mp4"),
        ("malta_acess2_novidade.pt", MALTA + "acessibilidade_2/novidade.pt"),
    ],
    "ACESSIBILIDADE": [
        ("ufsc_acessibilidade.mp4", UFSC_GLOSS + "3-sinal-2019-09-14-02-46-17.mp4"),
        ("sb_acessibilidade.mp4", SIGNBANK + "ACESSIBILIDADE.mp4"),
    ],
}


def sincronizar_pasta_curada():
    if not os.path.isdir(PASTA_VIDEOS_APRES):
        return
    validos_por_sinal = {s: {n for n, _ in lst} for s, lst in CATALOGO_APRESENTACAO.items()}
    for nome in list(os.listdir(PASTA_VIDEOS_APRES)):
        caminho_sub = os.path.join(PASTA_VIDEOS_APRES, nome)
        if not os.path.isdir(caminho_sub):
            continue
        if nome not in validos_por_sinal:
            for arq in os.listdir(caminho_sub):
                os.remove(os.path.join(caminho_sub, arq))
            os.rmdir(caminho_sub)
            continue
        permitidos = validos_por_sinal[nome]
        for arq in os.listdir(caminho_sub):
            if arq.startswith("yt_"):
                continue
            if arq not in permitidos:
                os.remove(os.path.join(caminho_sub, arq))


def baixar_videos_apresentacao(sinais_filtro=None):
    os.makedirs(PASTA_VIDEOS_APRES, exist_ok=True)
    sincronizar_pasta_curada()
    ctx_livre = ssl._create_unverified_context()
    for sinal, lista in CATALOGO_APRESENTACAO.items():
        if sinais_filtro and sinal not in sinais_filtro:
            continue
        pasta = os.path.join(PASTA_VIDEOS_APRES, sinal)
        os.makedirs(pasta, exist_ok=True)
        for nome_arq, url in lista:
            destino = os.path.join(pasta, nome_arq)
            if os.path.exists(destino) and os.path.getsize(destino) > 1000:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                ctx = ctx_livre if "videos.nals.cce.ufsc.br" in url else None
                with urllib.request.urlopen(req, timeout=25, context=ctx) as resp, open(destino, "wb") as out:
                    out.write(resp.read())
                print(f"[baixado] {sinal}/{nome_arq}")
            except Exception as erro:
                print(f"[falha]   {sinal}/{nome_arq}: {erro}")


def _ler_quadros_pt(caminho_pt):
    with zipfile.ZipFile(caminho_pt, "r") as z:
        membro_pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        membro_data = next(n for n in z.namelist() if n.endswith("/data/0"))
        pkl = z.read(membro_pkl)
        raw = z.read(membro_data)

    shape = None
    capturando = False
    pilha = []
    for op, arg, _ in pickletools.genops(pkl):
        if op.name == "BINPERSID":
            capturando = True
        elif capturando and op.name == "MARK":
            pilha = []
        elif capturando and op.name in ("BININT", "BININT1", "BININT2"):
            pilha.append(int(arg))
        elif capturando and op.name == "TUPLE" and 3 <= len(pilha) <= 4:
            shape = tuple(pilha)
            break

    if shape is None or len(shape) != 4:
        t_infer = len(raw) // (3 * 224 * 224)
        shape = (t_infer, 3, 224, 224)

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    frames_rgb = np.transpose(arr, (0, 2, 3, 1))
    return [np.ascontiguousarray(frames_rgb[i, :, :, ::-1]) for i in range(frames_rgb.shape[0])]


def carregar_quadros_video(caminho):
    if caminho.lower().endswith(".pt"):
        frames = _ler_quadros_pt(caminho)
        return frames, 13.0

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 120.0:
        fps = 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def listar_videos_apresentacao(sinais_filtro=None, incluir_compostos_teste=True):
    encontrados = {}
    extensoes = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".pt")

    for sinal in SINAIS_REUTILIZADOS:
        if sinais_filtro and sinal not in sinais_filtro:
            continue
        pasta = os.path.join(PASTA_VIDEOS_REAIS, sinal)
        if os.path.isdir(pasta):
            arqs = [
                os.path.join(pasta, f)
                for f in sorted(os.listdir(pasta))
                if f.lower().endswith(extensoes)
            ]
            if arqs:
                encontrados[sinal] = arqs

    for sinal, lista_extra in REUTILIZAR_ARQUIVOS.items():
        if sinais_filtro and sinal not in sinais_filtro:
            continue
        existentes = [cam for cam in lista_extra if os.path.exists(cam)]
        if existentes:
            encontrados.setdefault(sinal, []).extend(existentes)

    if incluir_compostos_teste:
        for sinal in sorted(SINAIS_COMPOSTOS_TESTE):
            if sinais_filtro and sinal not in sinais_filtro:
                continue
            pasta = os.path.join(PASTA_VIDEOS_REAIS, sinal)
            if os.path.isdir(pasta):
                arqs = [
                    os.path.join(pasta, f)
                    for f in sorted(os.listdir(pasta))
                    if f.lower().endswith(extensoes)
                ]
                if arqs:
                    encontrados[sinal] = arqs

    if os.path.isdir(PASTA_VIDEOS_APRES):
        for nome in sorted(os.listdir(PASTA_VIDEOS_APRES)):
            pasta = os.path.join(PASTA_VIDEOS_APRES, nome)
            if not os.path.isdir(pasta):
                continue
            sinal = nome.upper()
            if sinais_filtro and sinal not in sinais_filtro:
                continue
            arqs = [
                os.path.join(pasta, f)
                for f in sorted(os.listdir(pasta))
                if f.lower().endswith(extensoes)
            ]
            if arqs:
                encontrados.setdefault(sinal, []).extend(arqs)

    return encontrados


def verificar_duplicatas_md5():
    vistos = {}
    duplicatas = []
    for sinal, lista in listar_videos_apresentacao(incluir_compostos_teste=True).items():
        for cam in lista:
            h = hashlib.md5()
            with open(cam, "rb") as f:
                while True:
                    bloco = f.read(65536)
                    if not bloco:
                        break
                    h.update(bloco)
            digest = h.hexdigest()
            ref = f"{sinal}/{os.path.basename(cam)}"
            if digest in vistos:
                duplicatas.append((vistos[digest], ref, digest))
            else:
                vistos[digest] = ref
    if duplicatas:
        print("Duplicatas MD5 encontradas:")
        for a, b, d in duplicatas:
            print(f"  {a} == {b} ({d[:10]})")
    else:
        print("Nenhuma duplicata MD5 encontrada.")


def gerar_pranchas_inspecao(pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    for sinal, lista in listar_videos_apresentacao(incluir_compostos_teste=False).items():
        if sinal in SINAIS_REUTILIZADOS:
            continue
        linhas_img = []
        for cam in lista:
            frames, _ = carregar_quadros_video(cam)
            if not frames:
                continue
            idxs = np.linspace(0, len(frames) - 1, 5, dtype=int)
            celulas = []
            for idx in idxs:
                fr = cv2.resize(frames[int(idx)], (192, 144), interpolation=cv2.INTER_AREA)
                celulas.append(fr)
            tira = np.hstack(celulas)
            faixa = np.full((26, tira.shape[1], 3), 245, dtype=np.uint8)
            nome = os.path.basename(cam)
            cv2.putText(faixa, f"{sinal} / {nome} ({len(frames)}q)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
            linhas_img.append(np.vstack([faixa, tira]))
        if linhas_img:
            prancha = np.vstack(linhas_img)
            arq_out = os.path.join(pasta_saida, f"inspecao_{sinal}.jpg")
            cv2.imwrite(arq_out, prancha, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"[prancha] {arq_out} ({len(linhas_img)} videos)")


def transformar_quadro(quadro_bgr, recorte_celular=False, zoom=1.0):
    img = quadro_bgr
    if recorte_celular:
        alt, larg = img.shape[:2]
        larg_alvo = int(round(alt * 0.75))
        if larg > larg_alvo:
            ini_x = (larg - larg_alvo) // 2
            img = img[:, ini_x:ini_x + larg_alvo]
        elif alt > int(round(larg / 0.75)):
            alt_alvo = int(round(larg / 0.75))
            ini_y = max(0, (alt - alt_alvo) // 4)
            img = img[ini_y:ini_y + alt_alvo, :]

    if abs(zoom - 1.0) > 1e-3:
        h, w = img.shape[:2]
        if zoom < 1.0:
            nh, nw = max(4, int(round(h * zoom))), max(4, int(round(w * zoom)))
            y0 = (h - nh) // 2
            x0 = (w - nw) // 2
            img = img[y0:y0 + nh, x0:x0 + nw]
        else:
            pad_y = int(round(h * (zoom - 1.0) * 0.5))
            pad_x = int(round(w * (zoom - 1.0) * 0.5))
            img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

    if recorte_celular:
        return cv2.resize(img, (360, 480), interpolation=cv2.INTER_AREA)
    return img


def extrair_sequencia_video(extrator, caminho_video, recorte_celular=False, espelhar=False, zoom=1.0, coletar_neutro=False):
    frames_brutos, fps = carregar_quadros_video(caminho_video)
    if not frames_brutos:
        return None

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

    for idx_quadro, frame in enumerate(frames_brutos, start=1):
        if (idx_quadro - 1) % passo_quadro != 0:
            continue

        total_quadros += 1
        tempo_video = idx_quadro / fps

        if espelhar:
            frame = cv2.flip(frame, 1)
        frame = transformar_quadro(frame, recorte_celular=recorte_celular, zoom=zoom)

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

        if coletar_neutro and maos_brutas and not maos_ativas and len(vetores_neutros) < 6:
            outra_rep = maos_brutas[1] if len(maos_brutas) == 2 else None
            q_rep = calcular_vetor_quadro(maos_brutas[0], larg, alt, pontos_pose if pontos_pose else None, outra_rep)
            if q_rep is not None:
                vetores_neutros.append(montar_vetor_temporal_expandido([q_rep]))

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
            vetor = montar_vetor_temporal_expandido(list(trilha["quadros"]))
            opcoes_trilhas.append({
                "vetor": vetor,
                "centro_y": trilha["centro"][1],
                "quadros": len(trilha["quadros"]),
            })
        janelas.append({"vazio": False, "tempo": tempo_video, "trilhas": opcoes_trilhas})

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


def classificar_sequencia(seq, fn_prever):
    votacoes = deque(maxlen=3)
    estava_em_movimento = False
    predicoes_janela = []
    acumulador_gesto = {}
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
                if len(vetor) >= 132 and float(vetor[118]) > 0.25 and (maior_y - menor_y) > 0.10 and t["centro_y"] <= menor_y + 0.02:
                    prioridade += 0.08
                if em_mov:
                    prioridade += 0.14
            avaliacoes.append({
                "rotulo": rotulo,
                "conf": conf,
                "prioridade": prioridade,
                "em_mov": em_mov,
                "quadros": t["quadros"],
                "ranking": ranking,
            })

        if len(avaliacoes) == 2 and avaliacoes[0]["rotulo"] and avaliacoes[0]["rotulo"] == avaliacoes[1]["rotulo"]:
            escolhida = max(avaliacoes, key=lambda a: a["conf"])
            rot_bruto = escolhida["rotulo"]
            conf_bruta = min(1.0, escolhida["conf"] + 0.05)
            em_mov_atual = avaliacoes[0]["em_mov"] or avaliacoes[1]["em_mov"]
            buffer_atual = max(avaliacoes[0]["quadros"], avaliacoes[1]["quadros"])
        else:
            escolhida = max(avaliacoes, key=lambda a: a["prioridade"])
            rot_bruto = escolhida["rotulo"]
            conf_bruta = escolhida["conf"]
            em_mov_atual = escolhida["em_mov"]
            buffer_atual = escolhida["quadros"]

        for cls, p in escolhida["ranking"].items():
            if cls != "NEUTRO":
                soma_probs_classe[cls] += p
        total_avaliacoes += 1

        if em_mov_atual and not estava_em_movimento:
            votacoes = deque([v for v in votacoes if v[0] == rot_bruto], maxlen=3)
        estava_em_movimento = em_mov_atual

        votacoes.append((rot_bruto, conf_bruta))
        pred_base, media_conf = escolher_rotulo_votacao(votacoes)

        if pred_base:
            predicoes_janela.append((pred_base, media_conf))

        if pred_base and media_conf >= 0.36:
            reg = acumulador_gesto.setdefault(pred_base, [0.0, 0.0, 0.0, 0, tempo_atual, tempo_atual])
            reg[0] += media_conf * min(1.0, buffer_atual / 4.0)
            reg[1] += media_conf
            if media_conf > reg[2]:
                reg[2] = media_conf
            reg[3] += 1
            reg[5] = tempo_atual

    melhor_rotulo = ""
    melhor_conf = 0.0
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
    }


def extrair_dataset_apresentacao(sinais_filtro=None):
    videos_por_sinal = listar_videos_apresentacao(sinais_filtro, incluir_compostos_teste=False)
    if not videos_por_sinal:
        print("Nenhum video encontrado para extracao.")
        return

    extrator = ExtratorPontos()
    cabecalho = ["rotulo", "origem_video"] + [f"coord_{i}" for i in range(TOTAL_CARACTERISTICAS)]
    linhas = [cabecalho]
    neutros_total = 0

    visoes_base = (
        (False, False, 1.0),
        (False, True, 1.0),
        (True, False, 1.0),
        (True, True, 1.0),
        (True, False, 1.18),
    )
    visoes_reforco = visoes_base + (
        (True, True, 1.18),
        (True, False, 0.90),
    )
    sinais_2m = {"ACESSIBILIDADE", "APLICATIVO", "LIBRAS", "TEXTO", "TRADUZIR", "TEMPO", "REAL", "INOVACAO"}
    sinais_reforco = {"APLICATIVO", "INOVACAO", "ACESSIBILIDADE", "TEXTO", "TRANSFORMAR"}

    for sinal, lista in videos_por_sinal.items():
        total_sinal = 0
        visoes = visoes_reforco if sinal in sinais_reforco else visoes_base
        for caminho in lista:
            origem = f"{sinal}/{os.path.basename(caminho)}"
            for recorte, espelhar, zoom in visoes:
                seq = extrair_sequencia_video(
                    extrator,
                    caminho,
                    recorte_celular=recorte,
                    espelhar=espelhar,
                    zoom=zoom,
                    coletar_neutro=(not espelhar and not recorte and abs(zoom - 1.0) < 1e-3),
                )
                if seq is None:
                    continue

                if seq["vetores_neutros"] and neutros_total < 220:
                    for vn in seq["vetores_neutros"][:3]:
                        linhas.append(["NEUTRO", origem] + [round(float(x), 6) for x in vn])
                        neutros_total += 1

                janelas_validas = [j for j in seq["janelas"] if not j.get("vazio") and j["trilhas"]]
                if not janelas_validas:
                    continue

                janelas_1m_inov = []
                if sinal == "INOVACAO":
                    janelas_1m_inov = [
                        j for j in janelas_validas
                        if all(t["vetor"][92] < 0.5 for t in j["trilhas"])
                        and any(
                            t["vetor"][74] > 0.7 and t["vetor"][75] > 0.7 and t["vetor"][76] < -0.2 and t["vetor"][77] < -0.2
                            for t in j["trilhas"]
                        )
                    ][:4]

                if sinal in sinais_2m:
                    j2m = [
                        j for j in janelas_validas
                        if any(t["vetor"][92] > 0.5 and t["vetor"][88] <= 1.38 for t in j["trilhas"])
                    ]
                    if len(j2m) >= 3:
                        janelas_validas = j2m

                n_val = len(janelas_validas)
                ini = int(n_val * 0.14) if n_val >= 8 else 0
                fim = int(n_val * 0.88) if n_val >= 8 else n_val
                selecionadas = janelas_validas[ini:max(fim, ini + 1)]

                if len(selecionadas) > 14:
                    indices = np.linspace(0, len(selecionadas) - 1, 14, dtype=int)
                    selecionadas = [selecionadas[i] for i in indices]

                for j in selecionadas:
                    for t in j["trilhas"]:
                        if t["quadros"] < 2 and n_val >= 4:
                            continue
                        if sinal in sinais_2m and t["vetor"][92] < 0.5 and any(tt["vetor"][92] > 0.5 for tt in j["trilhas"]):
                            continue
                        linhas.append([sinal, origem] + [round(float(x), 6) for x in t["vetor"]])
                        total_sinal += 1

                for j in janelas_1m_inov:
                    for t in j["trilhas"]:
                        if t["quadros"] < 2:
                            continue
                        linhas.append([sinal, origem] + [round(float(x), 6) for x in t["vetor"]])
                        total_sinal += 1
        print(f"[extraido] {sinal:14s}: {total_sinal} janelas reais ({len(lista)} videos)")

    print(f"[extraido] {'NEUTRO':14s}: {neutros_total} janelas reais de repouso")
    with open(CAMINHO_DATASET_APRES, "w", newline="", encoding="utf-8") as arq:
        csv.writer(arq).writerows(linhas)
    print(f"Dataset da apresentacao salvo em {CAMINHO_DATASET_APRES} ({len(linhas) - 1} amostras).")


def _expandir_sintetico_110_para_132(v):
    ang_sim = math.atan2(v[9 * 3 + 1], abs(v[9 * 3 + 0]))
    abs_det = min(1.5, abs(v[5 * 3 + 0] * v[17 * 3 + 1] - v[5 * 3 + 1] * v[17 * 3 + 0]))
    pol_base = min(2.5, math.hypot(v[4 * 3 + 0] - v[5 * 3 + 0], v[4 * 3 + 1] - v[5 * 3 + 1]))
    if v[92] > 0.5:
        biman_extras = [
            min(4.0, v[94]),
            min(4.0, v[93]),
            min(4.0, v[94]),
            min(4.0, max(0.2, v[93] * 0.7)),
            min(4.0, abs(v[96])),
            0.0,
            v[73],
            v[74],
            v[75],
            v[76],
            v[77],
        ]
    else:
        biman_extras = [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    temp_extras = [
        0.05,
        0.0,
        min(4.0, abs(v[105]) * 1.5),
        min(3.0, abs(v[105])),
        0.0,
        0.0,
        0.0,
        min(4.0, v[94]) if v[92] > 0.5 else 4.0,
    ]
    return v + [ang_sim, abs_det, pol_base] + biman_extras + temp_extras


def carregar_sintetico_filtrado(classes_permitidas):
    rotulos, dados = [], []
    if not os.path.exists(CAMINHO_DATASET_SINT):
        return np.empty((0, TOTAL_CARACTERISTICAS)), np.array([])
    with open(CAMINHO_DATASET_SINT, "r", encoding="utf-8") as arq:
        leitor = csv.reader(arq)
        next(leitor, None)
        for linha in leitor:
            if len(linha) != 111:
                continue
            r = linha[0]
            if r not in classes_permitidas:
                continue
            rotulos.append(r)
            dados.append(_expandir_sintetico_110_para_132([float(v) for v in linha[1:]]))
    return np.array(dados, dtype=np.float64), np.array(rotulos)


def carregar_dataset_videos(caminho_csv=CAMINHO_DATASET_APRES):
    rotulos, origens, dados = [], [], []
    if not os.path.exists(caminho_csv):
        return np.empty((0, TOTAL_CARACTERISTICAS)), np.array([]), np.array([])
    with open(caminho_csv, "r", encoding="utf-8") as arq:
        leitor = csv.reader(arq)
        next(leitor, None)
        for linha in leitor:
            if len(linha) != TOTAL_CARACTERISTICAS + 2:
                continue
            rotulos.append(linha[0])
            origens.append(linha[1])
            dados.append([float(v) for v in linha[2:]])
    return np.array(dados, dtype=np.float64), np.array(rotulos), np.array(origens)


def montar_pesos(rotulos_sint, rotulos_reais):
    subclasses = {"DIA_ROSTO", "DIA_SOL", "NEUTRO", "2_HORIZ", "2_VERT"}
    cont_sint = Counter(rotulos_sint.tolist()) if len(rotulos_sint) > 0 else Counter()
    cont_reais = Counter(rotulos_reais.tolist()) if len(rotulos_reais) > 0 else Counter()

    pesos_sint = []
    for r in rotulos_sint:
        alvo_s = 25.0 if (r in subclasses or r == "BOM") else 35.0
        pesos_sint.append(min(1.0, alvo_s / max(cont_sint[r], 1)))

    pesos_reais = []
    for r in rotulos_reais:
        if r == "DIA_SOL":
            alvo = 100.0
        elif r in subclasses:
            alvo = 85.0
        elif r == "BOM":
            alvo = 115.0
        else:
            alvo = 150.0
        if r in cont_sint:
            alvo -= 25.0 if (r in subclasses or r == "BOM") else 35.0
        pesos_reais.append(min(5.0, max(0.08, alvo / max(cont_reais[r], 1))))

    return np.array(pesos_sint + pesos_reais, dtype=np.float64)


def validar_lovo(sinais_filtro=None, recorte_celular=True):
    x_real, y_real, origens_real = carregar_dataset_videos()
    if len(x_real) == 0:
        print("Execute primeiro --extrair-treino.")
        return

    x_sint, y_sint = carregar_sintetico_filtrado({"DIA_SOL", "NEUTRO"})
    videos_por_sinal = listar_videos_apresentacao(sinais_filtro, incluir_compostos_teste=True)
    extrator = ExtratorPontos()

    print("\nExtraindo sequencias reais para LOVO sem vazamento (360x480)...")
    sequencias = []
    for sinal, lista in videos_por_sinal.items():
        for cam in lista:
            origem = f"{sinal}/{os.path.basename(cam)}"
            seq = extrair_sequencia_video(extrator, cam, recorte_celular=recorte_celular, espelhar=False)
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

        x_tr = np.vstack([x_sint, x_r]) if len(x_sint) > 0 else x_r
        y_tr = np.concatenate([y_sint, y_r]) if len(y_sint) > 0 else y_r
        pesos = montar_pesos(y_sint, y_r)

        rf = RandomForestClassifier(
            n_estimators=100,
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
            f"  [{marca:4s}] {origem:40s} | q={r['quadros']:2d} (0m:{r['q0']} 1m:{r['q1']} 2m:{r['q2']}) "
            f"| gesto={r['pred_gesto_25s'] or '-':14s} ({r['conf_gesto_25s']:.2f}) "
            f"| maj={r['pred_majoritaria'] or '-':14s} | top={r['top_probs'][:3]}"
        )

    modo = "360x480 (celular)" if recorte_celular else "quadro padrao"
    print(f"\n================ RESUMO LOVO APRESENTACAO - {modo} ================")
    print(f"Videos avaliados (100% fora do treino): {total}")
    print(f"Acerto modo Gravar Gesto (2.5s): {ok_gesto}/{total} ({100.0 * ok_gesto / max(total, 1):.1f}%)")
    print(f"Acerto maioria das janelas     : {ok_maj}/{total} ({100.0 * ok_maj / max(total, 1):.1f}%)")
    print(f"Sinal apareceu em >=1 janela   : {ok_alguma}/{total} ({100.0 * ok_alguma / max(total, 1):.1f}%)")
    print("\nPor sinal (gesto / maioria / total):")
    for sinal in sorted(por_sinal.keys()):
        g, m, tot = por_sinal[sinal]
        print(f"  {sinal:16s}: gesto={g}/{tot} | maioria={m}/{tot}")


def treinar_e_exportar():
    x_real, y_real, _ = carregar_dataset_videos()
    if len(x_real) == 0:
        print("Execute primeiro --extrair-treino.")
        return

    x_sint, y_sint = carregar_sintetico_filtrado({"DIA_SOL", "NEUTRO"})
    x = np.vstack([x_sint, x_real]) if len(x_sint) > 0 else x_real
    y = np.concatenate([y_sint, y_real]) if len(y_sint) > 0 else y_real
    pesos = montar_pesos(y_sint, y_real)

    x_tr, x_te, y_tr, y_te, p_tr, _ = train_test_split(x, y, pesos, test_size=0.2, random_state=42, stratify=y)
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=17,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(x_tr, y_tr, sample_weight=p_tr)
    acuracia = accuracy_score(y_te, modelo.predict(x_te))
    print(f"Acuracia holdout interno (apresentacao): {acuracia * 100:.2f}%")

    modelo.fit(x, y, sample_weight=pesos)
    modelo.n_jobs = 1
    rotulos_unicos = sorted(list(set(y)))

    os.makedirs(os.path.dirname(CAMINHO_MODELO_JOBLIB), exist_ok=True)
    joblib.dump(modelo, CAMINHO_MODELO_JOBLIB)
    joblib.dump(rotulos_unicos, CAMINHO_ROTULOS_JOBLIB)

    arvores = []
    for estimador in modelo.estimators_:
        arvore = estimador.tree_
        valores = arvore.value[:, 0, :]
        totais = valores.sum(axis=1, keepdims=True)
        probabilidades = np.divide(valores, totais, out=np.zeros_like(valores), where=totais != 0)
        probs_arred = np.round(probabilidades, 4).tolist()
        eh_folha = (arvore.children_left == -1).tolist()
        probs_otim = [probs_arred[idx] if folha else [] for idx, folha in enumerate(eh_folha)]
        arvores.append({
            "esquerda": arvore.children_left.tolist(),
            "direita": arvore.children_right.tolist(),
            "caracteristica": arvore.feature.tolist(),
            "limite": np.round(arvore.threshold, 6).tolist(),
            "probabilidades": probs_otim,
        })

    artefato = {
        "versao": 2,
        "fonte_dados": "apresentacao_videos_reais",
        "quantidade_caracteristicas": int(modelo.n_features_in_),
        "rotulos": [str(r) for r in modelo.classes_],
        "arvores": arvores,
    }
    with open(CAMINHO_MODELO_MOBILE, "w", encoding="utf-8") as arq:
        json.dump(artefato, arq, separators=(",", ":"), ensure_ascii=False)

    tam_mb = os.path.getsize(CAMINHO_MODELO_MOBILE) / (1024 * 1024)
    print(f"Modelo apresentacao salvo: {CAMINHO_MODELO_MOBILE} ({tam_mb:.2f} MB | classes={len(rotulos_unicos)})")


def main():
    parser = argparse.ArgumentParser(description="Curadoria, validacao real e treino do modo Apresentacao.")
    parser.add_argument("--baixar", action="store_true", help="Baixa os videos candidatos oficiais.")
    parser.add_argument("--inspecionar", type=str, default="", help="Gera pranchas de contato no diretorio informado.")
    parser.add_argument("--extrair-treino", action="store_true", help="Extrai janelas reais dos videos curados.")
    parser.add_argument("--validar-lovo", action="store_true", help="Roda validacao Leave-One-Video-Out sem vazamento.")
    parser.add_argument("--treinar", action="store_true", help="Treina e exporta modelo_apresentacao_mobile.json.")
    parser.add_argument("--recorte-celular", action="store_true", help="Simula recorte vertical 360x480.")
    parser.add_argument("--sinais", type=str, default="", help="Lista de sinais separados por virgula.")
    args = parser.parse_args()

    sinais_filtro = None
    if args.sinais.strip():
        sinais_filtro = {s.strip().upper() for s in args.sinais.split(",") if s.strip()}

    if args.baixar:
        baixar_videos_apresentacao(sinais_filtro)
        verificar_duplicatas_md5()

    if args.inspecionar.strip():
        gerar_pranchas_inspecao(args.inspecionar.strip())

    if args.extrair_treino:
        extrair_dataset_apresentacao(sinais_filtro)

    if args.validar_lovo:
        validar_lovo(sinais_filtro=sinais_filtro, recorte_celular=args.recorte_celular)

    if args.treinar:
        treinar_e_exportar()


if __name__ == "__main__":
    main()
