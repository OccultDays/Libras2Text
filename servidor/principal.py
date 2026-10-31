import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import sys
import time
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

diretorio_servidor = Path(__file__).resolve().parent
if str(diretorio_servidor) not in sys.path:
    sys.path.append(str(diretorio_servidor))

from classificador_libras import ClassificadorLibras
from extrator_pontos import ExtratorPontos, montar_vetor_temporal
from gerenciador_conexoes import GerenciadorConexoes
from processador_imagem import ProcessadorImagem

JANELA_TEMPORAL_SEGUNDOS = 1.1

extrator = None
classificador = None
gerenciador = GerenciadorConexoes()
trava_extrator = asyncio.Lock()


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    global extrator, classificador
    extrator = ExtratorPontos()
    classificador = ClassificadorLibras()
    yield


app = FastAPI(
    title="Servidor Tradutor Libras",
    lifespan=ciclo_vida,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decodificar_e_extrair(bytes_imagem, imagem_base64):
    imagem_bgr = (
        ProcessadorImagem.decodificar_bytes(bytes_imagem)
        if bytes_imagem is not None
        else ProcessadorImagem.decodificar_base64(imagem_base64)
    )
    if imagem_bgr is None:
        return None, 0.0

    inicio_deteccao = time.perf_counter()
    resultado_extracao = extrator.extrair_marcos(imagem_bgr)
    tempo_deteccao_ms = round((time.perf_counter() - inicio_deteccao) * 1000, 2)
    return resultado_extracao, tempo_deteccao_ms


def _atualizar_trilhas_maos(trilhas, quadros_maos, agora_mono):
    trilhas[:] = [t for t in trilhas if (agora_mono - t["ultimo_instante"]) <= 1.1]
    limite_dist = 0.85 if (len(quadros_maos) == 1 and len(trilhas) == 1) else 0.65

    pares = []
    for idx_q, q in enumerate(quadros_maos):
        for idx_t, t in enumerate(trilhas):
            dist = math.hypot(q["centro"][0] - t["centro"][0], q["centro"][1] - t["centro"][1])
            if dist <= limite_dist:
                pares.append((dist, idx_q, idx_t))
    pares.sort(key=lambda item: item[0])

    usados_q = set()
    usados_t = set()
    trilhas_ativas = []

    for _, idx_q, idx_t in pares:
        if idx_q in usados_q or idx_t in usados_t:
            continue
        usados_q.add(idx_q)
        usados_t.add(idx_t)
        q = quadros_maos[idx_q]
        trilha = trilhas[idx_t]
        while trilha["quadros"] and (q["instante"] - trilha["quadros"][0]["instante"]) > JANELA_TEMPORAL_SEGUNDOS:
            trilha["quadros"].popleft()
        if len(trilha["quadros"]) >= 2 and (q["instante"] - trilha["quadros"][-2]["instante"]) < 0.11:
            trilha["quadros"][-1] = q
        else:
            trilha["quadros"].append(q)
        trilha["centro"] = q["centro"]
        trilha["ultimo_instante"] = q["instante"]
        trilhas_ativas.append(trilha)

    for idx_q, q in enumerate(quadros_maos):
        if idx_q in usados_q:
            continue
        nova_trilha = {
            "centro": q["centro"],
            "ultimo_instante": q["instante"],
            "quadros": deque([q], maxlen=9),
        }
        trilhas.append(nova_trilha)
        trilhas_ativas.append(nova_trilha)

    if len(trilhas) > 2:
        trilhas.sort(key=lambda t: t["ultimo_instante"], reverse=True)
        del trilhas[2:]

    return trilhas_ativas


def _classificar_trilhas(trilhas_ativas):
    avaliacoes = []
    menor_y = min(t["centro"][1] for t in trilhas_ativas)
    maior_y = max(t["centro"][1] for t in trilhas_ativas)

    for trilha in trilhas_ativas:
        vetor = montar_vetor_temporal(trilha["quadros"])
        res = classificador.prever(vetor)
        rotulo = res["previsao"]
        conf = float(res["confianca"])
        energia = float(vetor[104])
        em_mov = energia >= 0.28 or max(float(vetor[100]), float(vetor[101])) >= 0.24

        prioridade = conf
        if len(trilhas_ativas) > 1:
            if (maior_y - menor_y) > 0.14 and trilha["centro"][1] <= menor_y + 0.02:
                prioridade += 0.08
            if em_mov:
                prioridade += 0.14

        avaliacoes.append({
            "rotulo": rotulo,
            "confianca": conf,
            "prioridade": prioridade,
            "quadros": len(trilha["quadros"]),
            "em_movimento": em_mov,
        })

    if len(avaliacoes) == 2 and avaliacoes[0]["rotulo"] and avaliacoes[0]["rotulo"] == avaliacoes[1]["rotulo"]:
        melhor_conf = min(1.0, max(avaliacoes[0]["confianca"], avaliacoes[1]["confianca"]) + 0.05)
        melhor_quadros = max(avaliacoes[0]["quadros"], avaliacoes[1]["quadros"])
        em_mov_duplo = avaliacoes[0]["em_movimento"] or avaliacoes[1]["em_movimento"]
        return avaliacoes[0]["rotulo"], melhor_conf, melhor_quadros, em_mov_duplo

    melhor = max(avaliacoes, key=lambda item: item["prioridade"])
    return melhor["rotulo"], melhor["confianca"], melhor["quadros"], melhor["em_movimento"]


def _escolher_rotulo_votacao(historico_votos):
    pesos_base = [0.60, 0.85, 1.30]
    pesos = pesos_base[-len(historico_votos):]
    pontuacoes = {}
    soma_conf = {}
    contagens = {}

    for (rotulo, conf), peso in zip(historico_votos, pesos):
        if not rotulo:
            continue
        pontuacoes[rotulo] = pontuacoes.get(rotulo, 0.0) + conf * peso
        soma_conf[rotulo] = soma_conf.get(rotulo, 0.0) + conf
        contagens[rotulo] = contagens.get(rotulo, 0) + 1

    if not pontuacoes:
        return "", 0.0

    rotulo_consenso = max(pontuacoes, key=pontuacoes.get)
    votos = contagens[rotulo_consenso]
    media_conf = soma_conf[rotulo_consenso] / votos

    if (votos >= 2 and media_conf >= 0.38) or media_conf >= 0.50:
        return rotulo_consenso, round(media_conf, 2)
    return "", round(media_conf, 2)


@app.get("/")
def verificar_status():
    modelo_carregado = classificador.modelo is not None if classificador else False
    return {
        "status": "online",
        "modelo_carregado": modelo_carregado,
    }


@app.websocket("/ws")
async def rota_websocket(websocket: WebSocket):
    await gerenciador.conectar(websocket)
    trilhas_maos = []
    historico_votos = deque(maxlen=3)
    quadros_sem_deteccao = 0
    ultimo_instante_mao = 0.0
    estava_em_movimento = False

    try:
        while True:
            quadro_recebido = await websocket.receive()
            inicio = time.perf_counter()

            bytes_imagem = quadro_recebido.get("bytes")
            mensagem = quadro_recebido.get("text") or ""
            imagem_base64 = mensagem
            if mensagem.startswith("{"):
                try:
                    dados_json = json.loads(mensagem)
                    if dados_json.get("tipo") == "limpar_buffer":
                        trilhas_maos.clear()
                        historico_votos.clear()
                        quadros_sem_deteccao = 0
                        estava_em_movimento = False
                        continue
                    imagem_base64 = dados_json.get("imagem") or dados_json.get("frame") or ""
                except Exception:
                    pass

            async with trava_extrator:
                resultado_extracao, tempo_deteccao_ms = await asyncio.to_thread(
                    _decodificar_e_extrair, bytes_imagem, imagem_base64
                )

            if resultado_extracao is None:
                await gerenciador.enviar_json(websocket, {
                    "prediction": "",
                    "confidence": 0.0,
                    "previsao": "",
                    "confianca": 0.0,
                    "status": "imagem_invalida",
                })
                continue

            agora_mono = time.monotonic()
            quadros_maos = resultado_extracao.get("quadros_maos") or []
            if not resultado_extracao["tem_mao"] or not quadros_maos:
                quadros_sem_deteccao += 1
                if quadros_sem_deteccao >= 5 or (ultimo_instante_mao > 0 and (agora_mono - ultimo_instante_mao) > 1.1):
                    trilhas_maos.clear()
                    historico_votos.clear()
                    estava_em_movimento = False
                await gerenciador.enviar_json(websocket, {
                    "prediction": "",
                    "confidence": 0.0,
                    "previsao": "",
                    "confianca": 0.0,
                    "largura_quadro": resultado_extracao["largura_quadro"],
                    "altura_quadro": resultado_extracao["altura_quadro"],
                    "pontos_mao": [],
                    "pontos_maos": [],
                    "status": "nenhuma_mao_detectada",
                })
                continue

            quadros_sem_deteccao = 0
            ultimo_instante_mao = agora_mono

            trilhas_ativas = _atualizar_trilhas_maos(trilhas_maos, quadros_maos, agora_mono)

            inicio_classificacao = time.perf_counter()
            rotulo_bruto, conf_bruta, tamanho_buffer, em_movimento = _classificar_trilhas(trilhas_ativas)
            tempo_classificacao_ms = round((time.perf_counter() - inicio_classificacao) * 1000, 2)

            if em_movimento and not estava_em_movimento:
                votos_mesmo_rotulo = [v for v in historico_votos if v[0] == rotulo_bruto]
                historico_votos.clear()
                historico_votos.extend(votos_mesmo_rotulo)
            estava_em_movimento = em_movimento

            historico_votos.append((rotulo_bruto, conf_bruta))
            rotulo_final, conf_final = _escolher_rotulo_votacao(historico_votos)
            tempo_ms = round((time.perf_counter() - inicio) * 1000, 2)

            await gerenciador.enviar_json(websocket, {
                "prediction": rotulo_final,
                "confidence": conf_final,
                "previsao": rotulo_final,
                "confianca": conf_final,
                "tempo_ms": tempo_ms,
                "tempo_deteccao_ms": tempo_deteccao_ms,
                "tempo_classificacao_ms": tempo_classificacao_ms,
                "quadros_buffer": tamanho_buffer,
                "largura_quadro": resultado_extracao["largura_quadro"],
                "altura_quadro": resultado_extracao["altura_quadro"],
                "pontos_mao": resultado_extracao["pontos_mao"],
                "pontos_maos": resultado_extracao["pontos_maos"],
                "status": "ok",
            })

    except WebSocketDisconnect:
        gerenciador.desconectar(websocket)
    except Exception:
        gerenciador.desconectar(websocket)


@app.post("/traduzir-frame")
async def traduzir_frame_http(arquivo: UploadFile = File(...)):
    inicio = time.perf_counter()
    conteudo = await arquivo.read()

    async with trava_extrator:
        resultado_extracao, _ = await asyncio.to_thread(
            _decodificar_e_extrair, conteudo, ""
        )

    if resultado_extracao is None:
        return {"status": "erro", "mensagem": "Arquivo de imagem invalido"}

    quadros_maos = resultado_extracao.get("quadros_maos") or []
    if not resultado_extracao["tem_mao"] or not quadros_maos:
        return {
            "status": "ok",
            "previsao": "",
            "confianca": 0.0,
            "largura_quadro": resultado_extracao["largura_quadro"],
            "altura_quadro": resultado_extracao["altura_quadro"],
            "mensagem": "Nenhuma mao detectada",
        }

    trilhas_temp = [{"centro": q["centro"], "quadros": deque([q], maxlen=9)} for q in quadros_maos]
    rotulo, conf, _, _ = _classificar_trilhas(trilhas_temp)
    conf_arred = round(conf, 2)
    return {
        "prediction": rotulo,
        "confidence": conf_arred,
        "previsao": rotulo,
        "confianca": conf_arred,
        "tempo_ms": round((time.perf_counter() - inicio) * 1000, 2),
        "largura_quadro": resultado_extracao["largura_quadro"],
        "altura_quadro": resultado_extracao["altura_quadro"],
        "pontos_mao": resultado_extracao["pontos_mao"],
        "pontos_maos": resultado_extracao["pontos_maos"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("principal:app", host="0.0.0.0", port=8000, reload=True)
