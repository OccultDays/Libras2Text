import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

import sys
from pathlib import Path

diretorio_servidor = Path(__file__).resolve().parent
if str(diretorio_servidor) not in sys.path:
    sys.path.append(str(diretorio_servidor))

from extrator_pontos import ExtratorPontos
from processador_imagem import ProcessadorImagem
from classificador_libras import ClassificadorLibras
from gerenciador_conexoes import GerenciadorConexoes

extrator = None
classificador = None
gerenciador = GerenciadorConexoes()

@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    global extrator, classificador
    extrator = ExtratorPontos()
    classificador = ClassificadorLibras()
    yield

app = FastAPI(
    title="Servidor Tradutor Libras",
    lifespan=ciclo_vida
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def verificar_status():
    modelo_carregado = classificador.modelo is not None if classificador else False
    return {
        "status": "online",
        "modelo_carregado": modelo_carregado
    }

from collections import deque

@app.websocket("/ws")
async def rota_websocket(websocket: WebSocket):
    await gerenciador.conectar(websocket)
    historico_marcos = deque(maxlen=10)
    try:
        while True:
            mensagem = await websocket.receive_text()
            inicio = time.time()

            imagem_base64 = mensagem
            if mensagem.startswith("{"):
                import json
                try:
                    dados_json = json.loads(mensagem)
                    if dados_json.get("tipo") == "limpar_buffer":
                        historico_marcos.clear()
                        continue
                    imagem_base64 = dados_json.get("imagem") or dados_json.get("frame") or ""
                except Exception:
                    pass

            imagem_bgr = ProcessadorImagem.decodificar_base64(imagem_base64)
            if imagem_bgr is None:
                await gerenciador.enviar_json(websocket, {
                    "prediction": "",
                    "confidence": 0.0,
                    "previsao": "",
                    "confianca": 0.0,
                    "status": "imagem_invalida"
                })
                continue

            marcos_atuais = extrator.extrair_marcos(imagem_bgr)
            if marcos_atuais is None:
                historico_marcos.clear()
                await gerenciador.enviar_json(websocket, {
                    "prediction": "",
                    "confidence": 0.0,
                    "previsao": "",
                    "confianca": 0.0,
                    "status": "nenhum_sinal_detectado"
                })
                continue

            historico_marcos.append(marcos_atuais)
            marcos_iniciais = historico_marcos[0]
            deslocamento = marcos_atuais - marcos_iniciais
            vetor_temporal = np.concatenate([marcos_atuais, deslocamento])

            resultado = classificador.prever(vetor_temporal)
            tempo_ms = round((time.time() - inicio) * 1000, 2)
            resultado["tempo_ms"] = tempo_ms
            resultado["quadros_buffer"] = len(historico_marcos)
            resultado["status"] = "ok"

            await gerenciador.enviar_json(websocket, resultado)

    except WebSocketDisconnect:
        gerenciador.desconectar(websocket)
    except Exception:
        gerenciador.desconectar(websocket)

@app.post("/traduzir-frame")
async def traduzir_frame_http(arquivo: UploadFile = File(...)):
    inicio = time.time()
    conteudo = await arquivo.read()
    vetor_np = np.frombuffer(conteudo, dtype=np.uint8)
    imagem_bgr = cv2.imdecode(vetor_np, cv2.IMREAD_COLOR)

    if imagem_bgr is None:
        return {"status": "erro", "mensagem": "Arquivo de imagem invalido"}

    marcos = extrator.extrair_marcos(imagem_bgr)
    if marcos is None:
        return {"status": "ok", "previsao": "", "confianca": 0.0, "mensagem": "Nenhuma mao detectada"}

    deslocamento = np.zeros_like(marcos)
    vetor_temporal = np.concatenate([marcos, deslocamento])
    resultado = classificador.prever(vetor_temporal)
    resultado["tempo_ms"] = round((time.time() - inicio) * 1000, 2)
    return resultado

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("principal:app", host="0.0.0.0", port=8000, reload=True)
