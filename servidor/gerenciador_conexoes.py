from fastapi import WebSocket

class GerenciadorConexoes:
    def __init__(self):
        self.conexoes_ativas = []

    async def conectar(self, websocket: WebSocket):
        await websocket.accept()
        self.conexoes_ativas.append(websocket)

    def desconectar(self, websocket: WebSocket):
        if websocket in self.conexoes_ativas:
            self.conexoes_ativas.remove(websocket)

    async def enviar_json(self, websocket: WebSocket, dados: dict):
        await websocket.send_json(dados)
