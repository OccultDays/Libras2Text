import os
import joblib
import numpy as np

class ClassificadorLibras:
    def __init__(self, caminho_modelo=None, caminho_rotulos=None):
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        
        if caminho_modelo is None:
            caminho_modelo = os.path.join(diretorio_atual, "modelos_ia", "modelo_libras.joblib")
        if caminho_rotulos is None:
            caminho_rotulos = os.path.join(diretorio_atual, "modelos_ia", "rotulos.joblib")

        self.caminho_modelo = caminho_modelo
        self.caminho_rotulos = caminho_rotulos
        self.modelo = None
        self.rotulos = None
        self.carregar_modelo()

    def carregar_modelo(self):
        if os.path.exists(self.caminho_modelo):
            self.modelo = joblib.load(self.caminho_modelo)
            if os.path.exists(self.caminho_rotulos):
                self.rotulos = joblib.load(self.caminho_rotulos)
            return True
        return False

    def prever(self, vetor_caracteristicas):
        if self.modelo is None:
            return {
                "prediction": "Modelo nao carregado",
                "confidence": 0.0,
                "previsao": "Modelo nao carregado",
                "confianca": 0.0
            }

        amostra = np.array(vetor_caracteristicas).reshape(1, -1)
        predicao = self.modelo.predict(amostra)[0]
        
        confianca = 1.0
        if hasattr(self.modelo, "predict_proba"):
            probabilidades = self.modelo.predict_proba(amostra)[0]
            confianca = float(np.max(probabilidades))

        rotulo_final = str(predicao)
        if self.rotulos is not None and isinstance(predicao, (int, np.integer)):
            if 0 <= predicao < len(self.rotulos):
                rotulo_final = str(self.rotulos[predicao])

        return {
            "prediction": rotulo_final,
            "confidence": round(confianca, 2),
            "previsao": rotulo_final,
            "confianca": round(confianca, 2)
        }
