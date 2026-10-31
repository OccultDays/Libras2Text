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
        self.classes = []
        self._arvores_rapidas = []
        self.carregar_modelo()

    def carregar_modelo(self):
        if not os.path.exists(self.caminho_modelo):
            return False

        self.modelo = joblib.load(self.caminho_modelo)
        if hasattr(self.modelo, "n_jobs"):
            self.modelo.n_jobs = 1

        if os.path.exists(self.caminho_rotulos):
            self.rotulos = joblib.load(self.caminho_rotulos)

        classes_modelo = getattr(self.modelo, "classes_", None)
        if classes_modelo is not None:
            self.classes = []
            for item in classes_modelo:
                if self.rotulos is not None and isinstance(item, (int, np.integer)) and 0 <= item < len(self.rotulos):
                    self.classes.append(str(self.rotulos[item]))
                else:
                    self.classes.append(str(item))

        self._arvores_rapidas = []
        for estimador in getattr(self.modelo, "estimators_", []):
            arvore = estimador.tree_
            valores = arvore.value[:, 0, :]
            somas = np.maximum(valores.sum(axis=1, keepdims=True), 1e-9)
            folhas_prob = valores / somas
            self._arvores_rapidas.append((
                arvore.children_left.tolist(),
                arvore.children_right.tolist(),
                arvore.feature.tolist(),
                arvore.threshold.tolist(),
                folhas_prob,
            ))

        return True

    def prever(self, vetor_caracteristicas):
        if self.modelo is None:
            return {
                "prediction": "Modelo nao carregado",
                "confidence": 0.0,
                "previsao": "Modelo nao carregado",
                "confianca": 0.0,
            }

        if self._arvores_rapidas and self.classes:
            x = [float(v) for v in vetor_caracteristicas]
            acumulado = np.zeros(len(self.classes), dtype=np.float64)
            for esquerda, direita, caracteristica, limite, folhas_prob in self._arvores_rapidas:
                no = 0
                while esquerda[no] != -1:
                    if x[caracteristica[no]] <= limite[no]:
                        no = esquerda[no]
                    else:
                        no = direita[no]
                acumulado += folhas_prob[no]

            probabilidades = acumulado / len(self._arvores_rapidas)
            melhor_indice = int(np.argmax(probabilidades))
            rotulo_final = self.classes[melhor_indice]
            confianca = float(probabilidades[melhor_indice])
        else:
            amostra = np.asarray(vetor_caracteristicas, dtype=np.float64).reshape(1, -1)
            if hasattr(self.modelo, "predict_proba"):
                probabilidades = self.modelo.predict_proba(amostra)[0]
                melhor_indice = int(np.argmax(probabilidades))
                predicao = self.modelo.classes_[melhor_indice]
                confianca = float(probabilidades[melhor_indice])
            else:
                predicao = self.modelo.predict(amostra)[0]
                confianca = 1.0

            rotulo_final = str(predicao)
            if self.rotulos is not None and isinstance(predicao, (int, np.integer)):
                if 0 <= predicao < len(self.rotulos):
                    rotulo_final = str(self.rotulos[predicao])

        confianca_arredondada = round(confianca, 2)
        return {
            "prediction": rotulo_final,
            "confidence": confianca_arredondada,
            "previsao": rotulo_final,
            "confianca": confianca_arredondada,
        }
