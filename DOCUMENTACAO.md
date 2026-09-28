# Documentacao do Projeto: Tradutor de Libras em Tempo Real (Libras2Text)

Este projeto consiste em um sistema completo para traducao de sinais da Lingua Brasileira de Sinais (Libras) capturados pela camera do celular para texto em tempo real. A solucao utiliza uma arquitetura cliente-servidor de baixa latencia, baseada em Expo Go (React Native) no aplicativo movel e FastAPI com MediaPipe e Scikit-Learn no servidor backend.

---

## 1. Visao Geral da Arquitetura

O sistema opera de forma assincrona e distribuida atraves de comunicacao bidirecional via WebSocket:

1. **Captura no Celular (Frontend)**: O app Expo Go acessa a camera do dispositivo, captura quadros otimizados em intervalos controlados (throttle a cada 400 ms) em formato Base64 e os envia para o servidor via WebSocket.
2. **Processamento e Decodificacao (Backend)**: O servidor FastAPI recebe os dados em Base64, decodifica para matrizes NumPy e converte para o espaco de cor adequado atraves do OpenCV.
3. **Extracao de Marcos da Cabeca, Bracos e Maos (MediaPipe)**: O modulo de visao computacional extrai as coordenadas espaciais 3D (x, y, z) de 34 pontos anatomicos atraves da execucao simultanea de `PoseLandmarker` (cabeca, ombros, cotovelos e pulsos) e `HandLandmarker` (21 pontos dos dedos).
4. **Normalizacao Matematica**: Os pontos sao transladados em relacao ao centro dos ombros (ou pulso) e escalados pela distancia euclidiana maxima, tornando o reconhecimento invariante a distancia da camera e enquadramento.
5. **Inferencia de Machine Learning**: O classificador treinado com Scikit-Learn e carregado na memoria no ciclo de vida da aplicacao e realiza a predicao do sinal ou palavra em menos de 15 ms por quadro.
6. **Retorno ao Cliente**: O servidor responde com o caractere detectado, a confianca da previsao e a latencia do processamento, atualizando a interface do usuario instantaneamente.

---

## 2. Estrutura de Diretorios do Projeto

```
Libras2Text/
├── DOCUMENTACAO.md                     # Documento detalhado com toda a explicacao do projeto
├── servidor/                           # Modulo do Backend e Servicos de Visao Computacional
│   ├── principal.py                    # Servidor FastAPI com endpoints WebSocket e HTTP
│   ├── extrator_pontos.py              # Extracao dos 21 marcos e normalizacao euclidiana
│   ├── processador_imagem.py           # Decodificacao de imagens em formato Base64
│   ├── classificador_libras.py         # Carregamento do modelo serializado e inferencia
│   ├── gerenciador_conexoes.py         # Gerenciamento persistente das conexoes WebSocket
│   ├── requisitos.txt                  # Dependencias Python do servidor
│   └── modelos_ia/                     # Arquivos de modelos serializados
│       ├── marcador_maos.task          # Modelo neural do MediaPipe HandLandmarker
│       ├── modelo_libras.joblib        # Classificador Random Forest treinado
│       └── rotulos.joblib              # Lista de classes e caracteres reconhecidos
├── modelo/                             # Modulo de Treinamento e Criacao do Dataset
│   ├── gerar_dataset_inicial.py        # Gerador sintese de configuracoes geometricas de Libras
│   ├── coletar_dados.py                # Script interativo para gravacao de novos sinais via webcam
│   ├── treinar_modelo.py               # Treinamento do modelo, avaliacao e exportacao joblib
│   └── dataset_libras.csv              # Dataset com as amostras de coordenadas normalizadas
└── aplicativo/                         # Frontend React Native utilizando Expo Go
    ├── App.js                          # Componente principal da aplicacao movel
    ├── app.json                        # Configuracoes do projeto Expo e permissoes de camera
    ├── package.json                    # Dependencias do ecossistema JavaScript/Node
    └── assets/                         # Icones e recursos visuais do aplicativo
```

---

## 3. Detalhamento Tecnico dos Componentes

### 3.1 Servidor Backend (`servidor/`)

- **`principal.py`**:
  - Implementa o gerenciador de ciclo de vida (`lifespan`) do FastAPI, carregando o modelo do classificador e o detector do MediaPipe na memoria RAM durante a inicializacao.
  - Disponibiliza o endpoint WebSocket `/ws` para transmissao continua de frames e desconexao limpa via `WebSocketDisconnect`.
  - Fornece um endpoint HTTP `/traduzir-frame` via POST multipart para testes pontuais de imagens.
  - Implementa politicas de CORS liberando acesso da rede local.

- **`extrator_pontos.py`**:
  - Integra simultaneamente o `PoseLandmarker` e o `HandLandmarker` da API de tarefas do MediaPipe.
  - Garante o download automatico dos modelos neurais (`marcador_maos.task` e `marcador_pose.task`).
  - Extrai 34 marcos espaciais (102 dimensoes):
    1. **Cabeca (7 marcos da pose)**: nariz, olhos, orelhas e extremidades da boca, capturando inclinacao, movimentos afirmativos/negativos e expressoes nao-manuais (ENM).
    2. **Bracos e Tronco (6 marcos da pose)**: ombros, cotovelos e pulsos, capturando o ponto de articulacao e alcance dos bracos.
    3. **Mao (21 marcos da mao)**: articulacoes e pontas de todos os dedos, identificando a configuracao de mao (CM).
  - Aplica normalizacao euclidiana relativa ao centro dos ombros (ou pulso) e escala maxima, tornando o reconhecimento invariante a distancia e tamanho corporal.

- **`processador_imagem.py`**:
  - Processa a string Base64 recebida, remove eventuais cabecalhos MIME (`data:image/jpeg;base64,`) e decodifica os bytes em array NumPy via `cv2.imdecode`.

- **`classificador_libras.py`**:
  - Carrega `modelo_libras.joblib` e a lista correspondente de rotulos em `rotulos.joblib`.
  - Executa inferencia atraves de `modelo.predict()` e extrai a probabilidade com `modelo.predict_proba()`.
  - Retorna o resultado com o sinal ou palavra prevista e o indice de confianca numerico.

- **`gerenciador_conexoes.py`**:
  - Mantem o registro das conexoes WebSocket ativas e realiza o envio seguro de mensagens em formato JSON.

### 3.2 Modulo de Inteligencia Artificial (`modelo/`)

- **`gerar_dataset_inicial.py`**:
  - Modela as posicoes anatomicas dos cinco dedos, dos bracos e a inclinacao da cabeca para sinais completos de Libras ("OI", "BOM DIA", "OBRIGADO", "POR FAVOR", "SIM", "NAO", "EU", "VOCE", "CASA", "AJUDA") e configuracoes basicas do alfabeto ("A", "B", "C", "L", "V").
  - Gera amostras com 102 caracteristicas geometricas normalizadas com variacoes gaussianas.
  - Salva os dados no arquivo `dataset_libras.csv`.

- **`coletar_dados.py`**:
  - Permite ao desenvolvedor/aluno usar a webcam do computador para gravar amostras reais de sinais capturando cabeca, bracos e maos ao pressionar a barra de espaco.

- **`treinar_modelo.py`**:
  - Carrega `dataset_libras.csv`, divide os dados em 80% treino e 20% teste com amostragem estratificada.
  - Treina um classificador `RandomForestClassifier` com 100 estimadores.
  - Exibe relatorio de classificacao com precisao, revocacao e F1-Score.
  - Salva os artefatos `modelo_libras.joblib` e `rotulos.joblib` diretamente na pasta `servidor/modelos_ia/`.

### 3.3 Aplicativo Movel (`aplicativo/`)

- **Fluxo de Permissoes**: Utiliza o hook `useCameraPermissions()` do `expo-camera` para verificar e solicitar acesso a camera do dispositivo de forma clara.
- **Amostragem Temporal Rapida (180 ms)**: Taxa de amostragem reduzida para 180 ms (~5.5 quadros por segundo), fornecendo densidade suficiente de amostras para capturar a trajetoria de gestos que exigem mais tempo.
- **Buffer Deslizante Temporal (10 Quadros)**: O servidor mantem uma fila deslizante dos ultimos 10 quadros recebidos (~1.8 segundos de movimento), calculando o vetor de deslocamento espacial de inicio a fim do gesto (204 dimensoes: 102 de marcos atuais + 102 de trajetoria).
- **Modo "Gravar Gesto Longo (2.5s)"**: Botao dedicado com contagem regressiva previa de 3 segundos para preparacao e posicionamento do usuario (`Prepare-se em 3s... 2s... 1s...`), seguido por 2.5 segundos de gravacao continua da trajetoria do sinal, adicionando a palavra traduzida diretamente ao texto final.
- **Captura Continua sem Piscadas**: A propriedade `animateShutter={false}` no `CameraView` e `shutterSound: false` na captura desativam a animacao nativa de obturador e o som de disparo, garantindo que o visor permaneca estavel e sem nenhum piscar de tela durante a traducao em tempo real.
- **Otimizacao de Imagem**: Captura os quadros com qualidade controlada (`quality: 0.3`) e obtencao direta em Base64, evitando processamentos desnecessarios no dispositivo.
- **Limpeza de Cache**: O arquivo temporario criado pelo metodo de foto e removido em seguida com `FileSystem.deleteAsync` para evitar ocupacao indevida de espaco em disco no celular.
- **Cliente WebSocket**: Conecta via `new WebSocket(url)` com reconexao automatica, exibindo o status atual (Conectado, Conectando, Desconectado ou Erro) e contador de quadros no buffer (`Buffer: X/10`).
- **Alternancia Segura de Camera (Frontal e Traseira)**: Permite alternar entre a camera frontal e traseira atraves do botao de acao ou da etiqueta visual. O ciclo de vida do sensor e gerenciado com o evento `onCameraReady`, bloqueando temporariamente novos disparos de fotos durante a mudanca do hardware para prevenir travamentos nos drivers nativos da camera.
- **Design Minimalista**: Interface construida com tons de branco (`#FFFFFF`, `#F8F9FA`), bordas sutis e tipografia escura, sem icones ou cores vibrantes desnecessarias.

---

## 4. Guia de Instalacao e Execucao

### 4.1 Instalacao das Dependencias do Servidor

No terminal, certifique-se de que o Python esteja instalado e execute:

```bash
pip install -r servidor/requisitos.txt
```

### 4.2 Treinamento do Modelo de IA

Para treinar o modelo com os dados de base e gerar os arquivos serializados:

```bash
python modelo/treinar_modelo.py
```

### 4.3 Inicializacao do Servidor FastAPI

Para iniciar o servidor FastAPI e escutar em todas as interfaces da rede local (porta 8000):

```bash
cd servidor
python -m uvicorn principal:app --host 0.0.0.0 --port 8000 --reload
```

Ou execute diretamente:

```bash
python servidor/principal.py
```

### 4.4 Configuracao da Rede e Firewall

1. Descubra o IP local do computador executando `ipconfig` (no Windows) ou `ifconfig`/`ip addr` (no Linux/Mac). Geralmente algo como `192.168.0.XX`.
2. Certifique-se de que o computador e o celular estao conectados na mesma rede Wi-Fi.
3. Se necessario, permita o trafego da porta 8000 no Firewall do Windows:
   - Abra o *Firewall do Windows com Seguranca Avancada*.
   - Crie uma *Regra de Entrada* para a porta TCP `8000`.

### 4.5 Execucao do Aplicativo Expo Go

No terminal, acesse a pasta do aplicativo e inicie o Expo:

```bash
cd aplicativo
npx expo start
```

1. Abra o app **Expo Go** no celular.
2. Escaneie o QR Code exibido no terminal.
3. No campo "IP do Servidor" no topo do aplicativo, digite o IP local do seu computador (exemplo: `192.168.0.12`) e toque em **Conectar**.
4. Toque em **Iniciar Traducao** e posicione a mao em frente a camera.
