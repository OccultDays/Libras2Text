# Documentacao do Projeto: Tradutor de Libras em Tempo Real (Libras2Text)

Este projeto consiste em um sistema para traducao de sinais da Lingua Brasileira de Sinais (Libras) capturados pela camera do celular para texto. No Android, a captura continua, o rastreamento MediaPipe e o classificador Random Forest sao executados no proprio aparelho por um modulo nativo; o servidor FastAPI deixa de participar do caminho de traducao. O app usa Expo SDK 58 / React Native 0.88 e requer um development build Android, pois Expo Go nao contem o modulo nativo local.

---

## 1. Visao Geral da Arquitetura

No Android, o fluxo de traducao local e:

1. **Captura nativa**: CameraX fornece quadros RGBA continuamente em 640x480 e descarta quadros atrasados para nunca formar fila.
2. **Rastreamento local**: MediaPipe `PoseLandmarker` e `HandLandmarker`, ambos em modo `VIDEO`, identificam cabeca, bracos e ate duas maos no proprio Android.
3. **Caracteristicas e classificacao local**: Com um modelo marcado como treinado em dados reais, o app normaliza os mesmos 102 valores espaciais usados pelo backend, acrescenta 102 valores de deslocamento temporal e executa as arvores do Random Forest no aparelho. O artefato embarcado e `aplicativo/assets/modelo_libras_mobile.json`; o modelo demonstrativo atual nao habilita traducoes.
4. **Interface**: Pontos, previsao e confianca sao atualizados com o evento nativo. Imagens e caracteristicas de traducao nao sao enviadas ao servidor no Android.

No iOS, o fluxo antigo com Expo Camera e WebSocket permanece como fallback. O modulo de processamento local desta entrega e Android.

### 1.1 Desempenho e limites atuais

- No Android, nao ha captura JPEG, leitura de arquivo temporario, trafego WebSocket nem ida e volta ao backend durante a traducao. A camera usa CameraX com `STRATEGY_KEEP_ONLY_LATEST`.
- A interface exibe `Inferencia local`, que mede o tempo de processamento do quadro no modulo nativo MediaPipe; nao inclui toda a latencia de exibicao/atualizacao da interface.
- Para executar no Android, instale o development build criado com `npx expo run:android`. Expo Go nao pode carregar o modulo Kotlin e os modelos nativos.
- O modelo atual foi treinado com dados sinteticos de demonstracao e pode continuar errando sinais mesmo com a inferencia local. On-device reduz dependencia e latencia de rede, nao melhora por si so a qualidade do treinamento.
- O artefato mobile atual esta marcado como `sintetico_demo`. No Android, ele nao deve gerar ou inserir traducoes ate ser substituido por um modelo treinado com `dataset_libras_real.csv`; isso evita apresentar palpites do conjunto geometrico sintetico como Libras correta. O rastreamento visual continua disponivel.

---

## 2. Estrutura de Diretorios do Projeto

```
Libras2Text/
├── DOCUMENTACAO.md                     # Documento detalhado com toda a explicacao do projeto
├── servidor/                           # Modulo do Backend e Servicos de Visao Computacional
│   ├── principal.py                    # Servidor FastAPI com endpoints WebSocket e HTTP
│   ├── extrator_pontos.py              # Extracao dos 21 marcos e normalizacao euclidiana
│   ├── processador_imagem.py           # Decodificacao de imagens em bytes ou Base64
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
│   ├── exportar_modelo_mobile.py        # Conversao do Random Forest para inferencia Android local
│   └── dataset_libras.csv              # Dataset com as amostras de coordenadas normalizadas
└── aplicativo/                         # React Native com development build Android
    ├── App.js                          # Componente principal da aplicacao movel
    ├── app.json                        # Configuracoes do projeto Expo e permissoes de camera
    ├── package.json                    # Dependencias e scripts do development build
    ├── assets/                         # Icones e modelo Random Forest embarcado
    └── modules/libras-native/           # CameraX + MediaPipe nativos para Android
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
  - Integra simultaneamente o `PoseLandmarker` e o `HandLandmarker` da API de tarefas do MediaPipe, em modo `VIDEO` para aproveitar o rastreamento entre quadros.
  - Garante o download automatico dos modelos neurais (`marcador_maos.task` e `marcador_pose.task`).
  - Carrega os arquivos `.task` diretamente na memoria em bytes (`model_asset_buffer`), prevenindo falhas de leitura do MediaPipe em diretorios com caracteres especiais ou acentos no Windows.
  - A entrada atual do classificador usa 34 marcos espaciais (102 dimensoes):
    1. **Cabeca (7 marcos da pose)**: nariz, olhos, orelhas e extremidades da boca, capturando inclinacao, movimentos afirmativos/negativos e expressoes nao-manuais (ENM).
    2. **Bracos e Tronco (6 marcos da pose)**: ombros, cotovelos e pulsos, capturando o ponto de articulacao e alcance dos bracos.
    3. **Mao (21 marcos da mao)**: articulacoes e pontas de todos os dedos, identificando a configuracao de mao (CM).
  - Detecta ate duas maos e envia ambas para a interface desenhar os pontos e conexoes verdes/vermelhos. Para preservar a compatibilidade com o classificador atual, seleciona para a inferencia a mao mais proxima de um dos pulsos detectados.
  - Aplica normalizacao euclidiana relativa ao centro dos ombros (ou pulso) e escala maxima, tornando o reconhecimento invariante a distancia e tamanho corporal.

- **`processador_imagem.py`**:
  - Decodifica quadros JPEG enviados como bytes (ou Base64 por clientes antigos) em array NumPy via `cv2.imdecode`.

- **`classificador_libras.py`**:
  - Carrega `modelo_libras.joblib` e a lista correspondente de rotulos em `rotulos.joblib`.
  - Executa inferencia atraves de `modelo.predict()` e extrai a probabilidade com `modelo.predict_proba()`.
  - Retorna o resultado com o sinal ou palavra prevista e o indice de confianca numerico.

- **`gerenciador_conexoes.py`**:
  - Mantem o registro das conexoes WebSocket ativas e realiza o envio seguro de mensagens em formato JSON.

### 3.2 Modulo de Inteligencia Artificial (`modelo/`)

- **`gerar_dataset_inicial.py`**:
  - Gera exemplos geometricos sinteticos para demonstracao nos rotulos previstos pelo projeto. Esses exemplos nao substituem gravações reais e nao medem a precisao de Libras no mundo real.
  - Salva os dados sinteticos no arquivo `dataset_libras.csv`.

- **`coletar_dados.py`**:
  - Grava sequencias reais de 1,5 segundo pela webcam. Pressione ESPACO para cada exemplo; execute o sinal durante a gravacao.
  - Salva 204 caracteristicas temporais em `dataset_libras_real.csv`, sem sobrescrever o conjunto sintetico. Colete pelo menos 30 exemplos por rotulo, variando pessoas e execucoes sempre que possivel.
  - O classificador atual usa os pontos da cabeca, bracos e de uma mao. Sinais que dependem das duas maos exigem uma ampliacao do vetor de caracteristicas e novas amostras antes de serem reconhecidos de forma confiavel.

- **`treinar_modelo.py`**:
  - Prefere `dataset_libras_real.csv` quando disponivel; usa `dataset_libras.csv` apenas como fallback de demonstracao.
  - Divide os dados em treino e teste com amostragem estratificada.
  - Treina um classificador `RandomForestClassifier` com 100 estimadores.
  - Exibe relatorio de classificacao com precisao, revocacao e F1-Score.
  - Salva os artefatos `modelo_libras.joblib` e `rotulos.joblib` diretamente na pasta `servidor/modelos_ia/` e atualiza `aplicativo/assets/modelo_libras_mobile.json` para inferencia no Android.
  - O JSON exportado inclui `fonte_dados` (`real` ou `sintetico_demo`). A aplicacao Android so permite traducao quando o artefato esta marcado como `real`.
- **`exportar_modelo_mobile.py`**: exporta as arvores e probabilidades do modelo treinado para JSON, sem depender do servidor durante a inferencia Android.

### 3.3 Aplicativo Movel (`aplicativo/`)

- **Fluxo de Permissoes**: Utiliza o hook `useCameraPermissions()` do `expo-camera` para verificar e solicitar acesso a camera do dispositivo de forma clara.
- **Android nativo**: `modules/libras-native` fornece pre-visualizacao CameraX e executa MediaPipe Pose e Hand Landmarkers em um executor nativo. O componente JS resolve a view com `requireNativeViewManager("LibrasNative", "LibrasNativeCameraView")`, usando o nome do modulo e o nome da view registrados em Kotlin. Usa resolucao 640x480 e `STRATEGY_KEEP_ONLY_LATEST`; nao grava fotos em cache.
- **Inferencia local**: `modelo/exportar_modelo_mobile.py` converte o Random Forest Scikit-Learn para arrays JSON de nos de arvore. O app calcula probabilidades no dispositivo, com 102 caracteristicas normalizadas e 102 de deslocamento.
- **Buffer temporal local (10 quadros)**: Mantido em memoria no app para gerar caracteristicas de movimento e limpo ao iniciar uma gravacao de gesto.
- **Fallback iOS**: Continua usando Expo Camera, JPEG e WebSocket para o backend; o modulo nativo desta entrega cobre Android.
- **Modo "Gravar Gesto Longo (2.5s)"**: Mantem contagem regressiva de 3 segundos e acumula a melhor predicao durante 2.5 segundos de rastreamento local no Android.
- **Indicador de latencia**: Em Android, `Inferencia local` inclui a conversao do quadro, os dois MediaPipe Landmarkers e a classificacao no aparelho; o desenho final dos pontos ainda pode adicionar latencia de interface.
- **Troca de camera**: No Android, CameraX permite alternar entre lentes frontal e traseira sem voltar ao envio JPEG/WebSocket.
- **iOS**: Mantem a implementacao Expo Camera antiga e depende do servidor ate a criacao do modulo nativo iOS.
- **Design Minimalista**: Interface construida com tons de branco (`#FFFFFF`, `#F8F9FA`), bordas sutis e tipografia escura, sem icones ou cores vibrantes desnecessarias.

---

## 4. Guia de Instalacao e Execucao

### 4.1 Instalacao das Dependencias do Servidor

No terminal, certifique-se de que o Python esteja instalado e execute:

```bash
pip install -r servidor/requisitos.txt
```

### 4.2 Treinamento do Modelo de IA

Para melhorar o reconhecimento, primeiro colete exemplos reais. Repita o comando para cada rotulo e grave pelo menos 30 exemplos por sinal:

```bash
python modelo/coletar_dados.py
```

Depois treine e gere os arquivos serializados. Se ainda nao houver dados reais, o script avisa que usara o conjunto sintetico de demonstracao:

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

### 4.5 Build e execucao Android (SDK 58)

Este computador foi preparado com Android Studio, Eclipse Temurin JDK 17, Android SDK Platform 37.2, Android Build Tools 37.0.0, Platform Tools (`adb`) e Command-line Tools. As variaveis `JAVA_HOME`, `ANDROID_HOME` e `ANDROID_SDK_ROOT` foram configuradas no perfil do usuario; abra um novo terminal para carrega-las. Em outra maquina, instale Android Studio, Android SDK e JDK 17 ou compativel.

Conecte o Android por USB, ative Opcoes do desenvolvedor e Depuracao USB, aceite a autorizacao exibida no aparelho e confirme que `adb devices` mostra o dispositivo. Depois execute:

```bash
cd aplicativo
npm run android
```

O comando gera os projetos nativos, compila e instala um development build personalizado no celular. Depois da primeira instalacao, use `npm run start:dev-client` para iniciar o Metro. Alteracoes JavaScript podem ser recarregadas sem recompilar; alteracoes Kotlin, dependencias nativas ou modelos `.task` exigem um novo build.

Expo Go nao pode carregar o modulo CameraX/MediaPipe deste projeto. Ao atualizar o SDK, alinhe as dependencias nativas com `npx expo install --fix` e confira com `npx expo-doctor`.
O primeiro build Android tambem instala os modelos `.task` dentro do APK. `npm run start:dev-client` inicia o Metro para o development build ja instalado; ele nao substitui a compilacao e instalacao do APK.

### Erro `Unable to load script` no development build

Em 29/10/2026, o celular mostrou a tela vermelha porque o Metro nao estava acessivel apos o processo de instalacao/build terminar. O APK instalado e um development build que carrega o JavaScript do Metro, portanto o servidor precisa continuar rodando durante o uso. Com o aparelho conectado via USB, execute `adb reverse tcp:8081 tcp:8081` e mantenha `npm run start:dev-client` ativo na pasta do app. Nesta ocorrencia a porta 8081 ja tinha um Metro respondendo; restabelecemos o encaminhamento USB e reiniciamos o app, que voltou a carregar e mostrou a interface. Se Metro nao responder em `http://127.0.0.1:8081/status`, inicie-o antes de recarregar o celular.

#### Observacoes de build no Windows

O projeto original estava dentro de um caminho com o caractere `º`; o Gradle nao conseguiu resolver corretamente os plugins nativos a partir desse caminho. A compilacao tambem encontrou o limite de 260 caracteres do Ninja/CMake. O React Native 0.88 exige a Nova Arquitetura; por isso, `aplicativo/android/app/build.gradle` define `CMAKE_OBJECT_PATH_MAX=200`, fazendo o CMake encurtar os caminhos dos objetos. A versao Android foi colocada em `C:\a` para manter curtos os caminhos que o Ninja calcula. As dependencias ficam em `C:\n\node_modules`, acessiveis por uma junction `C:\a\node_modules`. Execute `npm run android` a partir de `C:\a`; evite caminhos longos ou com caracteres especiais durante o build.
### Correcao da previa preta no Android

No Samsung SM-S721B, o `PreviewView` permanecia em `IDLE` mesmo recebendo a solicitacao de superficie do CameraX. A pre-visualizacao foi alterada para uma `TextureView` com `SurfaceTexture` e `Surface` fornecidas diretamente ao `Preview` do CameraX. A inicializacao espera a view estar dimensionada e a superficie disponivel; vincula `Preview` e `ImageAnalysis` juntos, mantendo o rastreamento local. Um identificador descarta inicializacoes antigas quando a tela ou a lente muda. O app marca a camera como pronta ao receber os primeiros quadros. A versao foi compilada, instalada e verificada no celular: a imagem da camera frontal apareceu e o indicador mudou para "Camera: Frontal". A validacao desta correcao foi focada na camera frontal. Para Metro via USB, use `adb reverse tcp:8081 tcp:8081`; no Windows, `NODE_OPTIONS=--dns-result-order=ipv4first` faz o servidor localhost escutar em IPv4. Alteracoes Kotlin exigem recompilar e reinstalar (`cd C:\a; npm run android`).

### Qualidade do rastreamento e da traducao local

O analisador CameraX so envia um vetor ao classificador quando Pose e Hand Landmarker encontram ambos os alvos e o pulso da mao detectada fica a ate 0,18 da coordenada normalizada de um pulso da pose. Se faltar mao, corpo ou essa associacao, os pontos disponiveis continuam desenhados, mas o vetor incompleto e descartado e a janela temporal e reiniciada; antes, marcos ausentes eram preenchidos com zeros/pulsos e podiam virar uma traducao aparentemente confiante. A coleta nativa e a extracao do servidor usam limiares de deteccao mais restritivos e iguais. A insercao continua tambem exige seis quadros estaveis e confianca de pelo menos 70%.

O modelo mobile versionado neste repositorio foi treinado com configuracoes geometricas sinteticas, nao com sinais gravados. Por isso, a traducao Android fica bloqueada enquanto `fonte_dados` nao for `real`; o app informa que e necessario treinar com exemplos reais em vez de inventar uma palavra. Colete exemplos por sinal e por diferentes pessoas/variacoes com `python modelo/coletar_dados.py`, depois rode `python modelo/treinar_modelo.py`; o exportador grava a origem no JSON mobile. O classificador atual representa uma mao; sinais que dependem das duas maos ainda precisam de uma representacao e treino proprios.

Em 28/10/2026, a revisao da traducao confirmou que o artefato embarcado continua com `fonte_dados: sintetico_demo` e que `modelo/dataset_libras_real.csv` ainda nao existe. Portanto, a classificacao real esta deliberadamente desabilitada; rastrear marcos corretamente nao basta para traduzir. Para habilita-la, e necessario coletar exemplos rotulados reais para os sinais desejados (idealmente 30 ou mais por rotulo, com variacoes), treinar e exportar de novo o modelo. Nao rotular previsoes do modelo sintetico como Libras real, pois isso produziria traducoes incorretas e confianca enganosa.

Em 29/10/2026, foi baixado o pacote de landmarks do dataset publico LIBRAS-EQT-UECE, versao Zenodo 1.1. O ZIP (773,6 MB) passou na verificacao de integridade. A inspecao corrigiu a descricao anterior: os dados sao arquivos NPY, nao CSV/JSON. Cada exemplo vem em uma sequencia temporal; existe uma arvore com 318 valores por quadro e outra apenas com as duas maos (126 valores). A licenca permite uso e treinamento com citacao da fonte. Referencias: https://zenodo.org/records/20497742 e https://doi.org/10.1109/ACCESS.2026.3715086. O conjunto tem 178 classes anotadas no pacote, incluindo palavras e frases, e requer adaptacao/validacao antes de substituir o modelo embarcado.

#### Experimento inicial com dados publicos

O script `modelo/treinar_dataset_publico.py` le o ZIP sem extrai-lo e normaliza os 21 pontos da mao em relacao ao pulso, da mesma forma que pode ser calculado a partir do vetor nativo atual. Para evitar que repeticoes de um mesmo clipe vazem entre treino e avaliacao, a validacao usa GroupKFold pelos identificadores de informante disponiveis no dataset. O primeiro piloto limitou-se a rótulos de uma palavra: 1.345 clipes validos, 44 classes e 11.332 janelas temporais; 10 clipes foram descartados por terem poucos quadros validos. A media nos cinco grupos de teste foi 34,5% de acuracia e macro-F1 0,326. Os identificadores de informante sao locais a cada item e nao identificam necessariamente pessoas diferentes de forma global, portanto essa avaliacao nao comprova generalizacao para sinalizantes nunca vistos.

O resultado ainda e insuficiente para uma traducao confiavel. O Random Forest de pesquisa fica em `servidor/modelos_ia/candidato_libras_eqt_uece.joblib`; ele nao foi exportado para o app, e o modelo mobile continua inalterado e bloqueando traducoes do modelo sintetico. Esse experimento tambem usa apenas a mao dominante estimada pelo movimento; sinais que exigem as duas maos, expressao facial, cabeca ou bracos completos precisam de uma representacao temporal compativel e nova validacao. Para reproduzir: `python modelo/treinar_dataset_publico.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip --lexical-only`.

#### Experimentos temporais e modelo de teste local

Em 29/10/2026, `modelo/avaliar_modelo_temporal_publico.py` compara sequencias completas de 16 quadros com classificacao LinearSVC multiclasse. A avaliacao usa os primeiros 20 rotulos lexicais do arquivo e 622 clipes validos, com cinco particoes de informante: media de 53,9% de acuracia e macro-F1 0,505; o pior grupo ficou em 32,5% e macro-F1 0,234. A quinta particao ainda mostra generalizacao fraca. Como IDs sao locais a cada item e nao identificadores globais de pessoas, este protocolo nao garante teste em pessoas nunca vistas. `--max-classes` seleciona as primeiras N classes lexicais da ordem do ZIP; nao e um vocabulario de produto.

Correcao do experimento anterior: a tentativa inicial de acrescentar posicao do pulso reamostrava incorretamente sequencias 2D e duplicava o eixo temporal. As metricas anteriormente registradas para a variante com posicao (inclusive 53,2%) sao invalidas e foram substituidas. Com a rotina corrigida, o piloto com posicao teve 53,1% / macro-F1 0,498, abaixo do piloto atual sem posicao (53,9% / 0,505); portanto, posicao foi retirada do modelo embarcado de teste. Para comparar, reproduza:

```bash
python modelo/avaliar_modelo_temporal_publico.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip --max-classes 20
python modelo/avaliar_modelo_temporal_publico.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip --include-position --max-classes 20
```

Foi criado um modo experimental Android baseado apenas nas 21 coordenadas 3D da mao normalizadas pelo pulso. A camera envia esses 21 pontos ao React Native e a inferencia roda localmente em uma janela deslizante de 16 quadros, restrita aos 20 rotulos do piloto. O artefato `aplicativo/assets/modelo_libras_temporal_piloto.json` usa coeficientes LinearSVC exportados; `modelo/treinar_modelo_temporal_piloto.py` o recria a partir do ZIP. A tela identifica o resultado como experimental e mostra margem relativa nao calibrada, nao uma probabilidade de acerto. O modelo e ajustado novamente em todos os 622 clipes dessas classes para permitir o teste ao vivo; a acuracia de 53,9% e a estimativa held-out dos cinco grupos, nao uma medicao de acuracia garantida no celular. Esse modo e apenas para testar, nao uma traducao confiavel de Libras; use com consciencia de classificacoes erradas.

O APK experimental foi compilado e instalado no Samsung SM-S721B em 29/10/2026; o Metro carregou a interface e a tela/camera abriu. A captura de verificacao enquadrou teto e parede, sem mao para medir uma previsao ao vivo. Portanto, compilacao e carregamento foram confirmados, mas o reconhecimento com sinais feitos pela pessoa usando o aparelho ainda precisa de verificacao manual. No modo de teste, toque em `Iniciar Traducao Continua`, centralize a mao inteira no quadro e aguarde o buffer de 16 quadros; `+ Frase` acrescenta a previsao exibida ao texto.

Para evitar que variacoes entre janelas sobrepostas acrescentem varias palavras de um unico gesto, a exibicao usa consenso de pelo menos 5 das ultimas 7 predicoes. Depois de aceitar uma palavra, o app trava essa predicao ate observar pausa de movimento seguida de novo movimento; perder a mao ou pausar a traducao tambem limpa o bloqueio. Essa segmentacao e heuristica para o piloto e deve ser avaliada ao vivo com sinais consecutivos.

O script imprime as confusoes mais frequentes por particao, por exemplo `Cômodo`/`Intervenção` classificados como `Cair` no quinto grupo. Os videos RGB continuam pendentes por decisao do usuario. Para reconstruir o modelo:

```bash
python modelo/treinar_modelo_temporal_piloto.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip
```

```bash
python modelo/avaliar_modelo_temporal_publico.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip --max-classes 20
python modelo/avaliar_modelo_temporal_publico.py C:\a\LIBRAS-EQT-UECE-Landmarks.zip --include-position --max-classes 20
```

Uma tentativa com o kernel RBF demorou excessivamente no conjunto completo e foi encerrada; o script usa um classificador linear para os pilotos acima. A proxima melhoria deve primeiro investigar a particao de baixo desempenho e a estrutura completa dos landmarks antes de expandir o vocabulario.


### Orientacao da previa e alinhamento do rastreamento

A previa CameraX usa `TextureView` com superficie direta. A previa CameraX usa a superficie direta da `TextureView`, sem matriz manual de recorte, escala, rotacao ou espelhamento. A `TextureView` renderiza o buffer da previa ocupando integralmente a moldura de 300 dp, sem barras laterais. Apenas `ImageAnalysis` recebe a rotacao do display para orientar os quadros encaminhados ao MediaPipe. O overlay frontal continua espelhado no eixo X. Os pontos do MediaPipe permanecem normalizados na imagem ja orientada, e o overlay frontal espelha o eixo X para coincidir com a previa. Foi compilado e instalado um novo APK; no primeiro registro de tela a camera ainda estava iniciando, e alguns segundos depois a imagem e os pontos apareceram. A conferencia de alinhamento dos marcos deve ser feita com uma pessoa e a mao dentro do enquadramento, pois um teto ou objetos podem gerar deteccoes falsas de pose.

### Ajuste de orientacao e enquadramento da previa frontal`r`n`r`nAs matrizes manuais de rotacao/escala aplicadas ao `TextureView` causavam giro e enquadramento parcial, pois a superficie direta do CameraX ja entrega o buffer para a previa. Removemos a transformacao manual e o target rotation do `Preview`; o preview usa o buffer em tamanho nativo na area definida pela interface. O target rotation de `ImageAnalysis` foi mantido para o MediaPipe. O novo APK foi compilado, instalado e conferido no Samsung conectado: a imagem frontal preencheu toda a area de camera, sem faixas laterais.
### Alinhamento dos pontos do rastreamento

Os pontos normalizados retornados por Hand/Pose Landmarker agora sao convertidos para a orientacao do display usando `ImageProxy.imageInfo.rotationDegrees` (0/90/180/270), antes de serem enviados ao overlay React Native. A mesma conversao e aplicada aos pontos de mao e pose; o vetor de caracteristicas do classificador permanece no sistema de coordenadas de inferencia. `Preview` e `ImageAnalysis` usam a mesma rotacao de display e a previa nao recebe matriz manual, evitando rotacao dupla. O APK foi compilado e instalado no Samsung conectado; a verificacao no momento mostrou a camera preenchendo o quadro, mas sem uma mao no campo de visao para conferir a sobreposicao.

### Diagnostico de mao nao detectada

Em 28/10/2026, uma captura do Samsung mostrou a palma aberta centralizada e visivel na previa, com a traducao continua ativa, mas sem os pontos do HandLandmarker. O limiar de deteccao anterior (0,35) nao foi suficiente. Reduzimos os limiares de deteccao, presenca e rastreamento da mao para 0,20 / 0,25 / 0,30 e adicionamos uma mensagem de diagnostico na interface que diferencia ausencia de mao, falta de associacao com o pulso e ausencia da parte superior do corpo. Isso aumenta a sensibilidade, podendo tambem aumentar deteccoes falsas; a verificacao com a nova compilacao ainda e necessaria.
### Diagnostico de rastreamento sem pontos

Na verificacao do Samsung, o botao `Iniciar Traducao Continua` estava pausado; foi iniciado para habilitar o analisador local. A previa capturada depois ainda enquadrava teto/parede e somente parte do braco, sem a mao e sem a parte superior do corpo. Nessa condicao, o detector nao tem os alvos necessarios; o app aguarda mao e corpo antes de formar um vetor de traducao. Para validar, enquadre a mao aberta e os ombros/parte superior do tronco no centro da previa, com iluminacao suficiente.

A correção da ordem dos canais foi compilada no APK Android e instalada no Samsung SM-S721B em 29/10/2026. A interface e a câmera frontal abriram. A captura posterior à instalação não continha uma mão no quadro e a tradução contínua estava pausada; por isso, a eficácia do rastreamento com uma mão real permanece pendente de validação ao vivo. Para validar, toque em `Iniciar Traducao Continua`, enquadre a mao inteira com iluminacao uniforme e confira se os 21 pontos acompanham os dedos.

Em 29/10/2026, o APK `aplicativo/android/app/build/outputs/apk/debug/app-debug.apk` foi colocado na pasta original do projeto. Ele foi gerado a partir da cópia de compilação `C:\a`, depois de sincronizar e conferir o mesmo código Kotlin corrigido por SHA-256. A tentativa de executar o Gradle usando a unidade temporária apontada para a pasta original falhou durante o autolinking Expo; portanto, este arquivo e uma copia do APK compilado, nao uma compilacao executada diretamente nesse caminho. O ADB nao listou dispositivo conectado nesta etapa, entao a instalacao no celular nao foi confirmada nesta etapa.

### Consolidacao dos arquivos na pasta original

Em 29/10/2026, conferi por SHA-256 que os arquivos centrais do aplicativo na pasta original correspondem a versao compilada: `App.js`, configuracoes Expo, dependencias declaradas, modulo Kotlin da camera, modelos MediaPipe e modelo temporal. O arquivo de dados `LIBRAS-EQT-UECE-Landmarks.zip` foi copiado para a raiz original e verificado por SHA-256. O APK de depuracao tambem esta em `aplicativo/android/app/build/outputs/apk/debug/app-debug.apk`. As capturas de tela feitas durante depuracao e os caches temporarios nao sao dependencias do projeto.

O conjunto ZIP de 773,6 MB e o APK de depuracao sao artefatos locais e ficam fora do Git para evitar commits muito grandes e binarios gerados. Os modelos `.task` pequenos usados pelo modulo nativo Android sao incluidos no controle de versao; as pastas de build, caches e o keystore de depuracao sao ignorados.

### Preparacao para controle de versao

Em 30/10/2026, o projeto foi sincronizado com `https://github.com/OccultDays/Libras2Text`, branch `main`. O Android nativo do aplicativo passou a ser versionavel, mantendo fora do Git os diretorios de build, caches, keystore local, APK e o ZIP de 773,6 MB. Os modelos `.task` embarcados do modulo nativo Android foram explicitamente incluidos por serem necessarios para o rastreamento no aparelho. O README foi atualizado para refletir o estado experimental e os passos atuais para development build.
