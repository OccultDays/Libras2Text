# Libras2Text

Aplicativo mobile em React Native para traducao em tempo real da Lingua Brasileira de Sinais (Libras) para texto, com rastreamento das maos via MediaPipe e classificacao por Random Forest diretamente no celular (100% offline).

---

## 1. Visao Geral

O Libras2Text executa todo o processamento no proprio aparelho Android:
1. **Captura e rastreamento nativo**: a camera utiliza CameraX em resolucao otimizada (240x320 com rotacao em C++) integrada ao MediaPipe Tasks Vision (`marcador_maos.task` em `RunningMode.VIDEO` com aceleracao GPU e `marcador_pose.task` em thread separada), detectando 21 pontos anatomicos de cada mao em tempo real.
2. **Line tracing**: uma `View` nativa Android (`Canvas` acelerado por hardware) desenha os pontos e as ligacoes dos dedos diretamente sobre a imagem da camera sem passar pela ponte JavaScript, com escala e espelhamento ajustados conforme a camera frontal ou traseira.
3. **Extracao de caracteristicas**: calcula um vetor de 110 caracteristicas (coordenadas relativas ao pulso, metricas geometricas, pontos de articulacao, interacao bimanual e derivadas temporais).
4. **Classificacao local**: as 75 arvores do modelo Random Forest estao embarcadas em JSON (`modelo_libras_mobile.json`) e sao percorridas pelo JavaScript em menos de 1 ms, sem necessidade de internet ou servidor externo.

---

## 2. Estrutura do Projeto

```text
Libras2Text/
├── Libras2Text.apk                             # APK de producao pronto para instalar
├── README.md                                   # Documentacao do projeto
├── aplicativo/                                 # Codigo do app React Native (Expo SDK 58)
│   ├── App.js                                  # Interface, extracao de caracteristicas e inferencia local
│   ├── index.js                                # Ponto de entrada
│   ├── app.json                                # Configuracoes e permissoes do aplicativo
│   ├── package.json                            # Dependencias do projeto
│   ├── assets/
│   │   └── modelo_libras_mobile.json           # Modelo Random Forest exportado para o app
│   ├── modules/libras-native/                  # Modulo nativo Android (CameraX + MediaPipe)
│   │   ├── expo-module.config.json             # Registro do modulo no Expo
│   │   ├── src/index.js                        # Exportacao do componente de camera nativa
│   │   └── android/
│   │       ├── build.gradle                    # Dependencias CameraX e MediaPipe
│   │       └── src/main/
│   │           ├── assets/
│   │           │   ├── marcador_maos.task      # Modelo MediaPipe HandLandmarker
│   │           │   └── marcador_pose.task      # Modelo MediaPipe PoseLandmarker
│   │           └── java/expo/modules/librasnative/
│   │               ├── LibrasNativeModule.kt       # Ponte de eventos e propriedades da view
│   │               └── LibrasNativeCameraView.kt   # Captura CameraX e deteccao MediaPipe
│   └── android/                                # Projeto nativo Android para build do APK
├── modelo/                                     # Scripts de treino e dataset
│   ├── dataset_libras.csv                      # Base de dados com 110 caracteristicas
│   ├── gerar_dataset_inicial.py                # Gerador com moldes baseados no INES e literatura
│   ├── coletar_dados.py                        # Coleta de amostras pela webcam
│   ├── treinar_modelo.py                       # Treinamento do classificador Random Forest
│   └── exportar_modelo_mobile.py               # Conversor do modelo scikit-learn para JSON
└── servidor/                                   # Servidor opcional Python (FastAPI + WebSocket)
    ├── principal.py                            # Backend para testes em computador
    ├── extrator_pontos.py                      # Extracao de pontos no servidor
    ├── processador_imagem.py                   # Processamento de imagem
    ├── classificador_libras.py                 # Classificador joblib
    └── requisitos.txt                          # Dependencias do servidor
```

---

## 3. Vocabulario Suportado (23 Sinais)

O modelo foi treinado com base nas referencias oficiais do INES e no Dicionario de Libras Ilustrado (SAT/FADERS):

### Alfabeto Manual (13 letras)
- **A**: mao fechada com polegar estendido ao lado do indicador.
- **B**: quatro dedos estendidos para cima e polegar dobrado na palma.
- **C**: dedos curvados em formato de arco de C.
- **D**: indicador estendido e ponta do polegar tocando o medio curvado.
- **E**: dedos flexionados na parte superior com polegar recolhido.
- **F**: indicador dobrado para frente com polegar apoiado na lateral externa.
- **I**: dedo minimo estendido para cima e os outros fechados.
- **L**: indicador para cima e polegar aberto formando angulo reto.
- **O**: pontas dos dedos unidas com a ponta do polegar em circulo.
- **U**: indicador e medio estendidos juntos para cima.
- **V**: indicador e medio estendidos e afastados em formato de V.
- **W**: indicador, medio e anelar estendidos e separados.
- **Y**: polegar e minimo estendidos, demais dedos fechados.

### Palavras e Expressoes (10 sinais)
- **OI**: mao partindo de configuracao O e abrindo o minimo em I com movimento.
- **BOM DIA**: mao tocando proximo a boca e abrindo em palma, seguida de D/DIA.
- **OBRIGADO**: mao em B tocando a testa e descendo para frente.
- **POR FAVOR**: maos em B na altura do peito, deslizando na frente uma da outra.
- **SIM**: mao em S flexionando o punho verticalmente para baixo.
- **NAO**: indicador oscilando lateralmente ou mao em Y balancando.
- **EU**: indicador apontando para o centro do peito.
- **VOCE**: indicador apontando para frente na direcao da camera.
- **CASA**: duas maos em B inclinadas tocando as pontas dos dedos em telhado.
- **AJUDA**: mao dominante em B impulsionada para frente com apoio da outra mao.

---

## 4. Vetor de 110 Caracteristicas

Para garantir a mesma representacao numerica no app e no treino:
1. **63 coordenadas centradas no pulso**: coordenadas dos 21 marcos da mao normalizadas pelo comprimento medio dos metacarpos e espelhadas horizontalmente pelo determinante da palma, tornando a extracao equivalente para mao esquerda e direita.
2. **25 medidas geometricas**: distancias das pontas dos dedos ao pulso, curvatura das articulacoes por produto escalar, distancias entre dedos adjacentes e pinças com o polegar.
3. **4 medidas de articulacao corporal**: altura do pulso e indicador em relacao ao nariz e largura dos ombros (via PoseLandmarker), com valores de fallback caso o tronco nao esteja visivel.
4. **6 metricas bimanuais**: indicador de presenca da segunda mao, distancia entre pulsos, distancias entre pontas dos indicadores e medios, diferenca de altura e cosseno entre as palmas.
5. **12 metricas de movimento temporal**: janela de ate 1,1 s avaliando deslocamento liquido, amplitude maxima, energias acumuladas dos marcos de referencia e variacao de abertura dos dedos.

---

## 5. Como Instalar e Testar o APK

O arquivo compilado pronto para uso esta na raiz:
- `Libras2Text.apk`

### Instalacao no celular
1. Conecte o celular Android ao computador via cabo USB ou transfira o APK por Bluetooth / Google Drive.
2. Caso utilize o cabo e tenha o `adb` instalado:
   ```bash
   adb install -r Libras2Text.apk
   ```
3. No celular, abra o aplicativo e conceda a permissao de uso da camera quando solicitada.
4. Aponte a camera frontal ou traseira para a sua mao. As linhas do rastreamento aparecerao sobrepostas aos dedos e o sinal sera traduzido na tela com a confianca e o tempo de inferencia em milissegundos.

---

## 6. Como Compilar um Novo APK

Caso faca alteracoes no codigo do aplicativo ou no modelo:

### Pre-requisitos
- **Node.js 18+** e `npm`
- **JDK 17** instalado
- **Android SDK** configurado (variavel `ANDROID_HOME` ou arquivo `aplicativo/android/local.properties` com `sdk.dir`)

### Passos de compilacao
1. Instalar as dependencias do JavaScript:
   ```bash
   cd aplicativo
   npm install
   ```

2. Compilar o APK de release usando o Gradle:
   ```bash
   cd android
   .\gradlew assembleRelease
   ```

3. O APK gerado ficara em:
   `aplicativo/android/app/build/outputs/apk/release/app-release.apk`

4. Para copiar para a raiz do repositorio (estando em `aplicativo/android`):
   ```bash
   copy app\build\outputs\apk\release\app-release.apk ..\..\Libras2Text.apk
   ```

---

## 7. Funcionamento da Interface

- **Cabecalho**: exibe o status de operacao ("Modo: Local Offline"), a contagem do buffer temporal (ate 9 quadros), o tempo total de inferencia em ms e a camera ativa.
- **Area de Video**: visualizacao da camera em formato 3:4 com o line tracing em `Canvas` nativo desenhado diretamente sobre as maos.
- **Painel de Traducao**: mostra o sinal reconhecido em destaque, o percentual de confianca e o texto acumulado.
- **Botoes de Acao**:
  - `Gravar Gesto Longo (2.5s)`: contagem regressiva de 3 s e captura focada de 2,5 s para sinais com movimento.
  - `Iniciar / Pausar Traducao Continua`: liga ou pausa a classificacao frame a frame.
  - `+ Frase`: insere manualmente o sinal atual no texto traduzido.
  - `Espaco`: adiciona espaco entre palavras.
  - `Apagar`: apaga o ultimo termo inserido.
  - `Limpar`: limpa todo o texto acumulado.
  - `Trocar Camera`: alterna entre camera frontal e traseira.
