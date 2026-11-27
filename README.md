# Libras2Text

Aplicativo mobile em React Native para traducao em tempo real da Lingua Brasileira de Sinais (Libras) para texto, com rastreamento nativo de maos e corpo via MediaPipe e classificacao por Random Forest diretamente no celular (100% offline).

---

## 1. Visao Geral

O Libras2Text executa todo o processamento no proprio aparelho Android:
1. **Captura e rastreamento nativo**: a camera utiliza CameraX (`360x480` em proporcao `4:3` alinhada entre previa e analise, com rotacao nativa) integrada ao MediaPipe Tasks Vision (`marcador_maos.task` em `RunningMode.VIDEO` com aceleracao GPU, limiares de presenca/rastreamento em `0.50`, timestamp real do quadro e reinicio automatico em pausas longas, alem de `marcador_pose.task` em thread separada a cada 250 ms), detectando 21 pontos anatomicos de cada mao e 13 pontos superiores do corpo.
2. **Line tracing nativo**: a `View` nativa Android (`CamadaRastreamentoCorpo`, usando `Canvas` acelerado por hardware com arrays pre-alocados) desenha em tempo real os pontos e ligacoes das maos em verde sobre a camera, respeitando escala e espelhamento da camera frontal ou traseira. O desenho dos 13 pontos superiores do corpo (rosto, torax superior e bracos) tambem esta implementado na view, mas desativado por padrao (`desenharCorpo = false`), enquanto o `PoseLandmarker` segue alimentando as metricas corporais do classificador.
3. **Extracao de caracteristicas e filtro de repouso**: calcula o vetor de 110 caracteristicas no Vocabulario Padrao e estende para 132 caracteristicas no Vocabulario Expandido (adicionando formato dos 5 dedos da mao de apoio, toque polegar-palma e indicador-pulso, orientacao da palma, separacao lateral e giro de punho), filtrando maos paradas na zona inferior de repouso e preservando o contexto bimanual durante oclusoes curtas.
4. **Classificacao local**: os classificadores Random Forest estao embarcados em JSON (`modelo_libras_mobile.json` com 80 arvores no vocabulario padrao e `modelo_apresentacao_mobile.json` com 100 arvores e folhas esparsas no vocabulario expandido, alternaveis no indicador de status do cabecalho) e sao percorridos pelo JavaScript em cerca de 1 ms, sem internet ou servidor externo.

---

## 2. Estrutura do Projeto

```text
Libras2Text/
├── Libras2Text.apk                             # APK de producao pronto para instalar
├── README.md                                   # Documentacao do projeto
├── aplicativo/                                 # Codigo do app React Native (Expo SDK 58)
│   ├── App.js                                  # Interface, extracao de 110 caracteristicas e inferencia local
│   ├── index.js                                # Ponto de entrada
│   ├── app.json                                # Configuracoes e permissoes do aplicativo
│   ├── package.json                            # Dependencias do projeto
│   ├── assets/
│   │   ├── modelo_libras_mobile.json           # Modelo Random Forest padrao exportado para o app
│   │   └── modelo_apresentacao_mobile.json     # Modelo Random Forest do vocabulario expandido
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
│   │               └── LibrasNativeCameraView.kt   # Captura CameraX, MediaPipe e line tracing nativo
│   └── android/                                # Projeto nativo Android para build do APK
├── modelo/                                     # Scripts de treino, dataset e validacao real
│   ├── dataset_libras.csv                      # Base anatomica sintetica calibrada (110 caracteristicas)
│   ├── dataset_videos_reais.csv                # Janelas reais extraidas pelo MediaPipe de videos publicos
│   ├── dataset_apresentacao_reais.csv          # Janelas reais curadas do vocabulario expandido
│   ├── gerar_dataset_inicial.py                # Gerador com moldes e faixas calibradas pelo MediaPipe
│   ├── coletar_dados.py                        # Coleta de amostras pela webcam
│   ├── validar_videos_reais.py                 # Validacao com videos reais (INES, V-LIBRASIL, UFV, SpreadTheSign) e LOVO
│   ├── treinar_modelo.py                       # Treinamento hibrido do Random Forest com pesos balanceados
│   ├── treinar_apresentacao.py                 # Curadoria, validacao LOVO e treino do vocabulario expandido
│   └── exportar_modelo_mobile.py               # Conversor do modelo scikit-learn para JSON
└── servidor/                                   # Servidor opcional Python (FastAPI + WebSocket)
    ├── principal.py                            # Backend para testes em computador
    ├── extrator_pontos.py                      # Extracao de pontos e filtro de mao em repouso
    ├── processador_imagem.py                   # Processamento de imagem
    ├── classificador_libras.py                 # Classificador joblib
    └── requisitos.txt                          # Dependencias do servidor
```

---

## 3. Vocabulario Suportado (24 Sinais Exibidos / 26 Classes Internas no Vocabulario Padrao)

O modelo foi treinado com base nas referencias oficiais do INES, Dicionario de Libras Ilustrado (SAT/FADERS) e extracoes reais do MediaPipe de videos publicos brasileiros (INES, V-LIBRASIL/UFPE, UFV e SpreadTheSign):

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

### Palavras e Expressoes — Vocabulario Padrao (11 sinais)
- **OI**: mao partindo de configuracao O e abrindo o minimo em I com movimento (ou aceno em palma aberta).
- **BOM**: mao fechada tocando proximo a boca e abrindo em palma para frente.
- **DIA**: reconhecido a partir de duas sub-variacoes internas (`DIA_ROSTO`, com mao em D saindo do rosto/tempora, e `DIA_SOL`, variacao bimanual em arco sobre o braco horizontal). Ao sinalizar `BOM` e `DIA` em sequencia, os dois termos entram separados formando `BOM DIA` no texto.
- **OBRIGADO**: mao plana tocando a testa e avancando para frente (uma ou duas maos).
- **POR FAVOR**: maos planas na altura do peito deslizando ou friccionando juntas.
- **SIM**: mao em S flexionando o punho verticalmente.
- **NAO**: indicador oscilando lateralmente ou mao em Y balancando.
- **EU**: indicador apontando para o centro do peito.
- **VOCE**: indicador apontando para frente na direcao da camera.
- **CASA**: duas maos inclinadas tocando as pontas dos dedos em formato de telhado.
- **AJUDA**: mao dominante impulsionada com apoio da outra mao na altura do peito.
- **NEUTRO** *(classe interna suprimida)*: treinada com quadros reais de repouso e transicoes para evitar falsos positivos entre gestos.

O **Vocabulario Expandido** adiciona mais sinais para frases completas em tempo real (detalhados na Secao 8).

---

## 4. Vetor de Caracteristicas (110 Padrao / 132 Expandido) e Pipeline

Para garantir a mesma representacao numerica no app e no treino:
1. **63 coordenadas centradas no pulso**: coordenadas dos 21 marcos da mao normalizadas pelo comprimento medio dos metacarpos e espelhadas horizontalmente pelo determinante da palma, tornando a extracao equivalente para mao esquerda e direita.
2. **25 medidas geometricas**: distancias das pontas dos dedos ao pulso, curvatura das articulacoes por produto escalar, distancias entre dedos adjacentes e pincas com o polegar.
3. **4 medidas de articulacao corporal**: altura do pulso e indicador em relacao ao nariz e largura dos ombros (via `PoseLandmarker`), com fallback caso o tronco nao esteja visivel.
4. **6 metricas bimanuais**: presenca da segunda mao ativa, distancia entre pulsos, distancias entre pontas dos indicadores e medios, diferenca de altura e cosseno entre as palmas.
5. **12 metricas de movimento temporal**: janela de ate 1,1 s avaliando deslocamento liquido, amplitude maxima, energias acumuladas e variacao de abertura dos dedos.
6. **22 atributos extras do Vocabulario Expandido (`montarVetorTemporalExpandido`, indices 110..131)**:
   - *14 atributos estaticos/bimanuais*: orientacao simetrica da palma, magnitude do determinante da palma, distancia polegar-base do indicador, distancias polegar/indicador ate palma e pulso da outra mao, separacao horizontal/vertical entre pulsos, diferenca de curvatura entre as maos e curvatura dos 5 dedos da mao de apoio.
   - *8 atributos temporais*: amplitude e inversao de sinal do determinante da palma (giro de punho), energia e amplitude do polegar relativo a palma, amplitude e cruzamento horizontal entre pulsos, oscilacao do indicador no pulso de apoio e menor distancia polegar-palma na janela.

Melhorias aplicadas no pipeline (espelhadas em `App.js` e `extrator_pontos.py`):
- **Filtro de mao em repouso e classe `NEUTRO`**: ignora maos paradas na regiao inferior de repouso e suprime quadros classificados como `NEUTRO` (repouso/transicao).
- **Memoria de oclusao bimanual**: preserva as metricas da segunda mao (inclusive os atributos da mao de apoio no vetor de 132 caracteristicas) dentro da janela de 1,1 s quando as maos estavam encostadas antes de uma cobrir a outra.
- **Agrupamento de sub-variacoes**: soma probabilidades das sub-classes internas (`DIA_ROSTO` + `DIA_SOL` -> `DIA` e `2_HORIZ` + `2_VERT` -> `2`), mantendo cada palavra independente no texto.
- **Acumulador de gesto longo (2,5 s) com maturidade do buffer**: soma a confianca ponderada pela maturidade da trilha (`min(1,0, quadros / 4,0)`) durante toda a gravacao para evitar que os primeiros quadros de entrada da mao ou transicoes curtas superem o sinal completo.

---

## 5. Validacao Real com MediaPipe (Sem Vazamento)

Para evitar falsos positivos de validacao sintetica, o projeto conta com o script `modelo/validar_videos_reais.py`, que testa o pipeline completo usando videos reais passando pelo MediaPipe (inclusive simulando o recorte `240x320` do celular), alem da validacao em `360x480` no `modelo/treinar_apresentacao.py`:

```bash
# Baixar videos reais de referencia (armazenados em modelo/videos_reais/, ignorados pelo git)
python modelo/validar_videos_reais.py --baixar

# Extrair janelas reais para treino hibrido (gera modelo/dataset_videos_reais.csv)
python modelo/validar_videos_reais.py --extrair-treino

# Rodar validacao Leave-One-Video-Out (o video avaliado fica 100% fora do treino)
python modelo/validar_videos_reais.py --validar-lovo --recorte-celular
```

Voce tambem pode adicionar videos gravados no seu celular em `modelo/videos_reais/<SINAL>/` (por exemplo `modelo/videos_reais/BOM DIA/teste1.mp4`) e reexecutar o script.

---

## 6. Como Instalar e Testar o APK

O arquivo compilado pronto para uso esta na raiz:
- `Libras2Text.apk`

### Instalacao no celular
1. Conecte o celular Android ao computador via cabo USB ou transfira o APK por Bluetooth / Google Drive.
2. Caso utilize o cabo e tenha o `adb` instalado:
   ```bash
   adb install -r Libras2Text.apk
   ```
3. No celular, abra o aplicativo e conceda a permissao de uso da camera quando solicitada.
4. Enquadre rosto, ombros e maos. O line tracing desenhara as maos em verde sobre a camera, exibindo o sinal traduzido, a confianca e o tempo de inferencia em milissegundos.

---

## 7. Como Compilar um Novo APK

Caso faca alteracoes no codigo do aplicativo ou no modelo:

### Pre-requisitos
- **Node.js 18+** e `npm`
- **JDK 17+** instalado
- **Android SDK** configurado (variavel `ANDROID_HOME` ou arquivo `aplicativo/android/local.properties` com `sdk.dir`)

### Passos de compilacao
1. (Opcional) Retreinar e exportar os modelos locais:
   ```bash
   python modelo/treinar_modelo.py
   python modelo/treinar_apresentacao.py --treinar
   ```

2. Instalar as dependencias do JavaScript:
   ```bash
   cd aplicativo
   npm install
   ```

3. Compilar o APK de release usando o Gradle:
   ```bash
   cd android
   .\gradlew assembleRelease
   ```

4. O APK gerado ficara em:
   `aplicativo/android/app/build/outputs/apk/release/app-release.apk`

5. Para copiar para a raiz do repositorio (estando em `aplicativo/android`):
   ```bash
   copy app\build\outputs\apk\release\app-release.apk ..\..\Libras2Text.apk
   ```

---

## 8. Funcionamento da Interface

- **Cabecalho**: exibe o seletor discreto de vocabulario na linha de status (`Vocabulario: Padrao (Offline)` ou `Vocabulario: Expandido (Offline)`, alternado com um toque direto no texto de status, sem botoes extras), a contagem do buffer temporal (ate 9 quadros), o tempo total de inferencia em ms e a camera ativa.
- **Area de Video**: visualizacao da camera em formato 3:4 (`360x480`) com o line tracing em `Canvas` nativo desenhado sobre as maos.
- **Painel de Traducao**: mostra o sinal reconhecido em destaque, o percentual de confianca e o texto acumulado.
- **Botoes de Acao**:
  - `Gravar Gesto Longo (2.5s)`: contagem regressiva de 3 s e captura focada de 2,5 s com pontuacao acumulada para sinais com movimento.
  - `Iniciar / Pausar Traducao Continua`: liga ou pausa a classificacao frame a frame.
  - `+ Frase`: insere manualmente o sinal atual no texto traduzido.
  - `Espaco`: adiciona espaco entre palavras.
  - `Apagar`: apaga o ultimo termo inserido.
  - `Limpar`: limpa todo o texto acumulado.
  - `Trocar Camera`: alterna entre camera frontal e traseira.

### Vocabulario Expandido (Apresentacao em Tempo Real)

Alem do vocabulario padrao (`aplicativo/assets/modelo_libras_mobile.json`), o aplicativo conta com o **Vocabulario Expandido** (`aplicativo/assets/modelo_apresentacao_mobile.json`, 3,88 MB com probabilidades gravadas apenas nas folhas, 100 arvores, 132 atributos, 12 sinais exibidos / 15 classes internas), alternavel ao tocar em `Vocabulario: Padrao (Offline)` no cabecalho e treinado especificamente com janelas reais curadas visualmente (7.454 amostras em `modelo/dataset_apresentacao_reais.csv`) das bases INES, V-LIBRASIL/UFPE, Glossario Letras-Libras UFSC, SignBank UFSC, SpreadTheSign, MALTA-LIBRAS e clipes educacionais do YouTube:

- **Sinais independentes em tempo real**: `BOM` · `DIA` · `LIBRAS` · `2` · `TEXTO` · `APLICATIVO` · `TRADUZIR` · `TEMPO` · `REAL` · `TRANSFORMAR` · `INOVACAO` (sinal oficial `NOVIDADE`) · `ACESSIBILIDADE` (`NEUTRO` suprimido na exibicao).
- **Validacao real sem vazamento (`modelo/treinar_apresentacao.py --validar-lovo --recorte-celular` em `360x480`)**: **61/64 (95,3%)** no modo *Gravar Gesto (2.5s)* e **61/64 (95,3%)** na maioria das janelas (**64/64 = 100,0%** com o sinal correto em pelo menos 1 janela), destacando **100%** (gesto e maioria) em `BOM` (4/4), `DIA_ROSTO` (5/5), `2_HORIZ` (4/4), `2_VERT` (3/3), `ACESSIBILIDADE` (3/3), `APLICATIVO` (2/2), `INOVACAO` (3/3), `TEMPO` (8/8), `TRADUZIR` (7/7) e `TRANSFORMAR` (4/4), alem de `TEXTO` (4/4 gesto / 3/4 maioria), `DIA_SOL` (2/2 maioria / 1/2 gesto), `REAL` (7/8) e `LIBRAS` (6/7).

### Estabilizacao Offline em Tempo Real (Sem Vies de Roteiro)

- **Classificacao 100% visual (`preverModeloLocal`)**: todas as predicoes dependem exclusivamente da camera e dos votos das arvores da Random Forest (sem bonus contextual nem juncao automatica de palavras), preservando `NEUTRO` suprimido na exibicao.
- **Filtro de firmeza na traducao continua (`App.js`)**: exige buffer minimo (`>= 4` quadros) e `3` leituras consistentes seguidas com confianca `>= 50%` antes de inserir um sinal no texto, evitando disparos acidentais durante a movimentacao de entrada e saida da mao.
- **Anti-repeticao leve**: impede duplicata consecutiva da mesma palavra e oscilacoes rapidas dentro de `1,5 s` (`1,2 s` para a mesma letra no mesmo final de palavra), liberando sempre via `+ Frase` ou `Apagar`.

---

## 9. Melhorias Futuras

Para a etapa atual do projeto, o classificador Random Forest (`modelo_libras_mobile.json` no Vocabulario Padrao e `modelo_apresentacao_mobile.json` no Vocabulario Expandido) foi mantido por oferecer estabilidade com a quantidade atual de videos reais por sinal. Como evolucao planejada apos esta versao:

- **Migracao para TensorFlow Lite / LiteRT com rede temporal (1D-CNN ou GRU)**: substituir a classificacao por resumo de 110 atributos no JavaScript por um modelo `.tflite` executando diretamente no modulo Kotlin ([LibrasNativeCameraView.kt](aplicativo/modules/libras-native/android/src/main/java/expo/modules/librasnative/LibrasNativeCameraView.kt)), recebendo sequencias frame a frame (`T x F`) para aprender sinais longos e compostos sem decomposicao manual e reduzir o tamanho do modelo embarcado.
- **Ampliacao da base de videos reais**: antes de treinar uma rede neural temporal, expandir a coleta real por sinal (integrando mais articuladores do V-LIBRASIL, base MINDS-Libras e gravacoes proprias no celular), aplicando *data augmentation* sobre as sequencias reais de landmarks e mantendo a validacao rigorosa sem vazamento (*Leave-One-Video-Out* / *Leave-One-Signer-Out*).
- **Deteccao de orientacao da palma (frente vs. costas da mao)**: hoje o extrator zera o eixo `z` e aplica espelhamento automatico por `detPalma` para suportar destros, canhotos e ambas as cameras, tornando palma e dorso equivalentes no vetor de 110 atributos. Como evolucao conjunta ao novo modelo, exportar `handednesses()` (*Left/Right*) do MediaPipe no modulo Kotlin ([LibrasNativeCameraView.kt](aplicativo/modules/libras-native/android/src/main/java/expo/modules/librasnative/LibrasNativeCameraView.kt)) corrigido pelo espelhamento da camera frontal e cruzar com `detPalma` / vetor normal 3D para capturar giros de punho (ex.: `TRANSFORMAR`, `TRADUZIR`, `BOM`) e desambiguar pares minimos sem quebrar canhotos.
- **Compatibilidade nativa no Gradle**: alinhar a dependencia do TFLite/LiteRT com o runtime ja embutido pelo MediaPipe `tasks-vision` (configurando resolucao de bibliotecas `.so` se necessario).
