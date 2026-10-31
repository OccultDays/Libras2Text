# Libras2Text

Projeto de tradução da Língua Brasileira de Sinais (Libras) para texto em tempo real usando a câmera do celular e rastreamento das mãos com MediaPipe.

## Visão Geral

O aplicativo captura os quadros da câmera, detecta os 21 pontos anatômicos de cada mão (exibindo o *line tracing* alinhado sobre os dedos em tempo real) e classifica a configuração de mão, o ponto de articulação, a relação entre as duas mãos e o movimento temporal para formar letras e palavras em português.

O sistema funciona automaticamente em dois modos:

1. **Modo Nativo Local no Android (*Development Build*)**: utiliza o módulo Kotlin em `aplicativo/modules/libras-native` com CameraX e MediaPipe Tasks Vision (`HandLandmarker` e `PoseLandmarker` em modo `IMAGE`), executando o rastreamento e o classificador Random Forest diretamente no aparelho, sem depender de servidor.
2. **Modo Servidor via WebSocket (Expo Go / iOS / Rede Local)**: quando aberto no Expo Go (onde módulos nativos customizados não estão compilados), o app utiliza `expo-camera` e envia os quadros ao servidor FastAPI (`servidor/principal.py`), que extrai os marcos pelo MediaPipe e devolve os pontos alinhados e o sinal reconhecido.

---

## Estrutura de Pastas

```text
Libras2Text/
├── README.md                                   # Documentação completa do projeto
├── aplicativo/                                 # Aplicativo móvel em React Native (Expo SDK 58)
│   ├── App.js                                  # Interface, line tracing SVG e inferência local
│   ├── index.js                                # Ponto de entrada do aplicativo
│   ├── app.json                                # Configurações do Expo e permissões de câmera
│   ├── package.json                            # Dependências e scripts npm
│   ├── assets/
│   │   └── modelo_libras_mobile.json           # Classificador Random Forest exportado para o celular
│   ├── android/                                # Projeto Android nativo para geração do APK
│   └── modules/libras-native/                  # Módulo nativo Android (CameraX + MediaPipe)
│       ├── expo-module.config.json             # Registro do módulo nativo no Expo
│       ├── package.json                        # Metadados do módulo local
│       ├── src/index.js                        # Carregamento seguro da view nativa (com fallback no Expo Go)
│       └── android/
│           ├── build.gradle                    # Dependências CameraX e MediaPipe Tasks Vision
│           └── src/main/
│               ├── assets/
│               │   ├── marcador_maos.task      # Modelo MediaPipe HandLandmarker
│               │   └── marcador_pose.task      # Modelo MediaPipe PoseLandmarker
│               └── java/expo/modules/librasnative/
│                   ├── LibrasNativeModule.kt       # Definição das propriedades e eventos da view
│                   └── LibrasNativeCameraView.kt   # Captura CameraX, rotação e rastreamento MediaPipe
├── modelo/                                     # Scripts de dados e treinamento do classificador
│   ├── gerar_dataset_inicial.py                # Gerador calibrado com moldes do PDF e do INES
│   ├── dataset_libras.csv                      # Base gerada com 110 características por amostra
│   ├── coletar_dados.py                        # Coleta de exemplos reais pela webcam com line tracing
│   ├── treinar_modelo.py                       # Treinamento e validação do Random Forest
│   └── exportar_modelo_mobile.py               # Conversão das árvores do modelo para JSON mobile
└── servidor/                                   # Backend Python com FastAPI
    ├── principal.py                            # Endpoints WebSocket (/ws) e HTTP (/traduzir-frame)
    ├── extrator_pontos.py                      # Extração MediaPipe e cálculo das 110 características
    ├── processador_imagem.py                   # Decodificação de quadros JPEG em bytes ou Base64
    ├── classificador_libras.py                 # Carregamento do modelo joblib e inferência rápida
    ├── gerenciador_conexoes.py                 # Controle das conexões WebSocket ativas
    ├── requisitos.txt                          # Dependências Python do projeto
    └── modelos_ia/
        ├── modelo_libras.joblib                # Classificador Random Forest serializado
        └── rotulos.joblib                      # Lista de sinais reconhecidos
```

---

## Fontes Lexicográficas e Vocabulário de Testes (23 Sinais)

Os moldes de mão, pontos de articulação e trajetórias dos 23 sinais do vocabulário de testes foram extraídos e calibrados a partir de duas fontes brasileiras oficiais de Libras:

1. **Dicionário da Língua Brasileira de Sinais — INES** (`https://dicionario.ines.gov.br/`): vídeos oficiais dos sinais e soletrações manuais gravados por intérprete do Instituto Nacional de Educação de Surdos.
2. **Dicionário de Libras Ilustrado** (*Secretaria da Administração e dos Recursos Humanos — SAT / FADERS / Governo do Estado do Rio Grande do Sul*): pranchas fotográficas do alfabeto manual (p. 5) e dos sinais *Ajudar-me / Ajudar-te* (p. 10), *Bo@ / Bom dia* (p. 17–18), *Casa* (p. 20), *Dia* (p. 28), *F* (p. 38), *Licença* (p. 58), *Não* (p. 67), *Obrigad@* (p. 70), *Sim* (p. 90) e *Você* (p. 104).

### Alfabeto Manual (13 letras)
- **`A`**: mão fechada com o polegar estendido verticalmente encostado na lateral do indicador.
- **`B`**: quatro dedos unidos apontando para cima e polegar dobrado sobre a palma.
- **`C`**: dedos unidos e curvados formando um arco em `C` com o polegar.
- **`D`**: indicador erguido e ponta do polegar tocando o dedo médio curvado.
- **`E`**: pontas dos quatro dedos flexionadas na parte superior e polegar recolhido na palma.
- **`F`**: médio, anelar e mínimo erguidos, indicador dobrado para frente e polegar encostado verticalmente pelo lado externo do indicador.
- **`I`**: apenas o dedo mínimo estendido para cima, demais dedos fechados.
- **`L`**: indicador apontando para cima e polegar aberto lateralmente em ângulo reto.
- **`O`**: pontas dos dedos curvadas tocando a ponta do polegar em formato circular.
- **`U`**: indicador e médio estendidos juntos para cima, demais dedos fechados.
- **`V`**: indicador e médio estendidos e afastados em `V`.
- **`W`**: indicador, médio e anelar estendidos e separados, mínimo e polegar recolhidos.
- **`Y`**: polegar e dedo mínimo estendidos com indicador, médio e anelar fechados (parado ou com leve inclinação vertical).

### Palavras e Expressões Cotidianas (10 sinais)
- **`OI`**: suporta tanto a execução clássica do INES (mão iniciando em `O` e abrindo o mínimo em `O+I` / `I` com giro na altura do ombro/pescoço) quanto o aceno lateral de *Olá* com os 5 dedos abertos.
- **`BOM DIA`**: sinal composto iniciando com os dedos unidos em `O` diante da boca abrindo para palma aberta (`BOM`) e seguindo para a lateral com configuração `D`/`DIA`.
- **`OBRIGADO`**: mão plana em `B` tocando a região da testa e afastando-se para frente/baixo.
- **`POR FAVOR`**: execução bimanual na altura do peito com a mão dominante plana em `B` na horizontal deslizando sobre/diante da mão de base em `B` (também aceita execução com uma mão quando a outra segura o celular).
- **`SIM`**: mão fechada em configuração `S` (polegar dobrado sobre os dedos fechados, sem levantar o polegar) flexionando o punho para frente e para baixo na altura do peito/ombro.
- **`NAO`**: contempla as duas variantes registradas no PDF e no INES — indicador erguido oscilando lateralmente de um lado para o outro (`NÃO 1`) ou mão em `Y` balançando horizontalmente (`NÃO 2`).
- **`EU`**: indicador voltado horizontalmente apontando/tocando o centro do peito.
- **`VOCE`**: indicador apontando diretamente para frente (em direção ao interlocutor/câmera) na altura do peito/pescoço.
- **`CASA`**: duas mãos planas em `B` inclinadas simetricamente (~45°) tocando as pontas dos dedos na frente do peito formando o telhado `/\` (com suporte alternativo unilateral).
- **`AJUDA`**: execução bimanual no peito em que uma mão serve de apoio por baixo/trás da outra mão (em `B` vertical ou em `A`, conforme o PDF e os vídeos `AJUDAR 1` e `AJUDAR 2` do INES) impulsionando-a para frente, além de suporte unilateral.

---

## Como Funciona o Reconhecimento e o Line Tracing

### 1. Alinhamento e Line Tracing das Mãos (`RastreamentoMao`)
- Tanto o módulo nativo Android quanto o servidor Python enviam as dimensões reais do quadro processado (`largura_quadro` e `altura_quadro`) junto com os 21 pontos normalizados de cada mão detectada.
- A moldura da câmera no aplicativo usa proporção `3:4` (compatível com os quadros retrato `480x640` da câmera) e mede o tamanho exato renderizado na tela com `onLayout`.
- O componente `RastreamentoMao` aplica a projeção proporcional de preenchimento (`cover` / `centerCrop`) e espelhamento na câmera frontal, mantendo as linhas dos dedos e da palma alinhadas sobre as duas mãos.

### 2. Vetor de 110 Características (`calcular_vetor_quadro` e `montar_vetor_temporal`)
O extrator no Python (`servidor/extrator_pontos.py`) e no React Native (`aplicativo/App.js`) calcula exatamente os mesmos 110 atributos numéricos:
- **63 coordenadas centradas no pulso, canônicas para esquerda/direita e escaladas pelo tamanho da palma**: corrige a proporção do quadro, subtrai o pulso, divide pelo comprimento médio dos metacarpos (`0 -> 5, 9, 13, 17`) e espelha o eixo horizontal pelo determinante da palma (`5` e `17`), fazendo com que a mão esquerda e a mão direita gerem a mesma geometria.
- **25 medidas geométricas invariantes**: distâncias das pontas ao pulso e às bases, cossenos de curvatura das articulações de cada dedo, distâncias de pinça entre o polegar e os demais dedos e abertura lateral entre dedos vizinhos.
- **4 atributos de ponto de articulação**: altura do pulso e do indicador e distância horizontal ao eixo do nariz/ombros quando a pose está visível, com estimativa automática caso apenas a mão apareça na imagem.
- **6 atributos de interação bimanual**: presença da segunda mão (`tem_outra_mao`), distância normalizada entre pulsos (`dist_pulsos`), distância entre pontas dos indicadores e médios (`dist_pontas_ind`, `dist_pontas_med`), diferença relativa de altura entre pulsos (`dif_altura_pulsos`) e cosseno do ângulo entre as duas palmas (`cos_palmas`), preservando o contexto bimanual mesmo quando uma mão cruza brevemente na frente da outra.
- **12 atributos temporais de movimento** (janela deslizante de até `1,1s` / 9 quadros espaçados em pelo menos `110 ms`): mede deslocamento líquido `(|dx|, dy)`, amplitude máxima pico a pico `(amp_x, amp_y)` e energia acumulada (`energia_x`, `energia_y`, `energia_total`) avaliando 4 pontos de referência da mão (pulso `0`, ponta do indicador `8`, ponta do mínimo `20` e base do médio `9`), além da variação de abertura dos 5 dedos (`delta_dedos`).

### 3. Desempenho e Rastreamento Dinâmico
- **Detecção em quadro completo (`RunningMode.IMAGE`)**: evita perder o recorte da mão quando ela se move rápido em taxas moderadas de captura.
- **Trilhas independentes para até duas mãos (`num_hands=2`)**: cada mão visível mantém seu próprio histórico temporal, reforçando a confiança quando ambas concordam em sinais bimanuais e priorizando a mão ativa em gestos unilaterais.
- **Inferência leve em memória**: `ClassificadorLibras` pré-carrega as 75 árvores do Random Forest em listas Python (`n_jobs=1`), classificando cada mão em menos de `0,2 ms`.

---

## Como Instalar e Executar

### Pré-requisitos
- **Python 3.10+**
- **Node.js 18+** e `npm`
- Para uso no **Expo Go**: celular e computador na mesma rede Wi-Fi
- Para uso **Nativo Local no Android** (sem servidor): JDK 17, Android SDK / Android Studio e celular Android com Depuração USB ativa

### 1. Instalar dependências do Python
Na raiz do projeto:

```bash
pip install -r servidor/requisitos.txt
```

### 2. Coletar exemplos reais e treinar o modelo (opcional)
Os arquivos treinados já estão prontos em `servidor/modelos_ia/` e `aplicativo/assets/modelo_libras_mobile.json`. Para gravar amostras próprias pela webcam (com visualização do *line tracing* das mãos):

```bash
python modelo/coletar_dados.py
```

Em seguida, treine e atualize os modelos do servidor e do aplicativo:

```bash
python modelo/treinar_modelo.py
```

### 3. Executar pelo Expo Go + Servidor Python
1. Inicie o servidor FastAPI na porta `8000`:
   ```bash
   python servidor/principal.py
   ```
2. Descubra o IP local do computador (`ipconfig` no Windows) e permita a porta `8000` no firewall se necessário.
3. Em outro terminal, inicie o aplicativo:
   ```bash
   cd aplicativo
   npm install
   npx expo start
   ```
4. Abra no **Expo Go**, informe o IP do servidor no topo da tela, toque em **Conectar** e depois em **Iniciar Traducao Continua**.

### 4. Executar com Processamento Local Nativo no Android (*Development Build*)
Com o celular Android conectado via USB (`adb devices`):

```bash
cd aplicativo
npm install
npm run android
```

Após instalar o APK de desenvolvimento pela primeira vez, nas execuções seguintes basta manter o Metro ativo:

```bash
adb reverse tcp:8081 tcp:8081
cd aplicativo
npm run start:dev-client
```

---

## Dicas para Melhor Reconhecimento

- **Iluminação**: evite janelas ou lâmpadas fortes diretamente atrás de você (contra-luz), pois isso escurece os dedos na câmera frontal.
- **Enquadramento**: mantenha as mãos inteiras dentro da moldura da câmera e, para sinais ancorados no rosto ou peito (`OBRIGADO`, `BOM DIA`, `POR FAVOR`, `EU`, `CASA`, `AJUDA`), enquadre da cabeça até o peito.
- **Coleta complementar**: gravar amostras reais adicionais com `python modelo/coletar_dados.py` permite adaptar ainda mais o modelo ao seu ambiente e câmera.
