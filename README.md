# Libras2Text

Prototipo de traducao de Libras para texto usando camera e reconhecimento de marcos corporais.

## Estado atual

- No Android, CameraX e MediaPipe rastreiam maos e parte superior do corpo no proprio aparelho.
- O aplicativo precisa de um development build Android; Expo Go nao inclui o modulo nativo local.
- O classificador temporal embarcado e experimental, cobre apenas um vocabulario piloto e ainda pode gerar previsoes incorretas. A pontuacao exibida nao e uma probabilidade calibrada.
- O processamento no dispositivo reduz a dependencia de rede, mas nao elimina os erros do modelo nem garante latencia zero.
- A documentacao completa, arquitetura, limites e procedimentos de treino estao em [DOCUMENTACAO.md](DOCUMENTACAO.md).

## Executar no Android

Com Node.js, Android Studio/SDK e um aparelho Android com depuracao USB configurados:

```bash
cd aplicativo
npm install
npx expo run:android
```

O aparelho e o computador devem estar conectados para instalar o development build. Expo Go nao e compativel com o modulo Kotlin de camera e rastreamento.

## Treino experimental

Os scripts em `modelo/` treinam e avaliam o piloto temporal usando o dataset publico LIBRAS-EQT-UECE-Landmarks. O arquivo ZIP do dataset nao e versionado neste repositorio por seu tamanho (cerca de 774 MB); obtenha-o na fonte indicada na documentacao antes de executar os scripts.
