import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, TextInput, ScrollView, Platform, NativeModules } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { File } from 'expo-file-system';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Circle, Line } from 'react-native-svg';
import LibrasNativeCamera from './modules/libras-native';
import modeloLocal from './assets/modelo_libras_mobile.json';

const usarCameraNativa = Platform.OS === 'android' && LibrasNativeCamera !== null;
const JANELA_TEMPORAL_MS = 1100;

function obterIpPadraoServidor() {
  const urlScript = NativeModules?.SourceCode?.scriptURL || '';
  const resultado = urlScript.match(/^https?:\/\/(\d+\.\d+\.\d+\.\d+)/);
  if (resultado && resultado[1] && resultado[1] !== '127.0.0.1') {
    return resultado[1];
  }
  return '192.168.0.16';
}

const CONEXOES_MAO = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17], [0, 17],
];

function limitar(valor, minimo, maximo) {
  return Math.max(minimo, Math.min(maximo, valor));
}

function norma3d(a, b = [0, 0, 0]) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function calcularVetorQuadro(pontosMao3d, larguraQuadro = 480, alturaQuadro = 640, pontosPose = null, outraMao3d = null) {
  if (!Array.isArray(pontosMao3d) || pontosMao3d.length !== 21) return null;

  const maiorLado = Math.max(larguraQuadro || 480, alturaQuadro || 640, 1);
  const escalaX = (larguraQuadro || 480) / maiorLado;
  const escalaY = (alturaQuadro || 640) / maiorLado;

  const mao = [];
  for (let i = 0; i < 21; i += 1) {
    const p = pontosMao3d[i];
    if (!Array.isArray(p) || p.length < 3) return null;
    const x = Number(p[0]) * escalaX;
    const y = Number(p[1]) * escalaY;
    const z = 0;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    mao.push([x, y, z]);
  }

  const pulso = mao[0];
  const pontosRef = [
    [mao[0][0], mao[0][1]],
    [mao[8][0], mao[8][1]],
    [mao[20][0], mao[20][1]],
    [mao[9][0], mao[9][1]],
  ];
  const relativos = mao.map((p) => [p[0] - pulso[0], p[1] - pulso[1], p[2] - pulso[2]]);
  const tamanhoPalma = (
    norma3d(relativos[5]) +
    norma3d(relativos[9]) +
    norma3d(relativos[13]) +
    norma3d(relativos[17])
  ) / 4.0;
  const escalaMao = tamanhoPalma / 0.45;
  if (escalaMao < 0.02) return null;

  const h = relativos.map((p) => [p[0] / escalaMao, p[1] / escalaMao, p[2] / escalaMao]);
  const detPalma = h[5][0] * h[17][1] - h[5][1] * h[17][0];
  if (detPalma < 0) {
    for (let i = 0; i < 21; i += 1) {
      h[i][0] *= -1;
    }
  }

  const valores = [];
  for (let i = 0; i < 21; i += 1) {
    valores.push(h[i][0], h[i][1], h[i][2]);
  }

  const pontas = [4, 8, 12, 16, 20];
  const bases = [2, 5, 9, 13, 17];
  const cadeias = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ];

  for (const p of pontas) {
    valores.push(norma3d(h[p]));
  }
  for (let i = 0; i < pontas.length; i += 1) {
    valores.push(norma3d(h[pontas[i]], h[bases[i]]));
  }
  for (const [a, b, c, d] of cadeias) {
    const v1 = [h[b][0] - h[a][0], h[b][1] - h[a][1], h[b][2] - h[a][2]];
    const v2 = [h[d][0] - h[c][0], h[d][1] - h[c][1], h[d][2] - h[c][2]];
    const n = norma3d(v1) * norma3d(v2) + 1e-6;
    valores.push((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / n);
  }
  for (const p of [8, 12, 16, 20]) {
    valores.push(norma3d(h[4], h[p]));
  }
  valores.push(norma3d(h[8], h[12]));
  valores.push(norma3d(h[12], h[16]));
  valores.push(norma3d(h[16], h[20]));
  for (const b of [5, 9, 13]) {
    valores.push(norma3d(h[4], h[b]));
  }

  if (Array.isArray(pontosPose) && pontosPose.length >= 9) {
    const narizX = Number(pontosPose[0][0]) * escalaX;
    const narizY = Number(pontosPose[0][1]) * escalaY;
    const ombroEsqX = Number(pontosPose[7][0]) * escalaX;
    const ombroEsqY = Number(pontosPose[7][1]) * escalaY;
    const ombroDirX = Number(pontosPose[8][0]) * escalaX;
    const ombroDirY = Number(pontosPose[8][1]) * escalaY;
    const largOmbros = Math.max(Math.hypot(ombroEsqX - ombroDirX, ombroEsqY - ombroDirY), 0.12);

    valores.push(
      limitar((pulso[1] - narizY) / largOmbros, -2.0, 2.5),
      limitar((mao[8][1] - narizY) / largOmbros, -2.0, 2.5),
      limitar(Math.abs(pulso[0] - narizX) / largOmbros, 0.0, 2.5),
      1.0
    );
  } else {
    valores.push(
      limitar((Number(pontosMao3d[0][1]) - 0.30) * 2.0, -2.0, 2.5),
      limitar((Number(pontosMao3d[8][1]) - 0.30) * 2.0, -2.0, 2.5),
      limitar(Math.abs(Number(pontosMao3d[0][0]) - 0.50) * 2.5, 0.0, 2.5),
      0.0
    );
  }

  let temOutraMao = 0.0;
  let distPulsos = 4.0;
  let distPontasInd = 4.0;
  let distPontasMed = 4.0;
  let difAlturaPulsos = 0.0;
  let cosPalmas = 0.0;

  if (Array.isArray(outraMao3d) && outraMao3d.length === 21) {
    const outra = [];
    let outraValida = true;
    for (let i = 0; i < 21; i += 1) {
      const p = outraMao3d[i];
      if (!Array.isArray(p) || p.length < 2) {
        outraValida = false;
        break;
      }
      const ox = Number(p[0]) * escalaX;
      const oy = Number(p[1]) * escalaY;
      if (!Number.isFinite(ox) || !Number.isFinite(oy)) {
        outraValida = false;
        break;
      }
      outra.push([ox, oy]);
    }
    if (outraValida) {
      temOutraMao = 1.0;
      distPulsos = limitar(Math.hypot(mao[0][0] - outra[0][0], mao[0][1] - outra[0][1]) / escalaMao, 0.0, 4.0);
      distPontasInd = limitar(Math.hypot(mao[8][0] - outra[8][0], mao[8][1] - outra[8][1]) / escalaMao, 0.0, 4.0);
      distPontasMed = limitar(Math.hypot(mao[12][0] - outra[12][0], mao[12][1] - outra[12][1]) / escalaMao, 0.0, 4.0);
      difAlturaPulsos = limitar((outra[0][1] - mao[0][1]) / escalaMao, -3.0, 3.0);
      const dirAX = mao[9][0] - mao[0][0];
      const dirAY = mao[9][1] - mao[0][1];
      const dirBX = outra[9][0] - outra[0][0];
      const dirBY = outra[9][1] - outra[0][1];
      const normaAB = Math.hypot(dirAX, dirAY) * Math.hypot(dirBX, dirBY) + 1e-6;
      cosPalmas = limitar((dirAX * dirBX + dirAY * dirBY) / normaAB, -1.0, 1.0);
    }
  }

  valores.push(temOutraMao, distPulsos, distPontasInd, distPontasMed, difAlturaPulsos, cosPalmas);

  return {
    vetor: valores,
    pontosRef,
    centro: [Number(pontosMao3d[0][0]), Number(pontosMao3d[0][1])],
    escalaMao,
    instante: Date.now(),
  };
}

function montarVetorTemporal(historico) {
  const atual = historico[historico.length - 1];
  const primeiro = historico[0];
  const escala = Math.max(atual.escalaMao, 0.05);

  let maxAbsDx = 0;
  let maxAmpX = 0;
  let maxAmpY = 0;
  let maxEnergiaX = 0;
  let maxEnergiaY = 0;

  for (let r = 0; r < 4; r += 1) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let energiaRX = 0;
    let energiaRY = 0;

    for (let i = 0; i < historico.length; i += 1) {
      const px = historico[i].pontosRef[r][0];
      const py = historico[i].pontosRef[r][1];
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
      if (i > 0) {
        energiaRX += Math.abs(px - historico[i - 1].pontosRef[r][0]) / escala;
        energiaRY += Math.abs(py - historico[i - 1].pontosRef[r][1]) / escala;
      }
    }

    const absDxR = Math.abs(atual.pontosRef[r][0] - primeiro.pontosRef[r][0]) / escala;
    const ampXR = (maxX - minX) / escala;
    const ampYR = (maxY - minY) / escala;

    if (absDxR > maxAbsDx) maxAbsDx = absDxR;
    if (ampXR > maxAmpX) maxAmpX = ampXR;
    if (ampYR > maxAmpY) maxAmpY = ampYR;
    if (energiaRX > maxEnergiaX) maxEnergiaX = energiaRX;
    if (energiaRY > maxEnergiaY) maxEnergiaY = energiaRY;
  }

  const dyPulso = (atual.pontosRef[0][1] - primeiro.pontosRef[0][1]) / escala;
  const dyBase = (atual.pontosRef[3][1] - primeiro.pontosRef[3][1]) / escala;
  const dyBruto = Math.abs(dyPulso) >= Math.abs(dyBase) ? dyPulso : dyBase;

  const absDx = limitar(maxAbsDx, 0.0, 3.0);
  const dy = limitar(dyBruto, -3.0, 3.0);
  const ampX = limitar(maxAmpX, 0.0, 4.0);
  const ampY = limitar(maxAmpY, 0.0, 4.0);
  const energiaX = Math.min(maxEnergiaX, 4.0);
  const energiaY = Math.min(maxEnergiaY, 4.0);
  const energiaTotal = Math.min(Math.hypot(energiaX, energiaY), 5.0);

  const vetorBase = atual.vetor.slice();
  if (vetorBase.length >= 98 && vetorBase[92] < 0.5 && historico.length >= 2) {
    const inicioBusca = Math.max(0, historico.length - 3);
    for (let i = historico.length - 2; i >= inicioBusca; i -= 1) {
      if (historico[i].vetor.length >= 98 && historico[i].vetor[92] > 0.5) {
        for (let k = 92; k < 98; k += 1) {
          vetorBase[k] = historico[i].vetor[k];
        }
        break;
      }
    }
  }

  const deltaDedos = [];
  for (let i = 0; i < 5; i += 1) {
    deltaDedos.push(atual.vetor[63 + i] - primeiro.vetor[63 + i]);
  }

  return vetorBase.concat([absDx, dy, ampX, ampY, energiaX, energiaY, energiaTotal], deltaDedos);
}

function preverModeloLocal(modelo, caracteristicas) {
  if (!modelo || caracteristicas.length !== modelo.quantidade_caracteristicas) {
    return { previsao: '', confianca: 0 };
  }

  const probabilidades = new Array(modelo.rotulos.length).fill(0);
  for (const arvore of modelo.arvores) {
    let no = 0;
    while (arvore.esquerda[no] !== -1) {
      const indice = arvore.caracteristica[no];
      no = caracteristicas[indice] <= arvore.limite[no]
        ? arvore.esquerda[no]
        : arvore.direita[no];
    }
    const folha = arvore.probabilidades[no];
    for (let classe = 0; classe < probabilidades.length; classe += 1) {
      probabilidades[classe] += folha[classe];
    }
  }

  const quantidade = modelo.arvores.length || 1;
  let melhorIndice = 0;
  for (let i = 0; i < probabilidades.length; i += 1) {
    probabilidades[i] /= quantidade;
    if (probabilidades[i] > probabilidades[melhorIndice]) melhorIndice = i;
  }
  return {
    previsao: modelo.rotulos[melhorIndice] || '',
    confianca: probabilidades[melhorIndice] || 0,
  };
}

function escolherRotuloVotacao(votacoes) {
  const pesosBase = [0.60, 0.85, 1.30];
  const pesos = pesosBase.slice(pesosBase.length - votacoes.length);
  const infoPorRotulo = new Map();

  for (let i = 0; i < votacoes.length; i += 1) {
    const voto = votacoes[i];
    if (!voto.previsao) continue;
    const atual = infoPorRotulo.get(voto.previsao) || { pontuacao: 0, somaConf: 0, votos: 0 };
    atual.pontuacao += voto.confianca * pesos[i];
    atual.somaConf += voto.confianca;
    atual.votos += 1;
    infoPorRotulo.set(voto.previsao, atual);
  }

  let melhorRotulo = '';
  let melhorInfo = null;
  for (const [rotulo, info] of infoPorRotulo.entries()) {
    if (!melhorInfo || info.pontuacao > melhorInfo.pontuacao) {
      melhorRotulo = rotulo;
      melhorInfo = info;
    }
  }

  if (!melhorInfo) return { predicao: '', mediaConf: 0 };
  const mediaConf = melhorInfo.somaConf / melhorInfo.votos;
  const aceito = (melhorInfo.votos >= 2 && mediaConf >= 0.38) || mediaConf >= 0.50;
  return { predicao: aceito ? melhorRotulo : '', mediaConf };
}

const RastreamentoMao = React.memo(function RastreamentoMao({
  pontosMao,
  pontosMaos,
  espelhado,
  dimensoesVista,
  tamanhoQuadro,
}) {
  const larguraVista = dimensoesVista.largura;
  const alturaVista = dimensoesVista.altura;
  if (larguraVista <= 0 || alturaVista <= 0) return null;

  const larguraQuadro = tamanhoQuadro.largura;
  const alturaQuadro = tamanhoQuadro.altura;

  function converterPonto(ponto) {
    const nx = espelhado ? 1 - ponto[0] : ponto[0];
    const ny = ponto[1];

    if (larguraQuadro > 0 && alturaQuadro > 0) {
      const escala = Math.max(larguraVista / larguraQuadro, alturaVista / alturaQuadro);
      const larguraRenderizada = larguraQuadro * escala;
      const alturaRenderizada = alturaQuadro * escala;
      const corteX = (larguraRenderizada - larguraVista) / 2;
      const corteY = (alturaRenderizada - alturaVista) / 2;
      return {
        x: nx * larguraRenderizada - corteX,
        y: ny * alturaRenderizada - corteY,
      };
    }

    return {
      x: nx * larguraVista,
      y: ny * alturaVista,
    };
  }

  const maosVisuais = pontosMaos && pontosMaos.length > 0
    ? pontosMaos
    : (pontosMao && pontosMao.length === 21 ? [pontosMao] : []);

  if (maosVisuais.length === 0) return null;

  const elementos = [];
  maosVisuais.forEach((mao, indiceMao) => {
    if (!mao || mao.length !== 21) return;

    CONEXOES_MAO.forEach((conexao, i) => {
      const p1 = converterPonto(mao[conexao[0]]);
      const p2 = converterPonto(mao[conexao[1]]);
      elementos.push(
        <Line
          key={`linha-${indiceMao}-${i}`}
          x1={p1.x}
          y1={p1.y}
          x2={p2.x}
          y2={p2.y}
          stroke="#16A34A"
          strokeWidth={3}
          strokeLinecap="round"
        />
      );
    });

    mao.forEach((ponto, i) => {
      const p = converterPonto(ponto);
      elementos.push(
        <Circle
          key={`ponto-${indiceMao}-${i}`}
          cx={p.x}
          cy={p.y}
          r={i === 0 ? 5.5 : 4.5}
          fill="#FFFFFF"
          stroke="#15803D"
          strokeWidth={2}
        />
      );
    });
  });

  if (elementos.length === 0) return null;

  return (
    <View style={estilosOverlay.container} pointerEvents="none">
      <Svg width={larguraVista} height={alturaVista}>
        {elementos}
      </Svg>
    </View>
  );
});

const estilosOverlay = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
});

function AplicativoPrincipal() {
  const [permissao, pedirPermissao] = useCameraPermissions();
  const [tipoCamera, setTipoCamera] = useState('front');
  const [cameraPronta, setCameraPronta] = useState(!usarCameraNativa);
  const [tamanhoFoto, setTamanhoFoto] = useState(undefined);
  const [dimensoesVista, setDimensoesVista] = useState({ largura: 0, altura: 0 });
  const [tamanhoQuadro, setTamanhoQuadro] = useState({ largura: 360, altura: 480 });
  const [statusConexao, setStatusConexao] = useState(
    usarCameraNativa ? 'Rastreamento local ativo' : 'Desconectado'
  );
  const [latencia, setLatencia] = useState(0);
  const [latenciaServidor, setLatenciaServidor] = useState(0);
  const [letraAtual, setLetraAtual] = useState('-');
  const [confianca, setConfianca] = useState(0);
  const [textoCompleto, setTextoCompleto] = useState('');
  const [capturando, setCapturando] = useState(false);
  const [ipServidor, setIpServidor] = useState(obterIpPadraoServidor);
  const [quadrosBuffer, setQuadrosBuffer] = useState(0);
  const [gravandoGesto, setGravandoGesto] = useState(false);
  const [tempoGestoRestante, setTempoGestoRestante] = useState(0);
  const [contagemRegressiva, setContagemRegressiva] = useState(0);
  const [pontosMao, setPontosMao] = useState([]);
  const [pontosMaos, setPontosMaos] = useState([]);

  const cameraRef = useRef(null);
  const capturandoRef = useRef(false);
  const trocandoCameraRef = useRef(false);
  const socketRef = useRef(null);
  const timerCapturaRef = useRef(null);
  const enviandoRef = useRef(false);
  const aguardandoRespostaRef = useRef(false);
  const quadroPendenteRef = useRef(null);
  const inicioQuadroRef = useRef(0);
  const ultimaLetraRef = useRef('');
  const ultimoSinalInseridoRef = useRef('');
  const contagemEstavelRef = useRef(0);
  const votacoesModeloRef = useRef([]);
  const gravandoGestoRef = useRef(false);
  const preparandoGestoRef = useRef(false);
  const melhorPredicaoGestoRef = useRef({ rotulo: '', conf: 0, contagem: 0 });
  const historicoLocalRef = useRef([]);
  const falhasMaoLocalRef = useRef(0);
  const quadrosSemMaoWsRef = useRef(0);
  const estavaEmMovimentoLocalRef = useRef(false);

  useEffect(() => {
    if (usarCameraNativa) {
      setStatusConexao('Rastreamento local ativo');
      return () => pararCaptura();
    }
    conectarSocket();
    return () => {
      fecharSocket();
      pararCaptura();
    };
  }, [ipServidor]);

  function atualizarDimensoesQuadro(largura, altura) {
    if (!largura || !altura) return;
    setTamanhoQuadro((atual) => (
      atual.largura === largura && atual.altura === altura
        ? atual
        : { largura, altura }
    ));
  }

  function fecharSocket() {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }

  function limparHistoricoPredicao() {
    historicoLocalRef.current = [];
    votacoesModeloRef.current = [];
    falhasMaoLocalRef.current = 0;
    quadrosSemMaoWsRef.current = 0;
    estavaEmMovimentoLocalRef.current = false;
    ultimaLetraRef.current = '';
    ultimoSinalInseridoRef.current = '';
    contagemEstavelRef.current = 0;
  }

  function enviarQuadroSocket(base64) {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }
    aguardandoRespostaRef.current = true;
    inicioQuadroRef.current = Date.now();
    socketRef.current.send(base64);
    return true;
  }

  function conectarSocket() {
    if (usarCameraNativa) {
      setStatusConexao('Rastreamento local ativo');
      return;
    }
    fecharSocket();
    setStatusConexao('Conectando...');

    try {
      const url = `ws://${ipServidor}:8000/ws`;
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setStatusConexao('Conectado');
        agendarProximoQuadro();
      };

      ws.onmessage = (evento) => {
        aguardandoRespostaRef.current = false;
        const latenciaQuadro = inicioQuadroRef.current
          ? Date.now() - inicioQuadroRef.current
          : 0;
        inicioQuadroRef.current = 0;
        if (latenciaQuadro) setLatencia(latenciaQuadro);

        const pendente = quadroPendenteRef.current;
        if (pendente && socketRef.current === ws && (capturandoRef.current || gravandoGestoRef.current)) {
          quadroPendenteRef.current = null;
          enviarQuadroSocket(pendente);
        }

        try {
          const dados = JSON.parse(evento.data);
          if (typeof dados.tempo_ms === 'number') {
            setLatenciaServidor(Math.round(dados.tempo_ms));
          }
          atualizarDimensoesQuadro(dados.largura_quadro, dados.altura_quadro);

          if (dados.status === 'ok') {
            quadrosSemMaoWsRef.current = 0;
            const predicao = dados.previsao || dados.prediction || '';
            const conf = dados.confianca || dados.confidence || 0;
            const tempoServidor = dados.tempo_ms || 0;
            const bufferLen = dados.quadros_buffer || 0;

            if (!latenciaQuadro) setLatencia(Math.round(tempoServidor));
            setQuadrosBuffer(bufferLen);
            setPontosMao(dados.pontos_mao || []);
            setPontosMaos(dados.pontos_maos || []);

            if (predicao && conf >= 0.38) {
              setLetraAtual(predicao);
              setConfianca(Math.round(conf * 100));
            } else {
              setLetraAtual('Analisando sinal...');
              setConfianca(0);
            }

            if (gravandoGestoRef.current && predicao && conf >= 0.40) {
              const anterior = melhorPredicaoGestoRef.current;
              const contagem = anterior.rotulo === predicao ? anterior.contagem + 1 : 1;
              if (conf >= anterior.conf || contagem > anterior.contagem) {
                melhorPredicaoGestoRef.current = { rotulo: predicao, conf, contagem };
              }
            }

            if (!gravandoGestoRef.current && predicao && conf >= 0.48) {
              if (predicao === ultimaLetraRef.current) {
                contagemEstavelRef.current += 1;
              } else {
                ultimaLetraRef.current = predicao;
                contagemEstavelRef.current = 1;
              }
              const prontoInserir = contagemEstavelRef.current >= 2 || (conf >= 0.66 && bufferLen >= 3);
              if (prontoInserir && ultimoSinalInseridoRef.current !== predicao) {
                adicionarAoTexto(predicao);
                ultimoSinalInseridoRef.current = predicao;
              }
            } else if (!gravandoGestoRef.current && !predicao) {
              ultimaLetraRef.current = '';
              contagemEstavelRef.current = 0;
            }
          } else {
            quadrosSemMaoWsRef.current += 1;
            if (quadrosSemMaoWsRef.current >= 3) {
              setPontosMao([]);
              setPontosMaos([]);
              setQuadrosBuffer(0);
              ultimaLetraRef.current = '';
              ultimoSinalInseridoRef.current = '';
              contagemEstavelRef.current = 0;
              if (!gravandoGestoRef.current) {
                setLetraAtual('Mostre a mao na camera');
                setConfianca(0);
              }
            }
          }
        } catch {
          inicioQuadroRef.current = 0;
        }

        if (socketRef.current === ws && !enviandoRef.current && !quadroPendenteRef.current) {
          agendarProximoQuadro();
        }
      };

      ws.onerror = () => {
        aguardandoRespostaRef.current = false;
        quadroPendenteRef.current = null;
        inicioQuadroRef.current = 0;
        setStatusConexao('Erro de conexao');
      };

      ws.onclose = () => {
        aguardandoRespostaRef.current = false;
        quadroPendenteRef.current = null;
        inicioQuadroRef.current = 0;
        setStatusConexao('Desconectado');
      };

      socketRef.current = ws;
    } catch {
      setStatusConexao('Falha ao abrir socket');
    }
  }

  function adicionarAoTexto(sinal) {
    if (
      !sinal ||
      sinal === '-' ||
      sinal === 'Mostre a mao na camera' ||
      sinal === 'Analisando sinal...' ||
      sinal.startsWith('Prepare-se') ||
      sinal.startsWith('Gravando') ||
      sinal === 'Sinal nao reconhecido'
    ) {
      return;
    }

    setTextoCompleto((anterior) => {
      const textoLimpo = anterior.trim();
      if (!textoLimpo) {
        return sinal;
      }
      if (sinal.length > 1 || anterior.endsWith(' ')) {
        return anterior.endsWith(' ') ? anterior + sinal : `${anterior} ${sinal}`;
      }
      return anterior + sinal;
    });
  }

  function agendarProximoQuadro(atrasoMs = 0) {
    if (!capturandoRef.current && !gravandoGestoRef.current) return;
    if (timerCapturaRef.current) clearTimeout(timerCapturaRef.current);
    timerCapturaRef.current = setTimeout(() => {
      timerCapturaRef.current = null;
      capturarEEnviar();
    }, atrasoMs);
  }

  function apagarUltimo() {
    setTextoCompleto((anterior) => {
      if (!anterior) return '';
      const texto = anterior.trimEnd();
      const ultimoEspaco = texto.lastIndexOf(' ');
      if (ultimoEspaco === -1) {
        return '';
      }
      return texto.substring(0, ultimoEspaco);
    });
  }

  function receberQuadroLocal(evento) {
    const dados = evento?.nativeEvent || {};
    const inicioClassificacao = Date.now();
    const tempoRastreamento = typeof dados.tempo_ms === 'number' ? Math.round(dados.tempo_ms) : 0;

    atualizarDimensoesQuadro(dados.largura_quadro, dados.altura_quadro);
    const temPontos = (Array.isArray(dados.pontos_maos) && dados.pontos_maos.length > 0)
      || (Array.isArray(dados.pontos_mao) && dados.pontos_mao.length === 21);
    if (temPontos) {
      setPontosMao(dados.pontos_mao || []);
      setPontosMaos(dados.pontos_maos || []);
    } else if (!capturandoRef.current && !gravandoGestoRef.current) {
      setPontosMao([]);
      setPontosMaos([]);
    }
    setLatencia(tempoRastreamento);
    setLatenciaServidor(0);

    if (!capturandoRef.current && !gravandoGestoRef.current) {
      return;
    }

    const listaMaos3d = Array.isArray(dados.pontos_maos_3d) && dados.pontos_maos_3d.length > 0
      ? dados.pontos_maos_3d
      : (dados.tem_mao && Array.isArray(dados.pontos_mao_3d) ? [dados.pontos_mao_3d] : []);

    const quadrosMaos = [];
    for (let idx = 0; idx < listaMaos3d.length; idx += 1) {
      const outraMao3d = listaMaos3d.length === 2 ? listaMaos3d[1 - idx] : null;
      const q = calcularVetorQuadro(
        listaMaos3d[idx],
        dados.largura_quadro || 360,
        dados.altura_quadro || 480,
        dados.pontos_pose,
        outraMao3d
      );
      if (q) quadrosMaos.push(q);
    }

    if (quadrosMaos.length === 0) {
      falhasMaoLocalRef.current += 1;
      if (falhasMaoLocalRef.current >= 3) {
        setPontosMao([]);
        setPontosMaos([]);
        if (!gravandoGestoRef.current) {
          setLetraAtual('Mostre a mao na camera');
          setConfianca(0);
        }
      }
      if (falhasMaoLocalRef.current >= 5) {
        limparHistoricoPredicao();
        setQuadrosBuffer(0);
      }
      return;
    }

    falhasMaoLocalRef.current = 0;
    const agoraMs = Date.now();
    const trilhas = historicoLocalRef.current.filter((t) => agoraMs - t.ultimoInstante <= 1100);
    const limiteDist = quadrosMaos.length === 1 && trilhas.length === 1 ? 0.85 : 0.65;
    const pares = [];
    for (let iq = 0; iq < quadrosMaos.length; iq += 1) {
      for (let it = 0; it < trilhas.length; it += 1) {
        const dist = Math.hypot(
          quadrosMaos[iq].centro[0] - trilhas[it].centro[0],
          quadrosMaos[iq].centro[1] - trilhas[it].centro[1]
        );
        if (dist <= limiteDist) pares.push([dist, iq, it]);
      }
    }
    pares.sort((a, b) => a[0] - b[0]);

    const usadosQ = new Set();
    const usadosT = new Set();
    const trilhasAtivas = [];

    for (const [, iq, it] of pares) {
      if (usadosQ.has(iq) || usadosT.has(it)) continue;
      usadosQ.add(iq);
      usadosT.add(it);
      const q = quadrosMaos[iq];
      const trilha = trilhas[it];
      while (trilha.quadros.length > 0 && q.instante - trilha.quadros[0].instante > JANELA_TEMPORAL_MS) {
        trilha.quadros.shift();
      }
      if (trilha.quadros.length >= 2 && q.instante - trilha.quadros[trilha.quadros.length - 2].instante < 110) {
        trilha.quadros[trilha.quadros.length - 1] = q;
      } else {
        trilha.quadros.push(q);
        if (trilha.quadros.length > 9) trilha.quadros.shift();
      }
      trilha.centro = q.centro;
      trilha.ultimoInstante = q.instante;
      trilhasAtivas.push(trilha);
    }

    for (let iq = 0; iq < quadrosMaos.length; iq += 1) {
      if (usadosQ.has(iq)) continue;
      const q = quadrosMaos[iq];
      const nova = { centro: q.centro, ultimoInstante: q.instante, quadros: [q] };
      trilhas.push(nova);
      trilhasAtivas.push(nova);
    }

    trilhas.sort((a, b) => b.ultimoInstante - a.ultimoInstante);
    historicoLocalRef.current = trilhas.slice(0, 2);

    const menorY = Math.min(...trilhasAtivas.map((t) => t.centro[1]));
    const maiorY = Math.max(...trilhasAtivas.map((t) => t.centro[1]));
    const avaliacoes = trilhasAtivas.map((trilha) => {
      const caracteristicas = montarVetorTemporal(trilha.quadros);
      const res = preverModeloLocal(modeloLocal, caracteristicas);
      const emMov = caracteristicas[104] >= 0.28 || Math.max(caracteristicas[100], caracteristicas[101]) >= 0.24;
      let prioridade = res.confianca;
      if (trilhasAtivas.length > 1) {
        if (maiorY - menorY > 0.14 && trilha.centro[1] <= menorY + 0.02) prioridade += 0.08;
        if (emMov) prioridade += 0.14;
      }
      return {
        previsao: res.previsao,
        confianca: res.confianca,
        prioridade,
        quadros: trilha.quadros.length,
        emMovimento: emMov,
      };
    });

    let resultado;
    let emMovimentoAtual = false;
    let bufferAtual = 0;
    if (avaliacoes.length === 2 && avaliacoes[0].previsao && avaliacoes[0].previsao === avaliacoes[1].previsao) {
      resultado = {
        previsao: avaliacoes[0].previsao,
        confianca: Math.min(1.0, Math.max(avaliacoes[0].confianca, avaliacoes[1].confianca) + 0.05),
      };
      emMovimentoAtual = avaliacoes[0].emMovimento || avaliacoes[1].emMovimento;
      bufferAtual = Math.max(avaliacoes[0].quadros, avaliacoes[1].quadros);
    } else {
      avaliacoes.sort((a, b) => b.prioridade - a.prioridade);
      resultado = { previsao: avaliacoes[0].previsao, confianca: avaliacoes[0].confianca };
      emMovimentoAtual = avaliacoes[0].emMovimento;
      bufferAtual = avaliacoes[0].quadros;
    }
    setQuadrosBuffer(bufferAtual);

    setLatencia(tempoRastreamento + (Date.now() - inicioClassificacao));

    let votacoes = votacoesModeloRef.current;
    if (emMovimentoAtual && !estavaEmMovimentoLocalRef.current) {
      votacoes = votacoes.filter((v) => v.previsao === resultado.previsao);
      votacoesModeloRef.current = votacoes;
    }
    estavaEmMovimentoLocalRef.current = emMovimentoAtual;

    votacoes.push(resultado);
    if (votacoes.length > 3) votacoes.shift();

    const { predicao, mediaConf } = escolherRotuloVotacao(votacoes);

    if (predicao) {
      setLetraAtual(predicao);
      setConfianca(Math.round(mediaConf * 100));
    } else if (!gravandoGestoRef.current) {
      setLetraAtual('Analisando sinal...');
      setConfianca(0);
    }

    if (gravandoGestoRef.current && predicao && mediaConf >= 0.40) {
      const anterior = melhorPredicaoGestoRef.current;
      const contagem = anterior.rotulo === predicao ? anterior.contagem + 1 : 1;
      if (mediaConf >= anterior.conf || contagem > anterior.contagem) {
        melhorPredicaoGestoRef.current = {
          rotulo: predicao,
          conf: mediaConf,
          contagem,
        };
      }
    }

    if (!gravandoGestoRef.current && predicao && mediaConf >= 0.48) {
      if (predicao === ultimaLetraRef.current) {
        contagemEstavelRef.current += 1;
      } else {
        ultimaLetraRef.current = predicao;
        contagemEstavelRef.current = 1;
      }
      const prontoInserir = contagemEstavelRef.current >= 2 || (mediaConf >= 0.66 && bufferAtual >= 3);
      if (prontoInserir && ultimoSinalInseridoRef.current !== predicao) {
        adicionarAoTexto(predicao);
        ultimoSinalInseridoRef.current = predicao;
      }
    } else if (!gravandoGestoRef.current && !predicao) {
      ultimaLetraRef.current = '';
      contagemEstavelRef.current = 0;
    }
  }

  async function capturarEEnviar() {
    if (usarCameraNativa) return;
    if (
      (!capturandoRef.current && !gravandoGestoRef.current) ||
      enviandoRef.current ||
      quadroPendenteRef.current !== null ||
      !cameraRef.current ||
      trocandoCameraRef.current
    ) {
      return;
    }

    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    enviandoRef.current = true;
    let falhaNaCaptura = false;
    try {
      const foto = await cameraRef.current.takePictureAsync({
        quality: 0.30,
        base64: true,
        exif: false,
        shutterSound: false,
      });

      if (foto?.uri) {
        try {
          new File(foto.uri).delete();
        } catch {}
      }

      if (foto?.base64 && (capturandoRef.current || gravandoGestoRef.current)) {
        if (!aguardandoRespostaRef.current) {
          enviarQuadroSocket(foto.base64);
        } else {
          quadroPendenteRef.current = foto.base64;
        }
      }
    } catch {
      falhaNaCaptura = true;
    } finally {
      enviandoRef.current = false;
      if ((capturandoRef.current || gravandoGestoRef.current) && quadroPendenteRef.current === null) {
        agendarProximoQuadro(falhaNaCaptura ? 80 : 0);
      }
    }
  }

  useEffect(() => {
    if (capturando) {
      capturarEEnviar();
    } else {
      pararCaptura();
      if (!usarCameraNativa) {
        setPontosMao([]);
        setPontosMaos([]);
      }
    }
    return () => pararCaptura();
  }, [capturando]);

  function pararCaptura() {
    if (timerCapturaRef.current) {
      clearTimeout(timerCapturaRef.current);
      timerCapturaRef.current = null;
    }
    enviandoRef.current = false;
    aguardandoRespostaRef.current = false;
    quadroPendenteRef.current = null;
    inicioQuadroRef.current = 0;
  }

  function alternarCaptura() {
    const novo = !capturando;
    capturandoRef.current = novo;
    limparHistoricoPredicao();
    if (!novo) {
      setQuadrosBuffer(0);
    }
    if (usarCameraNativa) {
      setStatusConexao(novo ? 'Traducao local ativa' : 'Rastreamento local ativo');
    }
    setCapturando(novo);
  }

  function iniciarGravacaoGesto() {
    if (gravandoGestoRef.current || preparandoGestoRef.current) {
      return;
    }

    if (!usarCameraNativa && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)) {
      setStatusConexao('Desconectado! Verifique o IP');
      conectarSocket();
      return;
    }

    preparandoGestoRef.current = true;
    let contagem = 3;
    setContagemRegressiva(contagem);
    setLetraAtual(`Prepare-se (${contagem}s)...`);

    const timerPreparacao = setInterval(() => {
      contagem -= 1;
      if (contagem > 0) {
        setContagemRegressiva(contagem);
        setLetraAtual(`Prepare-se (${contagem}s)...`);
      } else {
        clearInterval(timerPreparacao);
        setContagemRegressiva(0);
        preparandoGestoRef.current = false;
        executarGravacaoGesto();
      }
    }, 1000);
  }

  function executarGravacaoGesto() {
    limparHistoricoPredicao();
    quadroPendenteRef.current = null;
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ tipo: 'limpar_buffer' }));
    }

    gravandoGestoRef.current = true;
    setGravandoGesto(true);
    melhorPredicaoGestoRef.current = { rotulo: '', conf: 0, contagem: 0 };
    setLetraAtual('Gravando sinal...');
    setTempoGestoRestante(2.5);

    capturarEEnviar();

    const tempoInicio = Date.now();
    const duracaoTotal = 2500;

    const timerProgresso = setInterval(() => {
      const decorrido = Date.now() - tempoInicio;
      const restante = Math.max(0, (duracaoTotal - decorrido) / 1000).toFixed(1);
      setTempoGestoRestante(parseFloat(restante));

      if (decorrido >= duracaoTotal) {
        clearInterval(timerProgresso);
        gravandoGestoRef.current = false;
        setGravandoGesto(false);
        setTempoGestoRestante(0);

        const melhor = melhorPredicaoGestoRef.current;
        const gestoEstavel = !usarCameraNativa || melhor.contagem >= 2;
        if (melhor.rotulo && melhor.conf >= 0.42 && gestoEstavel) {
          adicionarAoTexto(melhor.rotulo);
          setLetraAtual(melhor.rotulo);
          setConfianca(Math.round(melhor.conf * 100));
        } else {
          setLetraAtual('Sinal nao reconhecido');
          setConfianca(0);
        }
      }
    }, 100);
  }

  function alternarCamera() {
    if (trocandoCameraRef.current) {
      return;
    }

    trocandoCameraRef.current = true;
    quadroPendenteRef.current = null;
    setPontosMao([]);
    setPontosMaos([]);
    limparHistoricoPredicao();
    setTipoCamera((atual) => (atual === 'front' ? 'back' : 'front'));

    setTimeout(() => {
      trocandoCameraRef.current = false;
      agendarProximoQuadro();
    }, 600);
  }

  function limparTexto() {
    setTextoCompleto('');
    setLetraAtual('-');
    setConfianca(0);
    limparHistoricoPredicao();
  }

  function adicionarEspaco() {
    setTextoCompleto((anterior) => `${anterior} `);
  }

  if (!permissao) {
    return (
      <SafeAreaView style={estilos.telaCheia}>
        <View style={estilos.centralizado}>
          <Text style={estilos.textoPadrao}>Carregando permissoes...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!permissao.granted) {
    return (
      <SafeAreaView style={estilos.telaCheia}>
        <View style={estilos.centralizado}>
          <Text style={estilos.titulo}>Permissao da Camera</Text>
          <Text style={estilos.descricao}>Precisamos de acesso a camera para traduzir os sinais de Libras.</Text>
          <TouchableOpacity style={estilos.botao} onPress={pedirPermissao}>
            <Text style={estilos.textoBotao}>Permitir Acesso</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={estilos.telaCheia}>
      <ScrollView contentContainerStyle={estilos.conteudoScroll}>
        <View style={estilos.cabecalho}>
          <Text style={estilos.titulo}>Tradutor de Libras</Text>
          {!usarCameraNativa && (
            <View style={estilos.linhaConfiguracao}>
              <Text style={estilos.rotulo}>IP do Servidor:</Text>
              <TextInput
                style={estilos.campoTexto}
                value={ipServidor}
                onChangeText={setIpServidor}
                placeholder="192.168.0.XX"
                autoCapitalize="none"
              />
              <TouchableOpacity style={estilos.botaoPequeno} onPress={conectarSocket}>
                <Text style={estilos.textoBotaoPequeno}>Conectar</Text>
              </TouchableOpacity>
            </View>
          )}
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>Status: {statusConexao}</Text>
            <Text style={estilos.rotuloStatus}>Buffer: {quadrosBuffer}/9 quadros</Text>
          </View>
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>
              {usarCameraNativa ? `Inferencia local: ${latencia} ms` : `Latencia: ${latencia} ms`}
            </Text>
            {!usarCameraNativa && <Text style={estilos.rotuloStatus}>Servidor: {latenciaServidor} ms</Text>}
          </View>
        </View>

        <View
          style={estilos.areaCamera}
          onLayout={(evento) => {
            const { width, height } = evento.nativeEvent.layout;
            setDimensoesVista((atual) => (
              atual.largura === width && atual.altura === height
                ? atual
                : { largura: width, altura: height }
            ));
          }}
        >
          {usarCameraNativa ? (
            <LibrasNativeCamera
              key={tipoCamera}
              style={estilos.camera}
              active={permissao.granted}
              tracking={permissao.granted}
              facing={tipoCamera}
              onCameraReady={() => {
                setCameraPronta(true);
                trocandoCameraRef.current = false;
              }}
              onCameraError={(evento) => {
                const mensagem = evento?.nativeEvent?.message || 'Falha na camera local';
                setStatusConexao(`Camera: ${mensagem}`);
              }}
              onLandmarks={receberQuadroLocal}
            />
          ) : (
            <CameraView
              ref={cameraRef}
              style={estilos.camera}
              facing={tipoCamera}
              ratio="4:3"
              animateShutter={false}
              flash="off"
              mute={true}
              onCameraReady={() => {
                setCameraPronta(true);
                cameraRef.current?.getAvailablePictureSizesAsync()
                  .then((tamanhos) => {
                    const resolucoes = tamanhos
                      .map((tamanho) => {
                        const [largura, altura] = tamanho.split('x').map(Number);
                        const maior = Math.max(largura, altura);
                        const menor = Math.min(largura, altura);
                        return {
                          tamanho,
                          largura,
                          altura,
                          maior,
                          menor,
                          area: largura * altura,
                          quatroPorTres: Math.abs(maior / menor - 4 / 3) < 0.06,
                        };
                      })
                      .filter((item) => item.largura && item.altura)
                      .sort((a, b) => a.area - b.area);
                    const recomendada = resolucoes.find((item) => item.quatroPorTres && item.maior >= 320 && item.maior <= 480)
                      || resolucoes.find((item) => item.quatroPorTres && item.maior >= 320 && item.maior <= 640)
                      || resolucoes.find((item) => item.maior >= 320 && item.maior <= 480)
                      || resolucoes.find((item) => item.maior >= 320 && item.maior <= 640)
                      || resolucoes[0];
                    if (recomendada) setTamanhoFoto(recomendada.tamanho);
                  })
                  .catch(() => {});
              }}
              pictureSize={tamanhoFoto}
              onMountError={() => {
                setCameraPronta(false);
                trocandoCameraRef.current = false;
              }}
            />
          )}
          <RastreamentoMao
            pontosMao={pontosMao}
            pontosMaos={pontosMaos}
            espelhado={tipoCamera === 'front'}
            dimensoesVista={dimensoesVista}
            tamanhoQuadro={tamanhoQuadro}
          />
          <TouchableOpacity
            style={estilos.etiquetaCamera}
            onPress={alternarCamera}
            activeOpacity={0.7}
            disabled={!cameraPronta}
          >
            <Text style={estilos.textoEtiquetaCamera}>
              {!cameraPronta
                ? 'Iniciando camera...'
                : (tipoCamera === 'front' ? 'Camera: Frontal' : 'Camera: Traseira')}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={estilos.painelResultado}>
          <View style={estilos.caixaPrevisao}>
            <Text style={estilos.rotuloPainel}>Sinal Detectado:</Text>
            <Text style={estilos.valorPrevisao}>{letraAtual}</Text>
            <Text style={estilos.valorConfianca}>Confianca: {confianca}%</Text>
          </View>

          <View style={estilos.caixaTexto}>
            <Text style={estilos.rotuloPainel}>Texto Traduzido:</Text>
            <Text style={estilos.valorTexto}>{textoCompleto || '(nenhum sinal gravado)'}</Text>
          </View>
        </View>

        <View style={estilos.painelBotoes}>
          <TouchableOpacity
            style={[
              estilos.botaoDestaque,
              contagemRegressiva > 0 || gravandoGesto ? estilos.botaoGravando : null,
            ]}
            onPress={iniciarGravacaoGesto}
            disabled={contagemRegressiva > 0 || gravandoGesto || !cameraPronta}
          >
            <Text style={estilos.textoBotaoDestaque}>
              {contagemRegressiva > 0
                ? `Prepare-se (${contagemRegressiva}s)...`
                : gravandoGesto
                ? `Gravando Gesto... (${tempoGestoRestante}s)`
                : 'Gravar Gesto Longo (2.5s)'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[estilos.botao, capturando && !gravandoGesto ? estilos.botaoAtivo : null]}
            onPress={alternarCaptura}
            disabled={gravandoGesto}
          >
            <Text style={estilos.textoBotao}>
              {capturando ? 'Pausar Traducao Continua' : 'Iniciar Traducao Continua'}
            </Text>
          </TouchableOpacity>

          <View style={estilos.linhaBotoes}>
            <TouchableOpacity
              style={estilos.botaoAcao}
              onPress={() => adicionarAoTexto(letraAtual)}
            >
              <Text style={estilos.textoBotaoAcao}>+ Frase</Text>
            </TouchableOpacity>

            <TouchableOpacity style={estilos.botaoAcao} onPress={adicionarEspaco}>
              <Text style={estilos.textoBotaoAcao}>Espaco</Text>
            </TouchableOpacity>

            <TouchableOpacity style={estilos.botaoAcao} onPress={apagarUltimo}>
              <Text style={estilos.textoBotaoAcao}>Apagar</Text>
            </TouchableOpacity>

            <TouchableOpacity style={estilos.botaoAcao} onPress={limparTexto}>
              <Text style={estilos.textoBotaoAcao}>Limpar</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={[estilos.botaoSecundario, !cameraPronta ? estilos.botaoDesabilitado : null]}
            onPress={alternarCamera}
            disabled={!cameraPronta}
          >
            <Text style={estilos.textoBotaoSecundario}>
              {!cameraPronta
                ? 'Trocando Camera...'
                : (tipoCamera === 'front' ? 'Trocar para Camera Traseira' : 'Trocar para Camera Frontal')}
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AplicativoPrincipal />
    </SafeAreaProvider>
  );
}

const estilos = StyleSheet.create({
  telaCheia: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  conteudoScroll: {
    padding: 16,
    alignItems: 'center',
  },
  centralizado: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#FFFFFF',
  },
  cabecalho: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    padding: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    marginBottom: 12,
  },
  titulo: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 10,
    textAlign: 'center',
  },
  descricao: {
    fontSize: 14,
    color: '#4B5563',
    textAlign: 'center',
    marginBottom: 16,
  },
  linhaConfiguracao: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  rotulo: {
    fontSize: 13,
    color: '#374151',
    marginRight: 6,
  },
  campoTexto: {
    flex: 1,
    height: 36,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 6,
    paddingHorizontal: 8,
    backgroundColor: '#FFFFFF',
    fontSize: 13,
  },
  linhaStatus: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  rotuloStatus: {
    fontSize: 12,
    color: '#4B5563',
  },
  areaCamera: {
    width: '100%',
    aspectRatio: 3 / 4,
    maxHeight: 420,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#D1D5DB',
    backgroundColor: '#E5E7EB',
    marginBottom: 12,
  },
  camera: {
    flex: 1,
  },
  etiquetaCamera: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#D1D5DB',
  },
  textoEtiquetaCamera: {
    fontSize: 12,
    color: '#374151',
    fontWeight: '500',
  },
  painelResultado: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    padding: 14,
    marginBottom: 12,
  },
  caixaPrevisao: {
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    paddingBottom: 10,
    marginBottom: 10,
  },
  rotuloPainel: {
    fontSize: 13,
    color: '#6B7280',
    marginBottom: 4,
  },
  valorPrevisao: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#111827',
    minHeight: 38,
    textAlign: 'center',
  },
  valorConfianca: {
    fontSize: 12,
    color: '#4B5563',
    marginTop: 2,
    minHeight: 16,
  },
  caixaTexto: {
    minHeight: 50,
  },
  valorTexto: {
    fontSize: 18,
    color: '#111827',
    fontWeight: '500',
  },
  painelBotoes: {
    width: '100%',
    gap: 8,
  },
  botaoDestaque: {
    backgroundColor: '#111827',
    paddingVertical: 13,
    borderRadius: 6,
    alignItems: 'center',
  },
  botaoGravando: {
    backgroundColor: '#4B5563',
  },
  textoBotaoDestaque: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: 'bold',
  },
  botao: {
    backgroundColor: '#374151',
    paddingVertical: 11,
    borderRadius: 6,
    alignItems: 'center',
  },
  botaoAtivo: {
    backgroundColor: '#4B5563',
  },
  textoBotao: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '600',
  },
  botaoSecundario: {
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#D1D5DB',
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: 'center',
  },
  botaoDesabilitado: {
    opacity: 0.5,
  },
  textoBotaoSecundario: {
    color: '#374151',
    fontSize: 14,
    fontWeight: '500',
  },
  botaoPequeno: {
    backgroundColor: '#374151',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
    marginLeft: 6,
  },
  textoBotaoPequeno: {
    color: '#FFFFFF',
    fontSize: 12,
  },
  textoPadrao: {
    fontSize: 14,
    color: '#374151',
  },
  linhaBotoes: {
    flexDirection: 'row',
    gap: 6,
    width: '100%',
  },
  botaoAcao: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#D1D5DB',
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  textoBotaoAcao: {
    color: '#374151',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
});
