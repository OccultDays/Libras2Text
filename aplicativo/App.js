import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, TextInput, ScrollView, Dimensions, Platform } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system/legacy';
import { File } from 'expo-file-system';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Circle, Line } from 'react-native-svg';
import LibrasNativeCamera from './modules/libras-native';
import modeloLocal from './assets/modelo_libras_mobile.json';
import modeloTemporalPiloto from './assets/modelo_libras_temporal_piloto.json';

const modeloTreinadoComDadosReais = modeloLocal?.fonte_dados === 'real';
const modeloExperimentalDisponivel = modeloTemporalPiloto?.fonte_dados === 'real_publico_experimental';

function normalizarMaoParaModeloExperimental(pontos) {
  if (!Array.isArray(pontos) || pontos.length !== 21) return null;
  const mao = pontos.map((ponto) => Array.isArray(ponto) ? ponto.slice(0, 3).map(Number) : []);
  if (mao.some((ponto) => ponto.length !== 3 || ponto.some((valor) => !Number.isFinite(valor)))) return null;
  if (Math.hypot(mao[0][0], mao[0][1]) < 0.02) return null;
  if (mao.filter((ponto) => Math.hypot(ponto[0], ponto[1]) > 0.02).length < 18) return null;
  const pulso = mao[0];
  const relativos = mao.map((ponto) => ponto.map((valor, eixo) => valor - pulso[eixo]));
  const escala = Math.max(...relativos.map((ponto) => Math.hypot(...ponto)));
  if (!Number.isFinite(escala) || escala <= 1e-7) return null;
  return relativos.flatMap((ponto) => ponto.map((valor) => valor / escala));
}

function preverModeloTemporalPiloto(modelo, quadros) {
  const quantidade = modelo?.quantidade_caracteristicas || 0;
  if (!modelo || quadros.length !== modelo.frames || quantidade !== modelo.media.length) {
    return { prediction: '', confidence: 0 };
  }
  const caracteristicas = quadros.flat();
  if (caracteristicas.length !== quantidade) return { prediction: '', confidence: 0 };
  let melhorIndice = -1;
  let melhorPontuacao = -Infinity;
  let segundaPontuacao = -Infinity;
  for (let classe = 0; classe < modelo.rotulos.length; classe += 1) {
    const coeficientes = modelo.coeficientes[classe];
    let pontuacao = modelo.interceptos[classe];
    for (let i = 0; i < quantidade; i += 1) {
      const escalado = (caracteristicas[i] - modelo.media[i]) / modelo.escala[i];
      pontuacao += coeficientes[i] * escalado;
    }
    if (pontuacao > melhorPontuacao) {
      segundaPontuacao = melhorPontuacao;
      melhorPontuacao = pontuacao;
      melhorIndice = classe;
    } else if (pontuacao > segundaPontuacao) {
      segundaPontuacao = pontuacao;
    }
  }
  const margem = Math.max(0, melhorPontuacao - segundaPontuacao);
  const escalaMargem = Math.abs(melhorPontuacao) + Math.abs(segundaPontuacao) + 1e-6;
  return {
    prediction: modelo.rotulos[melhorIndice] || '',
    // Relative margin for display only; this is not a calibrated probability.
    confidence: margem / escalaMargem,
  };
}

function medirMovimentoMao(atual, anterior) {
  if (!Array.isArray(atual) || !Array.isArray(anterior) || atual.length !== 21 || anterior.length !== 21) return 0;
  let total = 0;
  for (let i = 0; i < atual.length; i += 1) {
    total += Math.hypot(atual[i][0] - anterior[i][0], atual[i][1] - anterior[i][1]);
  }
  return total / atual.length;
}

// Conexoes dos 21 pontos da mao (MediaPipe HandLandmarker)
const CONEXOES_MAO = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// Conexoes dos pontos da pose (cabeca + bracos)
// Indices na lista pontos_pose: 0-6 = cabeca, 7-12 = bracos
// 7=ombro_esq, 8=ombro_dir, 9=cotovelo_esq, 10=cotovelo_dir, 11=pulso_esq, 12=pulso_dir
const CONEXOES_POSE = [
  [7,8],
  [7,9],[9,11],
  [8,10],[10,12],
  // Face: nose, eyes, ears and mouth corners (selected Pose landmarks).
  [1,0],[0,2],
  [1,3],[3,5],[5,1],
  [2,4],[4,6],[6,2],
  [0,7],[0,8]
];

function OverlayTracking({ pontosPose, pontosMao, pontosMaos, espelhado }) {
  const largura = Dimensions.get('window').width - 32;
  const altura = 300;

  function converterPonto(ponto) {
    let x = ponto[0] * largura;
    let y = ponto[1] * altura;
    if (espelhado) {
      x = largura - x;
    }
    return { x, y };
  }

  const elementos = [];

  const maosVisuais = pontosMaos && pontosMaos.length
    ? pontosMaos
    : (pontosMao && pontosMao.length === 21 ? [pontosMao] : []);

  maosVisuais.forEach((mao, indiceMao) => {
    if (!mao || mao.length !== 21) return;
    CONEXOES_MAO.forEach((conexao, i) => {
      const p1 = converterPonto(mao[conexao[0]]);
      const p2 = converterPonto(mao[conexao[1]]);
      elementos.push(
        <Line key={`lm${indiceMao}-${i}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
          stroke="#00FF00" strokeWidth={2} />
      );
    });

    mao.forEach((ponto, i) => {
      const p = converterPonto(ponto);
      elementos.push(
        <Circle key={`cm${indiceMao}-${i}`} cx={p.x} cy={p.y} r={4}
          fill="#FF3333" stroke="#00FF00" strokeWidth={1} />
      );
    });
  });

  if (pontosPose && pontosPose.length === 13) {
    CONEXOES_POSE.forEach((conexao, i) => {
      const p1 = converterPonto(pontosPose[conexao[0]]);
      const p2 = converterPonto(pontosPose[conexao[1]]);
      elementos.push(
        <Line key={`lp${i}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
          stroke="#00FF00" strokeWidth={2} opacity={0.7} />
      );
    });

    pontosPose.forEach((ponto, i) => {
      const p = converterPonto(ponto);
      elementos.push(
        <Circle key={`cp${i}`} cx={p.x} cy={p.y} r={3}
          fill="#FFCC00" stroke="#00FF00" strokeWidth={1} />
      );
    });
  }

  if (elementos.length === 0) return null;

  return (
    <View style={estilosOverlay.container} pointerEvents="none">
      <Svg width={largura} height={altura}>
        {elementos}
      </Svg>
    </View>
  );
}

function preverModeloLocal(modelo, caracteristicas) {
  if (!modelo || caracteristicas.length !== modelo.quantidade_caracteristicas) {
    return { prediction: '', confidence: 0 };
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
    prediction: modelo.rotulos[melhorIndice] || '',
    confidence: probabilidades[melhorIndice] || 0
  };
}

const estilosOverlay = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  }
});

function AplicativoPrincipal() {
  const [permissao, pedirPermissao] = useCameraPermissions();
  const [tipoCamera, setTipoCamera] = useState('front');
  const [cameraPronta, setCameraPronta] = useState(Platform.OS !== 'android');
  const [tamanhoFoto, setTamanhoFoto] = useState(undefined);
  const [statusConexao, setStatusConexao] = useState(Platform.OS === 'android'
    ? (modeloExperimentalDisponivel ? 'Modelo experimental local · pausado' : 'Processamento local')
    : 'Desconectado');
  const [latencia, setLatencia] = useState(0);
  const [latenciaServidor, setLatenciaServidor] = useState(0);
  const [letraAtual, setLetraAtual] = useState('-');
  const [confianca, setConfianca] = useState(0);
  const [textoCompleto, setTextoCompleto] = useState('');
  const [capturando, setCapturando] = useState(false);
  const [ipServidor, setIpServidor] = useState('10.0.0.166');
  const [quadrosBuffer, setQuadrosBuffer] = useState(0);
  const [gravandoGesto, setGravandoGesto] = useState(false);
  const [tempoGestoRestante, setTempoGestoRestante] = useState(0);
  const [contagemRegressiva, setContagemRegressiva] = useState(0);
  const [pontosPose, setPontosPose] = useState([]);
  const [pontosMao, setPontosMao] = useState([]);
  const [pontosMaos, setPontosMaos] = useState([]);

  const cameraRef = useRef(null);
  const cameraProntaRef = useRef(true);
  const capturandoRef = useRef(false);
  const trocandoCameraRef = useRef(false);
  const socketRef = useRef(null);
  const timerCapturaRef = useRef(null);
  const enviandoRef = useRef(false);
  const aguardandoRespostaRef = useRef(false);
  const inicioQuadroRef = useRef(0);
  const ultimaLetraRef = useRef('');
  const contagemEstavelRef = useRef(0);
  const votacoesModeloRef = useRef([]);
  const pontosMaoAnterioresRef = useRef(null);
  const sinalExperimentalTravadoRef = useRef('');
  const quadrosSemMovimentoRef = useRef(0);
  const quadrosMovimentoReinicioRef = useRef(0);
  const gravandoGestoRef = useRef(false);
  const preparandoGestoRef = useRef(false);
  const melhorPredicaoGestoRef = useRef({ rotulo: '', conf: 0, contagem: 0 });
  const historicoLocalRef = useRef([]);

  useEffect(() => {
    if (Platform.OS === 'android') {
      setStatusConexao('Processamento local');
      return () => pararCaptura();
    }
    conectarSocket();
    return () => {
      fecharSocket();
      pararCaptura();
    };
  }, [ipServidor]);

  function fecharSocket() {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }

  function conectarSocket() {
    if (Platform.OS === 'android') {
      setStatusConexao('Processamento local');
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
        try {
          const dados = JSON.parse(evento.data);
          if (typeof dados.tempo_ms === 'number') {
            setLatenciaServidor(dados.tempo_ms);
          }
          if (dados.status === 'ok') {
            const predicao = dados.prediction || dados.previsao || '';
            const conf = dados.confidence || dados.confianca || 0;
            const tempoServidor = dados.tempo_ms || 0;
            const bufferLen = dados.quadros_buffer || 0;

            setLetraAtual(predicao || '-');
            setConfianca(Math.round(conf * 100));
            if (!latenciaQuadro) setLatencia(tempoServidor);
            setQuadrosBuffer(bufferLen);

            if (dados.pontos_pose) setPontosPose(dados.pontos_pose);
            if (dados.pontos_mao) setPontosMao(dados.pontos_mao);
            if (dados.pontos_maos) setPontosMaos(dados.pontos_maos);

            if (gravandoGestoRef.current && predicao && conf > melhorPredicaoGestoRef.current.conf) {
              melhorPredicaoGestoRef.current = { rotulo: predicao, conf: conf };
            }

            if (!gravandoGestoRef.current && predicao && conf >= 0.50) {
              if (predicao === ultimaLetraRef.current) {
                contagemEstavelRef.current += 1;
                if (contagemEstavelRef.current === 2) {
                  adicionarAoTexto(predicao);
                }
              } else {
                ultimaLetraRef.current = predicao;
                contagemEstavelRef.current = 1;
              }
            }
          } else if (
            dados.status === 'nenhum_sinal_detectado' ||
            dados.status === 'nenhuma_mao_detectada' ||
            dados.status === 'nenhum_corpo_detectado'
          ) {
            setLetraAtual('Posicione-se');
            setConfianca(0);
            setPontosPose([]);
            setPontosMao([]);
            setPontosMaos([]);
          }
        } catch (erro) {
          inicioQuadroRef.current = 0;
          // Ignora falha de parse
        }

        // Continue immediately when the previous inference returns. A fixed
        // interval could add up to 120 ms of idle time between frames.
        if (socketRef.current === ws) agendarProximoQuadro();
      };

      ws.onerror = () => {
        aguardandoRespostaRef.current = false;
        inicioQuadroRef.current = 0;
        setStatusConexao('Erro de conexao');
      };

      ws.onclose = () => {
        aguardandoRespostaRef.current = false;
        inicioQuadroRef.current = 0;
        setStatusConexao('Desconectado');
      };

      socketRef.current = ws;
    } catch (erro) {
      setStatusConexao('Falha ao abrir socket');
    }
  }

  function adicionarAoTexto(sinal) {
    if (
      !sinal ||
      sinal === '-' ||
      sinal === 'Posicione-se' ||
      sinal === 'Mostre a mao inteira' ||
      sinal === 'Estabilizando previsão...' ||
      sinal.startsWith('Preparando janela') ||
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
        return anterior.endsWith(' ') ? anterior + sinal : anterior + ' ' + sinal;
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
    const vetor = dados.vetor || [];
    const inicioClassificacao = Date.now();
    const tempoRastreamento = typeof dados.tempo_ms === 'number' ? dados.tempo_ms : 0;
    setLatencia(tempoRastreamento);
    setLatenciaServidor(0);
    if (dados.pontos_pose) setPontosPose(dados.pontos_pose);
    if (dados.pontos_mao) setPontosMao(dados.pontos_mao);
    if (dados.pontos_maos) setPontosMaos(dados.pontos_maos);

    const quadroExperimental = modeloExperimentalDisponivel
      ? normalizarMaoParaModeloExperimental(dados.pontos_mao_3d)
      : null;
    if (modeloExperimentalDisponivel ? !quadroExperimental : (vetor.length !== 102 || !dados.tem_pose || !dados.tem_mao)) {
      historicoLocalRef.current = [];
      ultimaLetraRef.current = '';
      contagemEstavelRef.current = 0;
      votacoesModeloRef.current = [];
      pontosMaoAnterioresRef.current = null;
      sinalExperimentalTravadoRef.current = '';
      quadrosSemMovimentoRef.current = 0;
      quadrosMovimentoReinicioRef.current = 0;
      setLetraAtual(modeloExperimentalDisponivel ? 'Mostre a mao inteira' : 'Mantenha mao e corpo visiveis');
      setConfianca(0);
      setQuadrosBuffer(0);
      setStatusConexao(dados.diagnostico || 'Aguardando mao e parte superior do corpo');
      return;
    }

    if (modeloExperimentalDisponivel) {
      const historico = historicoLocalRef.current;
      const pontosBrutosMao = dados.pontos_mao_3d;
      const movimento = medirMovimentoMao(pontosBrutosMao, pontosMaoAnterioresRef.current);
      pontosMaoAnterioresRef.current = pontosBrutosMao;

      if (sinalExperimentalTravadoRef.current) {
        if (movimento < 0.004) {
          quadrosSemMovimentoRef.current += 1;
          quadrosMovimentoReinicioRef.current = 0;
        } else if (quadrosSemMovimentoRef.current >= 8 && movimento >= 0.012) {
          quadrosMovimentoReinicioRef.current += 1;
        } else {
          quadrosMovimentoReinicioRef.current = 0;
          if (movimento >= 0.004) quadrosSemMovimentoRef.current = 0;
        }

        if (quadrosMovimentoReinicioRef.current >= 2) {
          sinalExperimentalTravadoRef.current = '';
          quadrosSemMovimentoRef.current = 0;
          quadrosMovimentoReinicioRef.current = 0;
          historico.length = 0;
          votacoesModeloRef.current = [];
          ultimaLetraRef.current = '';
          contagemEstavelRef.current = 0;
        } else {
          setLetraAtual(sinalExperimentalTravadoRef.current);
          setStatusConexao('Sinal registrado · aguarde uma pausa e faça o próximo');
          return;
        }
      }

      historico.push(quadroExperimental);
      if (historico.length > modeloTemporalPiloto.frames) historico.shift();
      setQuadrosBuffer(historico.length);
      setStatusConexao('Modelo experimental local · 20 sinais · pontuação não calibrada');
      if (historico.length < modeloTemporalPiloto.frames) {
        setLetraAtual('Preparando janela experimental');
        setConfianca(0);
        return;
      }
      const resultado = preverModeloTemporalPiloto(modeloTemporalPiloto, historico);
      setLatencia(tempoRastreamento + Date.now() - inicioClassificacao);
      const votacoes = votacoesModeloRef.current;
      votacoes.push(resultado);
      if (votacoes.length > 7) votacoes.shift();
      const contagens = new Map();
      for (const voto of votacoes) {
        if (!voto.prediction) continue;
        const anterior = contagens.get(voto.prediction) || { quantidade: 0, pontuacao: 0 };
        anterior.quantidade += 1;
        anterior.pontuacao += voto.confidence;
        contagens.set(voto.prediction, anterior);
      }
      const [rotuloVotado, consenso] = [...contagens.entries()].sort((a, b) => b[1].quantidade - a[1].quantidade)[0] || ['', { quantidade: 0, pontuacao: 0 }];
      const predicao = consenso.quantidade >= 5 ? rotuloVotado : '';
      const conf = predicao ? consenso.pontuacao / consenso.quantidade : 0;
      if (predicao) {
        setLetraAtual(predicao);
        setConfianca(Math.round(conf * 100));
      } else if (!ultimaLetraRef.current) {
        setLetraAtual('Estabilizando previsão...');
        setConfianca(0);
      }

      if (gravandoGestoRef.current && predicao && conf >= 0.70) {
        const anterior = melhorPredicaoGestoRef.current;
        const contagem = anterior.rotulo === predicao ? anterior.contagem + 1 : 1;
        melhorPredicaoGestoRef.current = {
          rotulo: predicao,
          conf: anterior.rotulo === predicao ? Math.max(anterior.conf, conf) : conf,
          contagem
        };
      }
      if (!gravandoGestoRef.current && predicao && conf >= 0.70) {
        if (predicao === ultimaLetraRef.current) {
          contagemEstavelRef.current += 1;
        } else {
          ultimaLetraRef.current = predicao;
          contagemEstavelRef.current = 1;
        }
        if (contagemEstavelRef.current >= 3 && !sinalExperimentalTravadoRef.current) {
          adicionarAoTexto(predicao);
          sinalExperimentalTravadoRef.current = predicao;
          quadrosSemMovimentoRef.current = 0;
          quadrosMovimentoReinicioRef.current = 0;
        }
      } else if (!gravandoGestoRef.current) {
        ultimaLetraRef.current = '';
        contagemEstavelRef.current = 0;
      }
      return;
    }

    // The checked-in mobile artifact was trained only with generated geometric
    // poses. It is useful for UI demos, but cannot translate real Libras.
    if (!modeloTreinadoComDadosReais) {
      historicoLocalRef.current = [];
      ultimaLetraRef.current = '';
      contagemEstavelRef.current = 0;
      setLetraAtual('Treino real necessario');
      setConfianca(0);
      setQuadrosBuffer(0);
      setStatusConexao('Rastreamento local · modelo demonstrativo');
      return;
    }

    const historico = historicoLocalRef.current;
    historico.push(vetor);
    if (historico.length > 10) historico.shift();
    const inicial = historico[0];
    const caracteristicas = vetor.concat(vetor.map((valor, indice) => valor - inicial[indice]));
    const resultado = preverModeloLocal(modeloLocal, caracteristicas);
    setLatencia(tempoRastreamento + Date.now() - inicioClassificacao);
    const predicao = resultado.prediction;
    const conf = resultado.confidence;
    setQuadrosBuffer(historico.length);
    setLetraAtual(predicao || '-');
    setConfianca(Math.round(conf * 100));

    if (gravandoGestoRef.current && predicao && conf >= 0.70) {
      const anterior = melhorPredicaoGestoRef.current;
      const contagem = anterior.rotulo === predicao ? anterior.contagem + 1 : 1;
      melhorPredicaoGestoRef.current = {
        rotulo: predicao,
        conf: anterior.rotulo === predicao ? Math.max(anterior.conf, conf) : conf,
        contagem
      };
    }
    if (!gravandoGestoRef.current && predicao && conf >= 0.70) {
      if (predicao === ultimaLetraRef.current) {
        contagemEstavelRef.current += 1;
        if (contagemEstavelRef.current === 6) adicionarAoTexto(predicao);
      } else {
        ultimaLetraRef.current = predicao;
        contagemEstavelRef.current = 1;
      }
    } else if (!gravandoGestoRef.current) {
      ultimaLetraRef.current = '';
      contagemEstavelRef.current = 0;
    }
  }

  async function capturarEEnviar() {
    if (Platform.OS === 'android') return;
    if (
      (!capturandoRef.current && !gravandoGestoRef.current) ||
      enviandoRef.current ||
      aguardandoRespostaRef.current ||
      !cameraRef.current ||
      trocandoCameraRef.current
    ) {
      return;
    }

    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    enviandoRef.current = true;
    inicioQuadroRef.current = Date.now();
    let falhaNaCaptura = false;
    try {
      const foto = await cameraRef.current.takePictureAsync({
        quality: 0.25,
        shutterSound: false
      });

      if (foto && foto.uri && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const bytesImagem = await new File(foto.uri).arrayBuffer();
        // Backpressure: do not enqueue stale camera frames on the socket.
        // Wait for this frame's inference response before sending another.
        aguardandoRespostaRef.current = true;
        socketRef.current.send(bytesImagem);
      }

      if (foto && foto.uri) {
        // Keep cache cleanup out of the critical path for the next frame.
        FileSystem.deleteAsync(foto.uri, { idempotent: true }).catch(() => {});
      }
    } catch (erro) {
      falhaNaCaptura = true;
      aguardandoRespostaRef.current = false;
      inicioQuadroRef.current = 0;
      // Ignora erro no quadro
    } finally {
      enviandoRef.current = false;
      if (!aguardandoRespostaRef.current) {
        agendarProximoQuadro(falhaNaCaptura ? 100 : 0);
      }
    }
  }

  useEffect(() => {
    if (capturando) {
      capturarEEnviar();
    } else {
      pararCaptura();
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
    inicioQuadroRef.current = 0;
  }

  function alternarCaptura() {
    const novo = !capturando;
    capturandoRef.current = novo;
    if (!novo) {
      historicoLocalRef.current = [];
      votacoesModeloRef.current = [];
      pontosMaoAnterioresRef.current = null;
      sinalExperimentalTravadoRef.current = '';
      quadrosSemMovimentoRef.current = 0;
      quadrosMovimentoReinicioRef.current = 0;
      ultimaLetraRef.current = '';
      contagemEstavelRef.current = 0;
    }
    if (Platform.OS === 'android') {
      setStatusConexao(novo
        ? ((modeloTreinadoComDadosReais || modeloExperimentalDisponivel) ? 'Rastreamento e traducao locais' : 'Rastreamento local · modelo demonstrativo')
        : (modeloExperimentalDisponivel ? 'Modelo experimental local · pausado' : 'Processamento local'));
    }
    setCapturando(novo);
  }

  function iniciarGravacaoGesto() {
    if (gravandoGestoRef.current || preparandoGestoRef.current) {
      return;
    }

    if (Platform.OS === 'android' && !modeloTreinadoComDadosReais && !modeloExperimentalDisponivel) {
      setLetraAtual('Treino real necessario');
      setConfianca(0);
      setStatusConexao('Rastreamento local · modelo demonstrativo');
      return;
    }

    if (Platform.OS !== 'android' && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)) {
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
    historicoLocalRef.current = [];
    votacoesModeloRef.current = [];
    pontosMaoAnterioresRef.current = null;
    sinalExperimentalTravadoRef.current = '';
    quadrosSemMovimentoRef.current = 0;
    quadrosMovimentoReinicioRef.current = 0;
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ tipo: "limpar_buffer" }));
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
      const restante = Math.max(0, ((duracaoTotal - decorrido) / 1000)).toFixed(1);
      setTempoGestoRestante(parseFloat(restante));

      if (decorrido >= duracaoTotal) {
        clearInterval(timerProgresso);
        gravandoGestoRef.current = false;
        setGravandoGesto(false);
        setTempoGestoRestante(0);

        const melhor = melhorPredicaoGestoRef.current;
        const gestoEstavel = Platform.OS !== 'android' || melhor.contagem >= 6;
        const confiancaMinima = Platform.OS === 'android' ? 0.70 : 0.30;
        if (melhor.rotulo && melhor.conf >= confiancaMinima && gestoEstavel) {
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
    ultimaLetraRef.current = '';
    contagemEstavelRef.current = 0;
    historicoLocalRef.current = [];
    votacoesModeloRef.current = [];
    pontosMaoAnterioresRef.current = null;
    sinalExperimentalTravadoRef.current = '';
    quadrosSemMovimentoRef.current = 0;
    quadrosMovimentoReinicioRef.current = 0;
  }

  function adicionarEspaco() {
    setTextoCompleto((anterior) => anterior + ' ');
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
          {Platform.OS !== 'android' && (
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
            <Text style={estilos.rotuloStatus}>
              {Platform.OS === 'android'
                ? `Quadros locais: ${quadrosBuffer}/${modeloExperimentalDisponivel ? modeloTemporalPiloto.frames : 10}`
                : `Buffer: ${quadrosBuffer}/10 quadros`}
            </Text>
          </View>
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>{Platform.OS === 'android' ? `Inferencia local: ${latencia} ms` : `Total: ${latencia} ms`}</Text>
            {Platform.OS !== 'android' && <Text style={estilos.rotuloStatus}>Servidor: {latenciaServidor} ms</Text>}
          </View>
        </View>

        <View style={estilos.areaCamera}>
          {Platform.OS === 'android' ? (
            <LibrasNativeCamera
              key={tipoCamera}
              style={estilos.camera}
              active={permissao.granted}
              tracking={capturando || gravandoGesto}
              facing={tipoCamera}
              onCameraReady={() => {
                cameraProntaRef.current = true;
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
              animateShutter={false}
              flash="off"
              mute={true}
              onCameraReady={() => {
                cameraProntaRef.current = true;
                setCameraPronta(true);
                cameraRef.current?.getAvailablePictureSizesAsync()
                  .then((tamanhos) => {
                    const resolucoes = tamanhos
                      .map((tamanho) => {
                        const [largura, altura] = tamanho.split('x').map(Number);
                        return { tamanho, largura, altura, area: largura * altura };
                      })
                      .filter((item) => item.largura && item.altura)
                      .sort((a, b) => a.area - b.area);
                    const recomendada = resolucoes.find((item) =>
                      item.largura >= 640 && item.altura >= 480
                    ) || resolucoes.find((item) => Math.max(item.largura, item.altura) >= 640)
                      || resolucoes[0];
                    if (recomendada) setTamanhoFoto(recomendada.tamanho);
                  })
                  .catch(() => {});
              }}
              pictureSize={tamanhoFoto}
              onMountError={() => {
                cameraProntaRef.current = false;
                setCameraPronta(false);
                trocandoCameraRef.current = false;
              }}
            />
          )}
          <OverlayTracking
            pontosPose={pontosPose}
            pontosMao={pontosMao}
            pontosMaos={pontosMaos}
            espelhado={tipoCamera === 'front'}
          />
          <TouchableOpacity
            style={estilos.etiquetaCamera}
            onPress={alternarCamera}
            activeOpacity={0.7}
            disabled={!cameraPronta}
          >
            <Text style={estilos.textoEtiquetaCamera}>
              {!cameraPronta
                ? "Iniciando camera..."
                : (tipoCamera === 'front' ? "Camera: Frontal" : "Camera: Traseira")}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={estilos.painelResultado}>
          <View style={estilos.caixaPrevisao}>
            <Text style={estilos.rotuloPainel}>Sinal Detectado:</Text>
            <Text style={estilos.valorPrevisao}>{letraAtual}</Text>
            <Text style={estilos.valorConfianca}>
              {modeloExperimentalDisponivel ? 'Pontuação experimental (não calibrada)' : 'Confianca'}: {confianca}%
            </Text>
          </View>

          <View style={estilos.caixaTexto}>
            <Text style={estilos.rotuloPainel}>Texto Traduzido:</Text>
            <Text style={estilos.valorTexto}>{textoCompleto || "(nenhum sinal gravado)"}</Text>
          </View>
        </View>

        <View style={estilos.painelBotoes}>
          <TouchableOpacity
            style={[
              estilos.botaoDestaque,
              contagemRegressiva > 0 || gravandoGesto ? estilos.botaoGravando : null
            ]}
            onPress={iniciarGravacaoGesto}
            disabled={contagemRegressiva > 0 || gravandoGesto || !cameraPronta}
          >
            <Text style={estilos.textoBotaoDestaque}>
              {contagemRegressiva > 0
                ? `Prepare-se (${contagemRegressiva}s)...`
                : gravandoGesto
                ? `Gravando Gesto... (${tempoGestoRestante}s)`
                : "Gravar Gesto Longo (2.5s)"}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[estilos.botao, capturando && !gravandoGesto ? estilos.botaoAtivo : null]}
            onPress={alternarCaptura}
            disabled={gravandoGesto}
          >
            <Text style={estilos.textoBotao}>
              {capturando ? "Pausar Traducao Continua" : "Iniciar Traducao Continua"}
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
                ? "Trocando Camera..."
                : (tipoCamera === 'front' ? "Trocar para Camera Traseira" : "Trocar para Camera Frontal")}
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
    height: 300,
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
