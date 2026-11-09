import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView } from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import LibrasNativeCamera from './modules/libras-native';
import modeloLocal from './assets/modelo_libras_mobile.json';

const JANELA_TEMPORAL_MS = 1100;

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

  let poseValida = false;
  if (Array.isArray(pontosPose) && pontosPose.length >= 9) {
    const p0 = pontosPose[0];
    const p7 = pontosPose[7];
    const p8 = pontosPose[8];
    if (Array.isArray(p0) && Array.isArray(p7) && Array.isArray(p8)) {
      const narizX = Number(p0[0]) * escalaX;
      const narizY = Number(p0[1]) * escalaY;
      const ombroEsqX = Number(p7[0]) * escalaX;
      const ombroEsqY = Number(p7[1]) * escalaY;
      const ombroDirX = Number(p8[0]) * escalaX;
      const ombroDirY = Number(p8[1]) * escalaY;
      if (
        Number.isFinite(narizX) && Number.isFinite(narizY) &&
        Number.isFinite(ombroEsqX) && Number.isFinite(ombroEsqY) &&
        Number.isFinite(ombroDirX) && Number.isFinite(ombroDirY)
      ) {
        const largOmbros = Math.max(Math.hypot(ombroEsqX - ombroDirX, ombroEsqY - ombroDirY), 0.12);
        valores.push(
          limitar((pulso[1] - narizY) / largOmbros, -2.0, 2.5),
          limitar((mao[8][1] - narizY) / largOmbros, -2.0, 2.5),
          limitar(Math.abs(pulso[0] - narizX) / largOmbros, 0.0, 2.5),
          1.0
        );
        poseValida = true;
      }
    }
  }
  if (!poseValida) {
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

function AplicativoPrincipal() {
  const [permissao, pedirPermissao] = useCameraPermissions();
  const [tipoCamera, setTipoCamera] = useState('front');
  const [cameraPronta, setCameraPronta] = useState(false);
  const [tempoInferencia, setTempoInferencia] = useState(0);
  const [letraAtual, setLetraAtual] = useState('-');
  const [confianca, setConfianca] = useState(0);
  const [textoCompleto, setTextoCompleto] = useState('');
  const [capturando, setCapturando] = useState(true);
  const [quadrosBuffer, setQuadrosBuffer] = useState(0);
  const [gravandoGesto, setGravandoGesto] = useState(false);
  const [tempoGestoRestante, setTempoGestoRestante] = useState(0);
  const [contagemRegressiva, setContagemRegressiva] = useState(0);

  const capturandoRef = useRef(true);
  const trocandoCameraRef = useRef(false);
  const ultimaLetraRef = useRef('');
  const ultimoSinalInseridoRef = useRef('');
  const contagemEstavelRef = useRef(0);
  const votacoesModeloRef = useRef([]);
  const gravandoGestoRef = useRef(false);
  const preparandoGestoRef = useRef(false);
  const melhorPredicaoGestoRef = useRef({ rotulo: '', conf: 0, contagem: 0 });
  const historicoLocalRef = useRef([]);
  const falhasMaoLocalRef = useRef(0);
  const estavaEmMovimentoLocalRef = useRef(false);
  const ultimoTempoUiRef = useRef(0);
  const timerPreparacaoRef = useRef(null);
  const timerProgressoRef = useRef(null);
  const timerCameraRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerPreparacaoRef.current) clearInterval(timerPreparacaoRef.current);
      if (timerProgressoRef.current) clearInterval(timerProgressoRef.current);
      if (timerCameraRef.current) clearTimeout(timerCameraRef.current);
    };
  }, []);

  function atualizarTempoUi(valorMs) {
    const agora = Date.now();
    if (agora - ultimoTempoUiRef.current < 200) return;
    ultimoTempoUiRef.current = agora;
    setTempoInferencia((atual) => (atual === valorMs ? atual : valorMs));
  }

  function limparHistoricoPredicao() {
    historicoLocalRef.current = [];
    votacoesModeloRef.current = [];
    falhasMaoLocalRef.current = 0;
    estavaEmMovimentoLocalRef.current = false;
    ultimaLetraRef.current = '';
    ultimoSinalInseridoRef.current = '';
    contagemEstavelRef.current = 0;
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
    try {
      processarQuadroLocal(evento);
    } catch (_erro) {}
  }

  function processarQuadroLocal(evento) {
    const dados = evento?.nativeEvent || {};
    const inicioClassificacao = Date.now();
    const tempoRastreamento = typeof dados.tempo_ms === 'number' ? Math.round(dados.tempo_ms) : 0;

    if (!capturandoRef.current && !gravandoGestoRef.current) {
      atualizarTempoUi(tempoRastreamento);
      return;
    }

    const listaMaos3d = Array.isArray(dados.pontos_maos_3d) ? dados.pontos_maos_3d : [];

    const quadrosMaos = [];
    for (let idx = 0; idx < listaMaos3d.length; idx += 1) {
      const outraMao3d = listaMaos3d.length === 2 ? listaMaos3d[1 - idx] : null;
      const q = calcularVetorQuadro(
        listaMaos3d[idx],
        dados.largura_quadro || 240,
        dados.altura_quadro || 320,
        dados.pontos_pose,
        outraMao3d
      );
      if (q) quadrosMaos.push(q);
    }

    if (quadrosMaos.length === 0) {
      falhasMaoLocalRef.current += 1;
      if (falhasMaoLocalRef.current >= 3 && !gravandoGestoRef.current) {
        setLetraAtual((atual) => (atual === 'Mostre a mao na camera' ? atual : 'Mostre a mao na camera'));
        setConfianca((atual) => (atual === 0 ? atual : 0));
      }
      if (falhasMaoLocalRef.current >= 5) {
        limparHistoricoPredicao();
        setQuadrosBuffer((atual) => (atual === 0 ? atual : 0));
      }
      atualizarTempoUi(tempoRastreamento);
      return;
    }

    falhasMaoLocalRef.current = 0;
    const agoraMs = Date.now();
    const trilhas = historicoLocalRef.current.filter((t) => agoraMs - t.ultimoInstante <= JANELA_TEMPORAL_MS);
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
    setQuadrosBuffer((atual) => (atual === bufferAtual ? atual : bufferAtual));

    const tempoGastoClassificacao = Date.now() - inicioClassificacao;
    atualizarTempoUi(tempoRastreamento + tempoGastoClassificacao);

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
      const confPct = Math.round(mediaConf * 100);
      setLetraAtual((atual) => (atual === predicao ? atual : predicao));
      setConfianca((atual) => (Math.abs(atual - confPct) >= 2 ? confPct : atual));
    } else if (!gravandoGestoRef.current) {
      setLetraAtual((atual) => (atual === 'Analisando sinal...' ? atual : 'Analisando sinal...'));
      setConfianca((atual) => (atual === 0 ? atual : 0));
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

  function alternarCaptura() {
    const novo = !capturando;
    capturandoRef.current = novo;
    limparHistoricoPredicao();
    if (!novo) {
      setQuadrosBuffer(0);
    }
    setCapturando(novo);
  }

  function iniciarGravacaoGesto() {
    if (gravandoGestoRef.current || preparandoGestoRef.current) {
      return;
    }

    preparandoGestoRef.current = true;
    let contagem = 3;
    setContagemRegressiva(contagem);
    setLetraAtual(`Prepare-se (${contagem}s)...`);

    timerPreparacaoRef.current = setInterval(() => {
      contagem -= 1;
      if (contagem > 0) {
        setContagemRegressiva(contagem);
        setLetraAtual(`Prepare-se (${contagem}s)...`);
      } else {
        clearInterval(timerPreparacaoRef.current);
        timerPreparacaoRef.current = null;
        setContagemRegressiva(0);
        preparandoGestoRef.current = false;
        executarGravacaoGesto();
      }
    }, 1000);
  }

  function executarGravacaoGesto() {
    limparHistoricoPredicao();
    gravandoGestoRef.current = true;
    setGravandoGesto(true);
    melhorPredicaoGestoRef.current = { rotulo: '', conf: 0, contagem: 0 };
    setLetraAtual('Gravando sinal...');
    setTempoGestoRestante(2.5);

    const tempoInicio = Date.now();
    const duracaoTotal = 2500;

    timerProgressoRef.current = setInterval(() => {
      const decorrido = Date.now() - tempoInicio;
      const restante = Math.max(0, (duracaoTotal - decorrido) / 1000).toFixed(1);
      setTempoGestoRestante(parseFloat(restante));

      if (decorrido >= duracaoTotal) {
        clearInterval(timerProgressoRef.current);
        timerProgressoRef.current = null;
        gravandoGestoRef.current = false;
        setGravandoGesto(false);
        setTempoGestoRestante(0);

        const melhor = melhorPredicaoGestoRef.current;
        if (melhor.rotulo && melhor.conf >= 0.42) {
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
    if (trocandoCameraRef.current) return;
    trocandoCameraRef.current = true;
    setCameraPronta(false);
    limparHistoricoPredicao();
    setTipoCamera((atual) => (atual === 'front' ? 'back' : 'front'));

    if (timerCameraRef.current) clearTimeout(timerCameraRef.current);
    timerCameraRef.current = setTimeout(() => {
      trocandoCameraRef.current = false;
    }, 400);
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
          <Text style={estilos.descricao}>O aplicativo precisa de acesso a camera para reconhecer Libras.</Text>
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
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>Modo: Local Offline</Text>
            <Text style={estilos.rotuloStatus}>Buffer: {quadrosBuffer}/9 quadros</Text>
          </View>
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>Inferencia: {tempoInferencia} ms</Text>
            <Text style={estilos.rotuloStatus}>Camera: {cameraPronta ? (tipoCamera === 'front' ? 'Frontal' : 'Traseira') : 'Iniciando...'}</Text>
          </View>
        </View>

        <View style={estilos.areaCamera}>
          {LibrasNativeCamera ? (
            <LibrasNativeCamera
              style={estilos.camera}
              active={permissao.granted}
              tracking={permissao.granted}
              facing={tipoCamera}
              onCameraReady={() => {
                setCameraPronta(true);
                trocandoCameraRef.current = false;
              }}
              onCameraError={(evento) => {
                const msg = evento?.nativeEvent?.message || 'Falha na camera local';
                setLetraAtual(`Erro camera: ${msg}`);
              }}
              onLandmarks={receberQuadroLocal}
            />
          ) : (
            <View style={estilos.centralizado}>
              <Text style={estilos.textoPadrao}>Modulo de camera nativo indisponivel.</Text>
            </View>
          )}

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
            <Text style={estilos.valorConfianca}>Confianca: {confianca}% | Tempo: {tempoInferencia} ms</Text>
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
    marginBottom: 8,
    textAlign: 'center',
  },
  descricao: {
    fontSize: 14,
    color: '#4B5563',
    textAlign: 'center',
    marginBottom: 16,
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
    backgroundColor: '#F3F4F6',
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
