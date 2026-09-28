import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, TextInput, SafeAreaView, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

export default function AplicativoPrincipal() {
  const [permissao, pedirPermissao] = useCameraPermissions();
  const [tipoCamera, setTipoCamera] = useState('front');
  const [cameraPronta, setCameraPronta] = useState(false);
  const [statusConexao, setStatusConexao] = useState('Desconectado');
  const [latencia, setLatencia] = useState(0);
  const [letraAtual, setLetraAtual] = useState('-');
  const [confianca, setConfianca] = useState(0);
  const [textoCompleto, setTextoCompleto] = useState('');
  const [capturando, setCapturando] = useState(false);
  const [ipServidor, setIpServidor] = useState('192.168.0.12');
  const [quadrosBuffer, setQuadrosBuffer] = useState(0);
  const [gravandoGesto, setGravandoGesto] = useState(false);
  const [tempoGestoRestante, setTempoGestoRestante] = useState(0);
  const [contagemRegressiva, setContagemRegressiva] = useState(0);

  const cameraRef = useRef(null);
  const cameraProntaRef = useRef(false);
  const trocandoCameraRef = useRef(false);
  const socketRef = useRef(null);
  const intervaloRef = useRef(null);
  const enviandoRef = useRef(false);
  const ultimaLetraRef = useRef('');
  const contagemEstavelRef = useRef(0);
  const gravandoGestoRef = useRef(false);
  const preparandoGestoRef = useRef(false);
  const melhorPredicaoGestoRef = useRef({ rotulo: '', conf: 0 });

  useEffect(() => {
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
    fecharSocket();
    setStatusConexao('Conectando...');

    try {
      const url = `ws://${ipServidor}:8000/ws`;
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setStatusConexao('Conectado');
      };

      ws.onmessage = (evento) => {
        try {
          const dados = JSON.parse(evento.data);
          if (dados.status === 'ok') {
            const predicao = dados.prediction || dados.previsao || '';
            const conf = dados.confidence || dados.confianca || 0;
            const tempoServidor = dados.tempo_ms || 0;
            const bufferLen = dados.quadros_buffer || 0;

            setLetraAtual(predicao || '-');
            setConfianca(Math.round(conf * 100));
            setLatencia(tempoServidor);
            setQuadrosBuffer(bufferLen);

            if (gravandoGestoRef.current && predicao && conf > melhorPredicaoGestoRef.current.conf) {
              melhorPredicaoGestoRef.current = { rotulo: predicao, conf: conf };
            }

            if (!gravandoGestoRef.current && predicao && conf >= 0.65) {
              if (predicao === ultimaLetraRef.current) {
                contagemEstavelRef.current += 1;
                if (contagemEstavelRef.current === 3) {
                  setTextoCompleto((anterior) => {
                    if (!anterior) return predicao;
                    if (predicao.length > 1 || anterior.slice(-1) === ' ') {
                      return anterior + (anterior.endsWith(' ') ? '' : ' ') + predicao;
                    }
                    return anterior + predicao;
                  });
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
          }
        } catch (erro) {
          // Ignora falha de parse
        }
      };

      ws.onerror = () => {
        setStatusConexao('Erro de conexao');
      };

      ws.onclose = () => {
        setStatusConexao('Desconectado');
      };

      socketRef.current = ws;
    } catch (erro) {
      setStatusConexao('Falha ao abrir socket');
    }
  }

  async function capturarEEnviar() {
    if (
      !capturando ||
      enviandoRef.current ||
      !cameraRef.current ||
      !cameraProntaRef.current ||
      trocandoCameraRef.current
    ) {
      return;
    }

    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    enviandoRef.current = true;
    try {
      const foto = await cameraRef.current.takePictureAsync({
        quality: 0.3,
        base64: true,
        shutterSound: false
      });

      if (foto && foto.base64 && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(foto.base64);
      }

      if (foto && foto.uri) {
        await FileSystem.deleteAsync(foto.uri, { idempotent: true }).catch(() => {});
      }
    } catch (erro) {
      // Ignora erro no frame
    } finally {
      enviandoRef.current = false;
    }
  }

  useEffect(() => {
    if (capturando) {
      intervaloRef.current = setInterval(() => {
        capturarEEnviar();
      }, 180);
    } else {
      pararCaptura();
    }
    return () => pararCaptura();
  }, [capturando]);

  function pararCaptura() {
    if (intervaloRef.current) {
      clearInterval(intervaloRef.current);
      intervaloRef.current = null;
    }
    enviandoRef.current = false;
  }

  function alternarCaptura() {
    setCapturando(!capturando);
  }

  function iniciarGravacaoGesto() {
    if (gravandoGestoRef.current || preparandoGestoRef.current || !cameraProntaRef.current) {
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
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ tipo: "limpar_buffer" }));
    }

    gravandoGestoRef.current = true;
    setGravandoGesto(true);
    melhorPredicaoGestoRef.current = { rotulo: '', conf: 0 };
    setLetraAtual('Gravando...');
    setTempoGestoRestante(2.5);

    const capturaJaAtiva = capturando;
    if (!capturaJaAtiva) {
      setCapturando(true);
    }

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

        if (!capturaJaAtiva) {
          setCapturando(false);
        }

        const melhor = melhorPredicaoGestoRef.current;
        if (melhor.rotulo && melhor.conf >= 0.40) {
          setTextoCompleto((anterior) => {
            if (!anterior) return melhor.rotulo;
            return anterior + (anterior.endsWith(' ') ? '' : ' ') + melhor.rotulo;
          });
          setLetraAtual(melhor.rotulo);
          setConfianca(Math.round(melhor.conf * 100));
        } else {
          setLetraAtual('-');
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
    cameraProntaRef.current = false;
    setCameraPronta(false);

    setTipoCamera((atual) => (atual === 'front' ? 'back' : 'front'));

    setTimeout(() => {
      trocandoCameraRef.current = false;
    }, 600);
  }

  function limparTexto() {
    setTextoCompleto('');
    setLetraAtual('-');
    setConfianca(0);
    ultimaLetraRef.current = '';
    contagemEstavelRef.current = 0;
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
          <View style={estilos.linhaStatus}>
            <Text style={estilos.rotuloStatus}>Status: {statusConexao}</Text>
            <Text style={estilos.rotuloStatus}>Buffer: {quadrosBuffer}/10 quadros</Text>
            <Text style={estilos.rotuloStatus}>Latencia: {latencia} ms</Text>
          </View>
        </View>

        <View style={estilos.areaCamera}>
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
            }}
            onMountError={() => {
              cameraProntaRef.current = false;
              setCameraPronta(false);
              trocandoCameraRef.current = false;
            }}
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
            <Text style={estilos.valorConfianca}>Confianca: {confianca}%</Text>
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

          <TouchableOpacity style={estilos.botaoSecundario} onPress={adicionarEspaco}>
            <Text style={estilos.textoBotaoSecundario}>Espaco</Text>
          </TouchableOpacity>

          <TouchableOpacity style={estilos.botaoSecundario} onPress={limparTexto}>
            <Text style={estilos.textoBotaoSecundario}>Limpar</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
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
});
