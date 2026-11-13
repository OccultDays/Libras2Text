package expo.modules.librasnative

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class LibrasNativeCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  private val conteinerCamera = FrameLayout(context)
  private val previewView = TextureView(context)
  private val camadaRastreamento = CamadaRastreamentoMao(context)
  private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private val executorPose: ExecutorService = Executors.newSingleThreadExecutor()
  private val poseOcupada = AtomicBoolean(false)

  private var cameraProvider: ProcessCameraProvider? = null
  private var previewUseCase: Preview? = null
  private var analysisUseCase: ImageAnalysis? = null
  private var handLandmarker: HandLandmarker? = null
  private var poseLandmarker: PoseLandmarker? = null
  private var tentarGpuMao = true
  private var ultimoTimestampMaoMs = 0L
  private var ultimoTimestampPoseMs = 0L

  @Volatile private var active = false
  private var readyNotified = false
  @Volatile private var tracking = false
  @Volatile private var facing = "front"
  private var lastError = ""
  private var layoutListener: OnLayoutChangeListener? = null
  private var bindGeneration = 0
  private var bufferLargura = 480
  private var bufferAltura = 640
  @Volatile private var instanteUltimaPoseMs = 0L
  @Volatile private var ultimaPose: List<List<Double>> = emptyList()

  private val onLandmarks by EventDispatcher()
  private val onCameraReady by EventDispatcher()
  private val onCameraError by EventDispatcher()

  init {
    val paramsFrame = FrameLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT,
      ViewGroup.LayoutParams.MATCH_PARENT
    )
    conteinerCamera.addView(previewView, paramsFrame)
    conteinerCamera.addView(
      camadaRastreamento,
      FrameLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.MATCH_PARENT
      )
    )
    addView(
      conteinerCamera,
      LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
    )

    previewView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
      override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        atualizarTransformacaoPrevia()
        if (active) startCamera()
      }
      override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        atualizarTransformacaoPrevia()
      }
      override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
        notifyCameraReady()
      }
      override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
    }
  }

  private fun postarNaInterface(bloco: () -> Unit) {
    if (Looper.myLooper() == Looper.getMainLooper()) {
      if (isAttachedToWindow) bloco()
    } else {
      post {
        if (isAttachedToWindow) bloco()
      }
    }
  }

  private fun atualizarTransformacaoPrevia() {
    val larguraVista = previewView.width.toFloat()
    val alturaVista = previewView.height.toFloat()
    val largBuf = bufferLargura.toFloat()
    val altBuf = bufferAltura.toFloat()
    if (larguraVista <= 0f || alturaVista <= 0f || largBuf <= 0f || altBuf <= 0f) return

    val escala = max(larguraVista / largBuf, alturaVista / altBuf)
    val escalaX = (largBuf * escala) / larguraVista
    val escalaY = (altBuf * escala) / alturaVista
    val matriz = Matrix().apply {
      setScale(escalaX, escalaY, larguraVista / 2f, alturaVista / 2f)
    }
    previewView.setTransform(matriz)
  }

  fun setActive(value: Boolean) {
    if (active == value) return
    active = value
    if (value) startCamera() else stopCamera()
  }

  fun setTracking(value: Boolean) {
    tracking = value
    if (!value) {
      camadaRastreamento.limpar()
    }
  }

  fun setFacing(value: String) {
    val normalized = if (value == "back") "back" else "front"
    if (facing == normalized) return
    facing = normalized
    readyNotified = false
    camadaRastreamento.limpar()
    atualizarTransformacaoPrevia()
    if (active) startCamera()
  }

  private fun startCamera() {
    val generation = ++bindGeneration
    readyNotified = false
    if (previewView.width == 0 || previewView.height == 0 || !previewView.isAvailable) {
      if (layoutListener == null) {
        val listener = OnLayoutChangeListener { view, _, _, _, _, _, _, _, _ ->
          if (view.width > 0 && view.height > 0) {
            layoutListener?.let(view::removeOnLayoutChangeListener)
            layoutListener = null
            atualizarTransformacaoPrevia()
            if (active) startCamera()
          }
        }
        layoutListener = listener
        previewView.addOnLayoutChangeListener(listener)
      }
      return
    }

    val activity = appContext.activityProvider?.currentActivity
    if (activity == null) {
      reportError("Atividade Android indisponivel para abrir a camera.")
      return
    }
    if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      reportError("Permissao da camera ainda nao concedida.")
      return
    }

    val providerFuture = ProcessCameraProvider.getInstance(context)
    providerFuture.addListener({
      try {
        if (!active || generation != bindGeneration || !previewView.isAttachedToWindow) return@addListener
        val provider = providerFuture.get()
        cameraProvider = provider
        val displayRotation = previewView.display?.rotation ?: Surface.ROTATION_0
        val executorPrincipal = ContextCompat.getMainExecutor(context)

        val preview = Preview.Builder()
          .setTargetRotation(displayRotation)
          .build()
        preview.setSurfaceProvider(executorPrincipal) { request ->
          val texture = previewView.surfaceTexture
          if (texture == null || !previewView.isAvailable) {
            request.willNotProvideSurface()
            return@setSurfaceProvider
          }
          request.setTransformationInfoListener(executorPrincipal) { info ->
            val rot = ((info.rotationDegrees % 360) + 360) % 360
            if (rot == 90 || rot == 270) {
              bufferLargura = request.resolution.height
              bufferAltura = request.resolution.width
            } else {
              bufferLargura = request.resolution.width
              bufferAltura = request.resolution.height
            }
            atualizarTransformacaoPrevia()
          }
          texture.setDefaultBufferSize(request.resolution.width, request.resolution.height)
          val surface = Surface(texture)
          request.provideSurface(surface, executorPrincipal) {
            surface.release()
          }
        }

        val analysis = ImageAnalysis.Builder()
          .setTargetResolution(Size(240, 320))
          .setTargetRotation(displayRotation)
          .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
          .setOutputImageRotationEnabled(true)
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .build()
        analysis.setAnalyzer(analyzerExecutor, ::analyzeFrame)

        val selector = if (facing == "front") CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        val owner = activity as? LifecycleOwner
          ?: throw IllegalStateException("A atividade Android nao controla o ciclo de vida da camera.")
        provider.unbindAll()
        if (generation != bindGeneration || !active) return@addListener
        provider.bindToLifecycle(owner, selector, preview, analysis)
        previewUseCase = preview
        analysisUseCase = analysis
        lastError = ""
      } catch (error: Throwable) {
        val msg = error.cause?.message ?: error.message ?: "Nao foi possivel iniciar a camera."
        reportError(msg)
      }
    }, ContextCompat.getMainExecutor(context))
  }

  private fun notifyCameraReady() {
    postarNaInterface {
      if (!active || readyNotified) return@postarNaInterface
      readyNotified = true
      onCameraReady(mapOf("ready" to true))
    }
  }

  private fun stopCamera() {
    bindGeneration++
    readyNotified = false
    camadaRastreamento.limpar()
    val provider = cameraProvider
    if (provider != null) {
      try {
        val preview = previewUseCase
        val analysis = analysisUseCase
        if (preview != null) provider.unbind(preview)
        if (analysis != null) provider.unbind(analysis)
      } catch (_: Throwable) {}
    }
    previewUseCase = null
    analysisUseCase = null
    if (!analyzerExecutor.isShutdown) {
      try {
        analyzerExecutor.execute {
          try { handLandmarker?.close() } catch (_: Throwable) {}
          handLandmarker = null
          ultimoTimestampMaoMs = 0L
        }
      } catch (_: Throwable) {}
    }
    if (!executorPose.isShutdown) {
      try {
        executorPose.execute {
          try { poseLandmarker?.close() } catch (_: Throwable) {}
          poseLandmarker = null
          ultimoTimestampPoseMs = 0L
          ultimaPose = emptyList()
          instanteUltimaPoseMs = 0L
        }
      } catch (_: Throwable) {}
    }
  }

  private fun criarDetectorMaos(usarGpu: Boolean): HandLandmarker {
    val baseOptions = BaseOptions.builder()
      .setModelAssetPath("marcador_maos.task")
      .setDelegate(if (usarGpu) Delegate.GPU else Delegate.CPU)
      .build()
    val opcoes = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(baseOptions)
      .setNumHands(2)
      .setMinHandDetectionConfidence(0.25f)
      .setMinHandPresenceConfidence(0.25f)
      .setMinTrackingConfidence(0.25f)
      .setRunningMode(RunningMode.VIDEO)
      .build()
    return HandLandmarker.createFromOptions(context, opcoes)
  }

  private fun garantirDetectorMaos(): HandLandmarker {
    var existente = handLandmarker
    if (existente != null) return existente

    ultimoTimestampMaoMs = 0L
    existente = if (tentarGpuMao) {
      try {
        criarDetectorMaos(true)
      } catch (_: Throwable) {
        tentarGpuMao = false
        criarDetectorMaos(false)
      }
    } else {
      criarDetectorMaos(false)
    }
    handLandmarker = existente
    return existente
  }

  private fun detectarMaosVideo(mpImage: MPImage): HandLandmarkerResult {
    val detector = garantirDetectorMaos()
    var ts = SystemClock.elapsedRealtime()
    if (ts <= ultimoTimestampMaoMs) ts = ultimoTimestampMaoMs + 1L
    ultimoTimestampMaoMs = ts

    return try {
      detector.detectForVideo(mpImage, ts)
    } catch (erro: Throwable) {
      if (!tentarGpuMao) throw erro
      try { handLandmarker?.close() } catch (_: Throwable) {}
      handLandmarker = null
      tentarGpuMao = false
      val detectorCpu = garantirDetectorMaos()
      var tsCpu = SystemClock.elapsedRealtime()
      if (tsCpu <= ultimoTimestampMaoMs) tsCpu = ultimoTimestampMaoMs + 1L
      ultimoTimestampMaoMs = tsCpu
      detectorCpu.detectForVideo(mpImage, tsCpu)
    }
  }

  private fun agendarPoseAssincrona(origem: Bitmap) {
    if (executorPose.isShutdown || !poseOcupada.compareAndSet(false, true)) return
    val copia = try {
      origem.copy(Bitmap.Config.ARGB_8888, false)
    } catch (_: Throwable) {
      null
    }
    if (copia == null) {
      poseOcupada.set(false)
      return
    }
    try {
      executorPose.execute { processarPoseAssincrona(copia) }
    } catch (_: Throwable) {
      if (!copia.isRecycled) copia.recycle()
      poseOcupada.set(false)
    }
  }

  private fun processarPoseAssincrona(bitmapPose: Bitmap) {
    var imagemPose: MPImage? = null
    try {
      if (!active || !tracking) return
      if (poseLandmarker == null) {
        ultimoTimestampPoseMs = 0L
        val opcoesPose = PoseLandmarker.PoseLandmarkerOptions.builder()
          .setBaseOptions(
            BaseOptions.builder()
              .setModelAssetPath("marcador_pose.task")
              .setDelegate(Delegate.CPU)
              .build()
          )
          .setNumPoses(1)
          .setMinPoseDetectionConfidence(0.45f)
          .setMinPosePresenceConfidence(0.45f)
          .setMinTrackingConfidence(0.45f)
          .setRunningMode(RunningMode.VIDEO)
          .build()
        poseLandmarker = PoseLandmarker.createFromOptions(context, opcoesPose)
      }
      val detector = poseLandmarker ?: return
      imagemPose = BitmapImageBuilder(bitmapPose).build()
      var tsPose = SystemClock.elapsedRealtime()
      if (tsPose <= ultimoTimestampPoseMs) tsPose = ultimoTimestampPoseMs + 1L
      ultimoTimestampPoseMs = tsPose

      val pose = detector.detectForVideo(imagemPose, tsPose)
      instanteUltimaPoseMs = SystemClock.elapsedRealtime()
      val poseLandmarks = pose.landmarks().firstOrNull()
      if (poseLandmarks != null && poseLandmarks.size > 16) {
        val indicesPose = intArrayOf(0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        ultimaPose = indicesPose.map { idx ->
          val p = poseLandmarks[idx]
          listOf(p.x().toDouble(), p.y().toDouble())
        }
      } else {
        ultimaPose = emptyList()
      }
    } catch (_: Throwable) {
      ultimaPose = emptyList()
    } finally {
      try { imagemPose?.close() } catch (_: Throwable) {}
      if (!bitmapPose.isRecycled) bitmapPose.recycle()
      poseOcupada.set(false)
    }
  }

  private fun analyzeFrame(proxy: ImageProxy) {
    val inicioNs = SystemClock.elapsedRealtimeNanos()
    var rawBitmap: Bitmap? = null
    var uprightBitmap: Bitmap? = null
    var mpImage: MPImage? = null
    try {
      if (!active || !tracking) return

      rawBitmap = proxy.toBitmap()
      val graus = ((proxy.imageInfo.rotationDegrees % 360) + 360) % 360
      uprightBitmap = if (graus != 0) {
        val matriz = Matrix().apply { postRotate(graus.toFloat()) }
        Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matriz, true)
      } else {
        rawBitmap
      }

      val larguraImg = uprightBitmap.width
      val alturaImg = uprightBitmap.height
      val espelhar = facing == "front"
      mpImage = BitmapImageBuilder(uprightBitmap).build()

      val hands = detectarMaosVideo(mpImage)
      val listaMaos = hands.landmarks().sortedBy { it[0].y() }

      if (listaMaos.isNotEmpty()) {
        val maosTela = listaMaos.map { mao ->
          val coords = FloatArray(42)
          for (i in 0 until 21) {
            val pt = mao[i]
            coords[i * 2] = pt.x()
            coords[i * 2 + 1] = pt.y()
          }
          coords
        }
        camadaRastreamento.atualizarMaos(maosTela, larguraImg, alturaImg, espelhar)

        val agoraMs = SystemClock.elapsedRealtime()
        if (agoraMs - instanteUltimaPoseMs >= 250L) {
          agendarPoseAssincrona(uprightBitmap)
        }
      } else {
        camadaRastreamento.limpar()
        if (SystemClock.elapsedRealtime() - instanteUltimaPoseMs > 1200L) {
          ultimaPose = emptyList()
        }
      }

      val tempoMs = (SystemClock.elapsedRealtimeNanos() - inicioNs) / 1_000_000.0
      emitLandmarks(ultimaPose, listaMaos, tempoMs, larguraImg, alturaImg)
      if (lastError.isNotEmpty()) {
        postarNaInterface { lastError = "" }
      }
    } catch (error: Throwable) {
      val msg = error.cause?.message ?: error.message ?: "Falha no rastreamento local."
      reportError(msg)
    } finally {
      try { mpImage?.close() } catch (_: Throwable) {}
      if (uprightBitmap != null && !uprightBitmap.isRecycled) {
        uprightBitmap.recycle()
      }
      if (rawBitmap != null && rawBitmap !== uprightBitmap && !rawBitmap.isRecycled) {
        rawBitmap.recycle()
      }
      try { proxy.close() } catch (_: Throwable) {}
    }
  }

  private fun emitLandmarks(
    rawPose: List<List<Double>>,
    hands: List<List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>>,
    tempoMs: Double,
    larguraQuadro: Int,
    alturaQuadro: Int
  ) {
    val modelHands3d = if (hands.isEmpty()) {
      emptyList()
    } else {
      hands.map { hand ->
        hand.map { listOf(it.x().toDouble(), it.y().toDouble(), it.z().toDouble()) }
      }
    }

    val dados = mapOf(
      "tem_mao" to hands.isNotEmpty(),
      "largura_quadro" to larguraQuadro,
      "altura_quadro" to alturaQuadro,
      "tempo_ms" to tempoMs,
      "pontos_pose" to rawPose,
      "pontos_maos_3d" to modelHands3d
    )

    postarNaInterface {
      if (active) {
        onLandmarks(dados)
      }
    }
  }

  private fun reportError(message: String) {
    postarNaInterface {
      if (message == lastError) return@postarNaInterface
      lastError = message
      onCameraError(mapOf("message" to message))
    }
  }

  override fun onDetachedFromWindow() {
    layoutListener?.let(previewView::removeOnLayoutChangeListener)
    layoutListener = null
    active = false
    stopCamera()
    if (!analyzerExecutor.isShutdown) analyzerExecutor.shutdown()
    if (!executorPose.isShutdown) executorPose.shutdown()
    super.onDetachedFromWindow()
  }

  private class CamadaRastreamentoMao(context: Context) : View(context) {
    private val densidade = resources.displayMetrics.density
    private val pincelLinha = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.parseColor("#16A34A")
      style = Paint.Style.STROKE
      strokeWidth = 3f * densidade
      strokeCap = Paint.Cap.ROUND
    }
    private val pincelPreenchimento = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      style = Paint.Style.FILL
    }
    private val pincelBorda = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.parseColor("#15803D")
      style = Paint.Style.STROKE
      strokeWidth = 2f * densidade
    }
    private val raioPulso = 5.5f * densidade
    private val raioPonto = 4.5f * densidade

    private val conexoes = arrayOf(
      intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4),
      intArrayOf(0, 5), intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(7, 8),
      intArrayOf(0, 9), intArrayOf(9, 10), intArrayOf(10, 11), intArrayOf(11, 12),
      intArrayOf(0, 13), intArrayOf(13, 14), intArrayOf(14, 15), intArrayOf(15, 16),
      intArrayOf(0, 17), intArrayOf(17, 18), intArrayOf(18, 19), intArrayOf(19, 20),
      intArrayOf(5, 9), intArrayOf(9, 13), intArrayOf(13, 17)
    )

    private val telaX = FloatArray(21)
    private val telaY = FloatArray(21)

    @Volatile private var maosAtuais: List<FloatArray> = emptyList()
    @Volatile private var larguraQuadro = 240
    @Volatile private var alturaQuadro = 320
    @Volatile private var espelhado = true

    fun atualizarMaos(maos: List<FloatArray>, larg: Int, alt: Int, espelhar: Boolean) {
      maosAtuais = maos
      larguraQuadro = larg
      alturaQuadro = alt
      espelhado = espelhar
      postInvalidateOnAnimation()
    }

    fun limpar() {
      if (maosAtuais.isEmpty()) return
      maosAtuais = emptyList()
      postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
      super.onDraw(canvas)
      val maos = maosAtuais
      if (maos.isEmpty()) return

      val larguraVista = width.toFloat()
      val alturaVista = height.toFloat()
      if (larguraVista <= 0f || alturaVista <= 0f) return

      val largQ = larguraQuadro.toFloat()
      val altQ = alturaQuadro.toFloat()
      val usarEscalaQuadro = largQ > 0f && altQ > 0f
      val escala = if (usarEscalaQuadro) max(larguraVista / largQ, alturaVista / altQ) else 1f
      val largRender = if (usarEscalaQuadro) largQ * escala else larguraVista
      val altRender = if (usarEscalaQuadro) altQ * escala else alturaVista
      val corteX = if (usarEscalaQuadro) (largRender - larguraVista) / 2f else 0f
      val corteY = if (usarEscalaQuadro) (altRender - alturaVista) / 2f else 0f
      val espelhar = espelhado

      for (coords in maos) {
        if (coords.size < 42) continue
        for (i in 0 until 21) {
          val px = coords[i * 2]
          val py = coords[i * 2 + 1]
          val nx = if (espelhar) 1f - px else px
          telaX[i] = nx * largRender - corteX
          telaY[i] = py * altRender - corteY
        }

        for (par in conexoes) {
          val a = par[0]
          val b = par[1]
          canvas.drawLine(telaX[a], telaY[a], telaX[b], telaY[b], pincelLinha)
        }

        for (i in 0 until 21) {
          val raio = if (i == 0) raioPulso else raioPonto
          canvas.drawCircle(telaX[i], telaY[i], raio, pincelPreenchimento)
          canvas.drawCircle(telaX[i], telaY[i], raio, pincelBorda)
        }
      }
    }
  }
}
