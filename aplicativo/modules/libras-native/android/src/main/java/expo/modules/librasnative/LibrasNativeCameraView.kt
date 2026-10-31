package expo.modules.librasnative

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class LibrasNativeCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  private val previewView = TextureView(context)
  private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private var cameraProvider: ProcessCameraProvider? = null
  private var previewUseCase: Preview? = null
  private var analysisUseCase: ImageAnalysis? = null
  private var handLandmarker: HandLandmarker? = null
  private var poseLandmarker: PoseLandmarker? = null
  private var active = false
  private var readyNotified = false
  @Volatile private var tracking = false
  private var facing = "front"
  private var lastError = ""
  private var layoutListener: OnLayoutChangeListener? = null
  private var bindGeneration = 0
  private var bufferLargura = 480
  private var bufferAltura = 640
  private var bufferRgba: ByteArray? = null
  private var contadorPose = 0
  private var instanteUltimaPoseMs = 0L
  private var ultimaPose: List<List<Double>> = emptyList()

  private val onLandmarks by EventDispatcher()
  private val onCameraReady by EventDispatcher()
  private val onCameraError by EventDispatcher()

  init {
    addView(previewView, LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
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
  }

  fun setFacing(value: String) {
    val normalized = if (value == "back") "back" else "front"
    if (facing == normalized) return
    facing = normalized
    readyNotified = false
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
          request.provideSurface(surface, executorPrincipal) { result ->
            surface.release()
            if (result.resultCode == androidx.camera.core.SurfaceRequest.Result.RESULT_SURFACE_USED_SUCCESSFULLY) {
              notifyCameraReady()
            }
          }
        }

        val analysis = ImageAnalysis.Builder()
          .setTargetResolution(Size(360, 480))
          .setTargetRotation(displayRotation)
          .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
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
      } catch (error: Exception) {
        reportError(error.message ?: "Nao foi possivel iniciar a camera.")
      }
    }, ContextCompat.getMainExecutor(context))
  }

  private fun notifyCameraReady() {
    if (!active || readyNotified) return
    readyNotified = true
    onCameraReady(mapOf("ready" to true))
  }

  private fun stopCamera() {
    bindGeneration++
    val provider = cameraProvider
    if (provider != null) {
      val preview = previewUseCase
      val analysis = analysisUseCase
      if (preview != null) provider.unbind(preview)
      if (analysis != null) provider.unbind(analysis)
    }
    previewUseCase = null
    analysisUseCase = null
    analyzerExecutor.execute {
      try { handLandmarker?.close() } catch (_: Exception) {}
      try { poseLandmarker?.close() } catch (_: Exception) {}
      handLandmarker = null
      poseLandmarker = null
      ultimaPose = emptyList()
      contadorPose = 0
    }
  }

  private fun ensureLandmarkers() {
    if (handLandmarker != null && poseLandmarker != null) return
    val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("marcador_maos.task").build())
      .setNumHands(2)
      .setMinHandDetectionConfidence(0.25f)
      .setMinHandPresenceConfidence(0.25f)
      .setRunningMode(RunningMode.IMAGE)
      .build()
    val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("marcador_pose.task").build())
      .setNumPoses(1)
      .setMinPoseDetectionConfidence(0.45f)
      .setRunningMode(RunningMode.IMAGE)
      .build()
    handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
    poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
  }

  private fun analyzeFrame(proxy: ImageProxy) {
    val inicioNs = SystemClock.elapsedRealtimeNanos()
    var rawBitmap: Bitmap? = null
    var uprightBitmap: Bitmap? = null
    try {
      if (!tracking) return
      ensureLandmarkers()
      rawBitmap = rgbaToBitmap(proxy)
      val graus = ((proxy.imageInfo.rotationDegrees % 360) + 360) % 360
      uprightBitmap = if (graus != 0) {
        val matriz = Matrix().apply { postRotate(graus.toFloat()) }
        Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matriz, true)
      } else {
        rawBitmap
      }

      val image = BitmapImageBuilder(uprightBitmap).build()
      val agoraMs = SystemClock.elapsedRealtime()

      val hands = handLandmarker!!.detect(image)
      if (hands.landmarks().isNotEmpty()) {
        val atualizarPose = (contadorPose % 4 == 0) || (agoraMs - instanteUltimaPoseMs) > 1000L
        contadorPose++
        if (atualizarPose) {
          val pose = poseLandmarker!!.detect(image)
          instanteUltimaPoseMs = agoraMs
          val poseLandmarks = pose.landmarks().firstOrNull()
          if (poseLandmarks != null) {
            val headIndices = intArrayOf(0, 2, 5, 7, 8, 9, 10)
            val armIndices = intArrayOf(11, 12, 13, 14, 15, 16)
            ultimaPose = (headIndices + armIndices).map { idx ->
              val p = poseLandmarks[idx]
              listOf(p.x().toDouble(), p.y().toDouble())
            }
          } else {
            ultimaPose = emptyList()
          }
        }
      }

      val tempoMs = (SystemClock.elapsedRealtimeNanos() - inicioNs) / 1_000_000.0
      emitLandmarks(ultimaPose, hands, tempoMs, uprightBitmap.width, uprightBitmap.height)
      lastError = ""
    } catch (error: Exception) {
      reportError(error.message ?: "Falha no rastreamento local.")
    } finally {
      if (uprightBitmap != null && uprightBitmap !== rawBitmap) {
        uprightBitmap.recycle()
      }
      rawBitmap?.recycle()
      proxy.close()
    }
  }

  private fun rgbaToBitmap(proxy: ImageProxy): Bitmap {
    val width = proxy.width
    val height = proxy.height
    val plane = proxy.planes[0]
    val source = plane.buffer.duplicate()
    val sourceStride = plane.rowStride
    val sourcePixelStride = plane.pixelStride
    val totalBytes = width * height * 4
    var rgba = bufferRgba
    if (rgba == null || rgba.size != totalBytes) {
      rgba = ByteArray(totalBytes)
      bufferRgba = rgba
    }
    for (y in 0 until height) {
      for (x in 0 until width) {
        val input = y * sourceStride + x * sourcePixelStride
        val output = (y * width + x) * 4
        rgba[output] = source.get(input + 2)
        rgba[output + 1] = source.get(input + 1)
        rgba[output + 2] = source.get(input)
        rgba[output + 3] = source.get(input + 3)
      }
    }
    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
      it.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
    }
  }

  private fun emitLandmarks(
    rawPose: List<List<Double>>,
    handResult: HandLandmarkerResult,
    tempoMs: Double,
    larguraQuadro: Int,
    alturaQuadro: Int
  ) {
    val hasPose = rawPose.isNotEmpty()
    val hands = handResult.landmarks().sortedBy { it[0].y() }
    val selectedHand = hands.firstOrNull()

    val rawHands = hands.map { hand ->
      hand.map { listOf(it.x().toDouble(), it.y().toDouble()) }
    }
    val modelHands3d = hands.map { hand ->
      hand.map { listOf(it.x().toDouble(), it.y().toDouble(), it.z().toDouble()) }
    }
    val selectedRawHand = selectedHand?.map { listOf(it.x().toDouble(), it.y().toDouble()) } ?: emptyList()
    val selectedModelHand = modelHands3d.firstOrNull() ?: emptyList()

    onLandmarks(mapOf(
      "tem_pose" to hasPose,
      "tem_mao" to (selectedHand != null),
      "maos_detectadas" to hands.size,
      "largura_quadro" to larguraQuadro,
      "altura_quadro" to alturaQuadro,
      "tempo_ms" to tempoMs,
      "pontos_pose" to rawPose,
      "pontos_mao" to selectedRawHand,
      "pontos_mao_3d" to selectedModelHand,
      "pontos_maos" to rawHands,
      "pontos_maos_3d" to modelHands3d
    ))
  }

  private fun reportError(message: String) {
    if (message == lastError) return
    lastError = message
    onCameraError(mapOf("message" to message))
  }

  override fun onDetachedFromWindow() {
    layoutListener?.let(previewView::removeOnLayoutChangeListener)
    layoutListener = null
    active = false
    stopCamera()
    if (!analyzerExecutor.isShutdown) analyzerExecutor.shutdown()
    super.onDetachedFromWindow()
  }
}
