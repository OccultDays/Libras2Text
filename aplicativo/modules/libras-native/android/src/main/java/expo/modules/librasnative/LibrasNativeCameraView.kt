package expo.modules.librasnative

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.os.SystemClock
import android.util.Size
import android.view.ViewGroup
import android.view.Surface
import android.view.TextureView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

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
  private var lastTimestampMs = 0L
  private var lastError = ""
  private var layoutListener: OnLayoutChangeListener? = null
  private var bindGeneration = 0

  private val onLandmarks by EventDispatcher()
  private val onCameraReady by EventDispatcher()
  private val onCameraError by EventDispatcher()

  init {
    addView(previewView, LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    previewView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
      override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        if (active) startCamera()
      }
      override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit
      override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
        notifyCameraReady()
      }
      override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
    }
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
    // Wait until React Native has measured the embedded PreviewView. Binding a
    // CameraX preview to a zero-sized TextureView can leave its surface black.
    if (previewView.width == 0 || previewView.height == 0 || !previewView.isAvailable) {
      if (layoutListener == null) {
        val listener = OnLayoutChangeListener { view, _, _, _, _, _, _, _, _ ->
          if (view.width > 0 && view.height > 0) {
            layoutListener?.let(view::removeOnLayoutChangeListener)
            layoutListener = null
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
        // Match Preview and ImageAnalysis to the current display orientation.
        val preview = Preview.Builder()
          .setTargetRotation(displayRotation)
          .build()
        preview.setSurfaceProvider(ContextCompat.getMainExecutor(context)) { request ->
          val texture = previewView.surfaceTexture
          if (texture == null || !previewView.isAvailable) {
            request.willNotProvideSurface()
            return@setSurfaceProvider
          }
          texture.setDefaultBufferSize(request.resolution.width, request.resolution.height)
          val surface = Surface(texture)
          request.provideSurface(surface, ContextCompat.getMainExecutor(context)) { result ->
            surface.release()
            if (result.resultCode == androidx.camera.core.SurfaceRequest.Result.RESULT_SURFACE_USED_SUCCESSFULLY) notifyCameraReady()
          }
        }
        val analysis = ImageAnalysis.Builder()
          .setTargetResolution(Size(640, 480))
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
    }
  }

  private fun ensureLandmarkers() {
    if (handLandmarker != null && poseLandmarker != null) return
    val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("marcador_maos.task").build())
      .setNumHands(2)
      .setMinHandDetectionConfidence(0.20f)
      .setMinHandPresenceConfidence(0.25f)
      .setMinTrackingConfidence(0.30f)
      .setRunningMode(RunningMode.VIDEO)
      .build()
    val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("marcador_pose.task").build())
      .setNumPoses(1)
      .setMinPoseDetectionConfidence(0.5f)
      .setMinTrackingConfidence(0.55f)
      .setRunningMode(RunningMode.VIDEO)
      .build()
    handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
    poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
  }

  private fun analyzeFrame(proxy: ImageProxy) {
    val inicioNs = SystemClock.elapsedRealtimeNanos()
    var bitmap: Bitmap? = null
    try {
      if (!tracking) return
      ensureLandmarkers()
      val frameBitmap = rgbaToBitmap(proxy)
      bitmap = frameBitmap
      val image = BitmapImageBuilder(frameBitmap).build()
      val rotation = ImageProcessingOptions.builder()
        .setRotationDegrees(proxy.imageInfo.rotationDegrees)
        .build()
      val timestamp = maxOf(SystemClock.elapsedRealtime(), lastTimestampMs + 1)
      lastTimestampMs = timestamp

      val pose = poseLandmarker!!.detectForVideo(image, rotation, timestamp)
      val hands = handLandmarker!!.detectForVideo(image, rotation, timestamp)
      val tempoMs = (SystemClock.elapsedRealtimeNanos() - inicioNs) / 1_000_000.0
      emitLandmarks(pose, hands, tempoMs, proxy.imageInfo.rotationDegrees)
      lastError = ""
    } catch (error: Exception) {
      reportError(error.message ?: "Falha no rastreamento local.")
    } finally {
      bitmap?.recycle()
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
    val rgba = ByteArray(width * height * 4)
    for (y in 0 until height) {
      for (x in 0 until width) {
        val input = y * sourceStride + x * sourcePixelStride
        val output = (y * width + x) * 4
        // CameraX provides bytes as R,G,B,A. Android ARGB_8888 bitmaps store
        // pixel bytes as B,G,R,A on Android's little-endian devices.
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

  private fun emitLandmarks(poseResult: PoseLandmarkerResult, handResult: HandLandmarkerResult, tempoMs: Double, rotationDegrees: Int) {
    val poseLandmarks = poseResult.landmarks().firstOrNull()
    val hasPose = poseLandmarks != null
    val hands = handResult.landmarks()
    val hasHand = hands.isNotEmpty()

    val headIndices = intArrayOf(0, 2, 5, 7, 8, 9, 10)
    val armIndices = intArrayOf(11, 12, 13, 14, 15, 16)
    val head = MutableList(7) { DoubleArray(3) }
    val arms = MutableList(6) { DoubleArray(3) }
    val rawPose = mutableListOf<List<Double>>()

    if (poseLandmarks != null) {
      for (i in headIndices.indices) {
        val point = poseLandmarks[headIndices[i]]
        head[i] = doubleArrayOf(point.x().toDouble(), point.y().toDouble(), point.z().toDouble())
      }
      for (i in armIndices.indices) {
        val point = poseLandmarks[armIndices[i]]
        arms[i] = doubleArrayOf(point.x().toDouble(), point.y().toDouble(), point.z().toDouble())
      }
      for (index in headIndices + armIndices) {
        val point = poseLandmarks[index]
        rawPose.add(coordenadasNaOrientacaoDoDisplay(point.x().toDouble(), point.y().toDouble(), rotationDegrees))
      }
    }

    val rawHands = hands.map { hand -> hand.map { coordenadasNaOrientacaoDoDisplay(it.x().toDouble(), it.y().toDouble(), rotationDegrees) } }
    val selectedHand = if (hasHand) {
      if (poseLandmarks != null) {
        val wrists = listOf(arms[4], arms[5])
        val nearest = hands.minByOrNull { hand ->
          wrists.minOf { wrist -> distance2d(hand[0].x().toDouble(), hand[0].y().toDouble(), wrist[0], wrist[1]) }
        }
        nearest?.takeIf { hand ->
          wrists.minOf { wrist -> distance2d(hand[0].x().toDouble(), hand[0].y().toDouble(), wrist[0], wrist[1]) } <= 0.0324
        }
      } else hands.first()
    } else null

    val handPoints = MutableList(21) { DoubleArray(3) }
    if (selectedHand != null) {
      for (i in selectedHand.indices) {
        val point = selectedHand[i]
        handPoints[i] = doubleArrayOf(point.x().toDouble(), point.y().toDouble(), point.z().toDouble())
      }
    } else if (poseLandmarks != null) {
      for (i in handPoints.indices) handPoints[i] = arms[5].copyOf()
    }

    val allPoints = head + arms + handPoints
    val reference = if (hasPose) {
      DoubleArray(3) { (allPoints[7][it] + allPoints[8][it]) / 2.0 }
    } else allPoints[13].copyOf()
    val adjusted = allPoints.map { point -> DoubleArray(3) { point[it] - reference[it] } }
    val scale = adjusted.maxOf { point -> sqrt(point.sumOf { it * it }) }.coerceAtLeast(1e-9)
    val vector = adjusted.flatMap { point -> point.map { it / scale } }

    val selectedRawHand = selectedHand?.map { coordenadasNaOrientacaoDoDisplay(it.x().toDouble(), it.y().toDouble(), rotationDegrees) } ?: emptyList()
    val selectedModelHand = selectedHand?.map { point ->
      val xy = coordenadasNaOrientacaoDoDisplay(point.x().toDouble(), point.y().toDouble(), rotationDegrees)
      listOf(xy[0], xy[1], point.z().toDouble())
    } ?: emptyList()
    // The classifier was trained with a complete upper-body + hand vector.
    // Never fabricate zero-valued body/hand landmarks when either detector
    // misses; those vectors produced confident but meaningless translations.
    if (!hasPose || selectedHand == null) {
      onLandmarks(mapOf(
        "tem_pose" to hasPose,
        "tem_mao" to hasHand,
        "maos_detectadas" to hands.size,
        "diagnostico" to when {
          !hasHand -> "Nenhuma mao detectada; centralize a palma e melhore a iluminacao."
          selectedHand == null -> "Mao detectada, mas o pulso nao foi associado ao braco."
          else -> "Parte superior do corpo nao detectada."
        },
        "vetor" to emptyList<Double>(),
        "tempo_ms" to tempoMs,
        "pontos_pose" to rawPose,
        "pontos_mao" to selectedRawHand,
        "pontos_mao_3d" to selectedModelHand,
        "pontos_maos" to rawHands
      ))
      return
    }

    val event = mutableMapOf<String, Any>(
      "tem_pose" to hasPose,
      "tem_mao" to hasHand,
      "vetor" to vector,
      "tempo_ms" to tempoMs,
      "pontos_pose" to rawPose,
      "pontos_mao" to selectedRawHand,
      "pontos_mao_3d" to selectedModelHand,
      "pontos_maos" to rawHands
    )
    onLandmarks(event)
  }

  /** Maps normalized ImageAnalysis coordinates to the display orientation used by Preview. */
  private fun coordenadasNaOrientacaoDoDisplay(x: Double, y: Double, rotationDegrees: Int): List<Double> {
    return when (((rotationDegrees % 360) + 360) % 360) {
      90 -> listOf(1.0 - y, x)
      180 -> listOf(1.0 - x, 1.0 - y)
      270 -> listOf(y, 1.0 - x)
      else -> listOf(x, y)
    }
  }
  private fun distance2d(x1: Double, y1: Double, x2: Double, y2: Double): Double {
    val dx = x1 - x2
    val dy = y1 - y2
    return dx * dx + dy * dy
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
