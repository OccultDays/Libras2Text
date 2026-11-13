package expo.modules.librasnative

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class LibrasNativeModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("LibrasNative")

    View(LibrasNativeCameraView::class) {
      Events("onLandmarks", "onCameraReady", "onCameraError")

      Prop("active") { view: LibrasNativeCameraView, active: Boolean ->
        view.setActive(active)
      }

      Prop("tracking") { view: LibrasNativeCameraView, tracking: Boolean ->
        view.setTracking(tracking)
      }

      Prop("facing") { view: LibrasNativeCameraView, facing: String ->
        view.setFacing(facing)
      }
    }
  }
}
