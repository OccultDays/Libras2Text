import { requireOptionalNativeModule, requireNativeViewManager } from 'expo-modules-core';
import { Platform } from 'react-native';

const moduloNativo = Platform.OS === 'android'
  ? requireOptionalNativeModule('LibrasNative')
  : null;

const cameraNativa = moduloNativo
  ? requireNativeViewManager('LibrasNative', 'LibrasNativeCameraView')
  : null;

export default cameraNativa;
