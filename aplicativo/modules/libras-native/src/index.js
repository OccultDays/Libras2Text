import { requireNativeViewManager } from 'expo-modules-core';
import { Platform } from 'react-native';

export default Platform.OS === 'android'
  ? requireNativeViewManager('LibrasNative', 'LibrasNativeCameraView')
  : null;
