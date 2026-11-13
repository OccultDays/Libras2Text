# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /usr/local/Cellar/android-sdk/24.3.3/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# react-native-reanimated
-keep class com.swmansion.reanimated.** { *; }
-keep class com.facebook.react.turbomodule.** { *; }

# Add any project specific keep options here:
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keep interface com.google.protobuf.** { *; }
-keep class * extends com.google.protobuf.GeneratedMessageLite { *; }
-keep class com.google.common.flogger.** { *; }
-keep interface com.google.common.flogger.** { *; }
-keep class com.google.android.datatransport.** { *; }
-keep interface com.google.android.datatransport.** { *; }
-keep class expo.modules.librasnative.** { *; }
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate
