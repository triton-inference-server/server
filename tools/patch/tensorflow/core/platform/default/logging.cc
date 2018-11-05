--- tensorflow/core/platform/default/logging.cc	2018-07-13 12:53:35.159015218 -0700
+++ ../tensorrtserver/tools/patch/tensorflow/core/platform/default/logging.cc	2018-10-12 16:39:38.071855234 -0700
@@ -76,7 +76,31 @@
 
 #else
 
+// NVIDIA
+//
+// Update to deliver the message via tensorrtserver logger. The
+// extern declaration must be kept in sync with that defined in
+// tensorrtserver/src/core/logging.h.
+//
+#define LOG_DELEGATED_ERROR_LEVEL 0
+#define LOG_DELEGATED_WARNING_LEVEL 1
+#define LOG_DELEGATED_INFO_LEVEL 2
+extern "C" void DelegatedLogMessage(
+  int level, const char* file, int line, const std::string& msg) __attribute__((weak));
+
 void LogMessage::GenerateLogMessage() {
+  int level = LOG_DELEGATED_ERROR_LEVEL;
+  if (severity_ == INFO) {
+    level = LOG_DELEGATED_INFO_LEVEL;
+  } else if (severity_ == WARNING) {
+    level = LOG_DELEGATED_WARNING_LEVEL;
+  }
+
+  if (DelegatedLogMessage) {
+    DelegatedLogMessage(level, fname_, line_, str());
+    return;
+  }
+
   static EnvTime* env_time = tensorflow::EnvTime::Default();
   uint64 now_micros = env_time->NowMicros();
   time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
@@ -121,7 +145,22 @@
   return LogLevelStrToInt(tf_env_var_val);
 }
 
+// NVIDIA
+//
+// Update to get verbose logging level via inference-server
+// logger. The extern declaration must be kept in sync with that
+// defined in tensorrtserver/src/core/logging.h.
+//
+extern "C" uint32_t DelegatedVerboseLogLevel() __attribute__((weak));
+
 int64 MinVLogLevelFromEnv() {
+  if (DelegatedVerboseLogLevel) {
+    int64 lvl = DelegatedVerboseLogLevel();
+    // Reduce the inference-server verbose-log level by 1 for TF. That
+    // is, inference-server verbose level 2 is TF vlog level 1.
+    return (lvl <= 0) ? 0 : (lvl - 1);
+  }
+
   const char* tf_env_var_val = getenv("TF_CPP_MIN_VLOG_LEVEL");
   return LogLevelStrToInt(tf_env_var_val);
 }
