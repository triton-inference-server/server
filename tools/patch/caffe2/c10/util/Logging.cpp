diff --git a/c10/util/Logging.cpp b/c10/util/Logging.cpp
index 2540576..9fd3c97 100644
--- a/c10/util/Logging.cpp
+++ b/c10/util/Logging.cpp
@@ -183,8 +183,25 @@ void ShowLogInfoToStderr() {
   FLAGS_caffe2_log_level = INFO;
 }
 
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
 MessageLogger::MessageLogger(const char* file, int line, int severity)
-    : severity_(severity) {
+  : file_(file), line_(line), severity_(severity) {
+
+  if (DelegatedLogMessage) {
+    return;
+  }
+
   if (severity_ < FLAGS_caffe2_log_level) {
     // Nothing needs to be logged.
     return;
@@ -217,6 +234,19 @@ MessageLogger::MessageLogger(const char* file, int line, int severity)
 
 // Output the contents of the stream to the proper channel on destruction.
 MessageLogger::~MessageLogger() {
+
+  int level = LOG_DELEGATED_ERROR_LEVEL;
+  if (severity_ == INFO) {
+    level = LOG_DELEGATED_INFO_LEVEL;
+  } else if (severity_ == WARNING) {
+    level = LOG_DELEGATED_WARNING_LEVEL;
+  }
+
+  if (DelegatedLogMessage) {
+    DelegatedLogMessage(level, file_, line_, stream_.str());
+    return;
+  }
+
   if (severity_ < FLAGS_caffe2_log_level) {
     // Nothing needs to be logged.
     return;
