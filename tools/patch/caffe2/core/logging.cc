--- pytorch/caffe2/core/logging.cc	2018-11-02 12:29:48.223056985 -0700
+++ ../tensorrtserver/tools/patch/caffe2/core/logging.cc	2018-10-12 16:39:38.071855234 -0700
@@ -180,8 +180,25 @@
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
 MessageLogger::MessageLogger(const char *file, int line, int severity)
-  : severity_(severity) {
+  : file_(file), line_(line), severity_(severity) {
+
+  if (DelegatedLogMessage) {
+    return;
+  }
+
   if (severity_ < FLAGS_caffe2_log_level) {
     // Nothing needs to be logged.
     return;
@@ -212,6 +229,19 @@

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
