--- pytorch/caffe2/core/logging_is_not_google_glog.h	2018-11-02 12:29:48.223056985 -0700
+++ ../tensorrtserver/tools/patch/caffe2/core/logging_is_not_google_glog.h	2018-10-12 16:39:38.071855234 -0700
@@ -40,6 +40,8 @@
 
   const char* tag_;
   std::stringstream stream_;
+  const char* file_;
+  const int line_;
   int severity_;
 };
 
@@ -91,14 +93,40 @@
 #define LOG(n) \
   if (n >= CAFFE2_LOG_THRESHOLD) \
     ::caffe2::MessageLogger((char*)__FILE__, __LINE__, n).stream()
-#define VLOG(n) LOG((-n))
 
 #define LOG_IF(n, condition)                    \
   if (n >= CAFFE2_LOG_THRESHOLD && (condition)) \
   ::caffe2::MessageLogger((char*)__FILE__, __LINE__, n).stream()
-#define VLOG_IF(n, condition) LOG_IF((-n), (condition))
 
-#define VLOG_IS_ON(verboselevel) (CAFFE2_LOG_THRESHOLD <= -(verboselevel))
+namespace {
+// NVIDIA
+//
+// Get verbose logging level via inference-server logger. The extern
+// declaration must be kept in sync with that defined in
+// tensorrtserver/src/core/logging.h.
+//
+extern "C" uint32_t DelegatedVerboseLogLevel() __attribute__((weak));
+static inline int
+NVDelegatedVerboseLogLevel()
+{
+  if (DelegatedVerboseLogLevel) {
+    int lvl = DelegatedVerboseLogLevel();
+    // Reduce the inference-server verbose-log level by 1 for C2. That
+    // is, inference-server verbose level 2 is C2 vlog level 1.
+    return (lvl <= 0) ? 0 : (lvl - 1);
+  }
+
+  return 0;
+}
+} // namespace
+
+#define VLOG_IS_ON(n) ((n) <= NVDelegatedVerboseLogLevel())
+#define VLOG(n)                                                       \
+  if (VLOG_IS_ON(n))                                                  \
+    ::caffe2::MessageLogger((char*)__FILE__, __LINE__, INFO).stream()
+#define VLOG_IF(n, condition)                                         \
+  if (VLOG_IS_ON(n) && (condition))                                   \
+    ::caffe2::MessageLogger((char*)__FILE__, __LINE__, INFO).stream()
 
 // Log only if condition is met.  Otherwise evaluates to void.
 #define FATAL_IF(condition) \
