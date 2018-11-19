diff --git a/c10/util/logging_is_not_google_glog.h b/c10/util/logging_is_not_google_glog.h
index 8d8f7f6..756d199 100644
--- a/c10/util/logging_is_not_google_glog.h
+++ b/c10/util/logging_is_not_google_glog.h
@@ -45,6 +45,8 @@ class C10_API MessageLogger {
 
   const char* tag_;
   std::stringstream stream_;
+  const char* file_;
+  const int line_;
   int severity_;
 };
 
@@ -96,14 +98,40 @@ static_assert(
 #define LOG(n)                   \
   if (n >= CAFFE2_LOG_THRESHOLD) \
   ::c10::MessageLogger((char*)__FILE__, __LINE__, n).stream()
-#define VLOG(n) LOG((-n))
 
 #define LOG_IF(n, condition)                    \
   if (n >= CAFFE2_LOG_THRESHOLD && (condition)) \
   ::c10::MessageLogger((char*)__FILE__, __LINE__, n).stream()
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
 #define FATAL_IF(condition)            \
