diff --git a/tensorflow/tf_version_script.lds b/tensorflow/tf_version_script.lds
index 39d258c..dc76749 100644
--- a/tensorflow/tf_version_script.lds
+++ b/tensorflow/tf_version_script.lds
@@ -1,12 +1,6 @@
 tensorflow {
   global:
-    *tensorflow*;
-    *perftools*gputools*;
-    *TF_*;
-    *TFE_*;
-    *nsync_*;
-    *pywrap_xla*;
-    *stream_executor*;
+    *TRTISTF*;
   local:
     *;
 };
