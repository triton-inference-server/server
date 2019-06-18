diff --git a/tensorflow/tf_version_script.lds b/tensorflow/tf_version_script.lds
index 39d258c..94061df 100644
--- a/tensorflow/tf_version_script.lds
+++ b/tensorflow/tf_version_script.lds
@@ -1,12 +1,7 @@
 tensorflow {
   global:
     *tensorflow*;
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
