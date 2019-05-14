diff --git a/tensorflow/tf_version_script.lds b/tensorflow/tf_version_script.lds
index 39d258c..ea6718e 100644
--- a/tensorflow/tf_version_script.lds
+++ b/tensorflow/tf_version_script.lds
@@ -7,6 +7,7 @@ tensorflow {
     *nsync_*;
     *pywrap_xla*;
     *stream_executor*;
+    *TFWorkspace*;
   local:
     *;
 };
