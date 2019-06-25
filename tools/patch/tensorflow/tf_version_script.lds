diff --git a/tensorflow/tf_version_script.lds b/tensorflow/tf_version_script.lds
index 563d178..94061df 100644
--- a/tensorflow/tf_version_script.lds
+++ b/tensorflow/tf_version_script.lds
@@ -1,11 +1,7 @@
 tensorflow {
   global:
     *tensorflow*;
-    *perftools*gputools*;
-    *TF_*;
-    *TFE_*;
-    *nsync_*;
-    *stream_executor*;
+    *TRTISTF*;
   local:
     *;
 };
