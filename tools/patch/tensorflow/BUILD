diff --git a/tensorflow/BUILD b/tensorflow/BUILD
index 823ad8f..a370465 100644
--- a/tensorflow/BUILD
+++ b/tensorflow/BUILD
@@ -486,6 +486,20 @@ tf_cc_shared_object(
     ],
 )
 
+cc_library(
+    name = "trtis_tf",
+    srcs = [
+        "tensorflow_backend_tf.cc"
+    ],
+    hdrs = [
+        "tensorflow_backend_tf.h"
+    ],
+    deps = [
+        "//tensorflow/cc/saved_model:loader",
+        "//tensorflow/cc/saved_model:tag_constants",
+    ],
+)
+
 tf_cc_shared_object(
     name = "libtensorflow_cc.so",
     linkopts = select({
@@ -502,6 +516,7 @@ tf_cc_shared_object(
     }),
     visibility = ["//visibility:public"],
     deps = [
+        ":trtis_tf",
         "//tensorflow:tf_exported_symbols.lds",
         "//tensorflow:tf_version_script.lds",
         "//tensorflow/c:c_api",
