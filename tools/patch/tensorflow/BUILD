diff --git a/tensorflow/BUILD b/tensorflow/BUILD
index a04ddf9..28f1db3 100644
--- a/tensorflow/BUILD
+++ b/tensorflow/BUILD
@@ -531,6 +531,20 @@ tf_cc_shared_object(
 # an "-exported_symbols_list" command.  -z defs disallows undefined
 # symbols in object files.
 
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
     name = "tensorflow",
     linkopts = select({
@@ -585,6 +599,9 @@ tf_cc_shared_object(
         "//conditions:default": None,
     }),
     deps = [
+        ":trtis_tf",
+        "//tensorflow/contrib:contrib_kernels",
+        "//tensorflow/contrib:contrib_ops_op_lib",
         "//tensorflow:tf_exported_symbols.lds",
         "//tensorflow:tf_version_script.lds",
         "//tensorflow/c:c_api",
