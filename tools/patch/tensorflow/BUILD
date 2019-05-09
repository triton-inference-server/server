diff --git a/tensorflow/BUILD b/tensorflow/BUILD
index 823ad8f..ee1dabb 100644
--- a/tensorflow/BUILD
+++ b/tensorflow/BUILD
@@ -625,3 +625,32 @@ py_library(
     visibility = ["//visibility:public"],
     deps = ["//tensorflow/python:no_contrib"],
 )
+
+cc_library(
+    name = "trtis_tf",
+    srcs = [
+    ],
+    deps = [
+        "//tensorflow/contrib:contrib_kernels",
+        "//tensorflow/contrib:contrib_ops_op_lib",
+        "//tensorflow/contrib/tensorrt:trt_engine_op_op_lib",
+        "//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
+        "//tensorflow/contrib/tensorrt:trt_shape_function",
+        "//tensorflow/core:framework_internal",
+        "//tensorflow/core:tensorflow",
+        "//tensorflow/cc:cc_ops",
+        "//tensorflow/c:c_api",
+    ],
+)
+
+cc_binary(
+    name = "libtensorflow_trtis.so",
+    deps = [
+        ":trtis_tf",
+        ":libtensorflow_trtis.ldscript"
+    ],
+    linkopts = [
+        "-Wl,--version-script", "$(location :libtensorflow_trtis.ldscript)"
+    ],
+    linkshared = 1,
+)
