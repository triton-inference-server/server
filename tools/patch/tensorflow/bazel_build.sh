diff --git a/bazel_build.sh b/bazel_build.sh
index b3cb71f..2959753 100755
--- a/bazel_build.sh
+++ b/bazel_build.sh
@@ -40,9 +40,14 @@ else
 fi
 
 bazel build $BAZEL_OPTS \
-    tensorflow/tools/pip_package:build_pip_package
+    tensorflow/tools/pip_package:build_pip_package \
+    //tensorflow:libtensorflow_cc.so \
+    //tensorflow:libtensorflow_framework.so
 BAZEL_BUILD_RETURN=$?
 
+mkdir -p /usr/local/lib/tensorflow
+cp bazel-bin/tensorflow/libtensorflow_*.so* /usr/local/lib/tensorflow/
+
 if [ ${BAZEL_BUILD_RETURN} -gt 0 ]
 then
   exit ${BAZEL_BUILD_RETURN}
