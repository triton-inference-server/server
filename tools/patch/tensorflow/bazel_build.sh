diff --git a/bazel_build.sh b/bazel_build.sh
index b3460a488f..42698ae440 100755
--- a/bazel_build.sh
+++ b/bazel_build.sh
@@ -42,10 +42,11 @@ fi
 if [[ $IN_CONTAINER -eq 1 ]]; then
   bazel build $BAZEL_OPTS \
       tensorflow/tools/pip_package:build_pip_package \
-      //tensorflow:libtensorflow_cc.so
+      //tensorflow:libtensorflow_cc.so \
+      //tensorflow:libtensorflow_framework.so
   BAZEL_BUILD_RETURN=$?
   mkdir -p /usr/local/lib/tensorflow
-  cp bazel-bin/tensorflow/libtensorflow_cc.so.? /usr/local/lib/tensorflow
+  cp bazel-bin/tensorflow/libtensorflow_*.so.* /usr/local/lib/tensorflow
 else
   bazel build $BAZEL_OPTS \
       tensorflow/tools/pip_package:build_pip_package
