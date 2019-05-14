diff --git a/nvbuild.sh b/nvbuild.sh
index 674c0dd..e0b8656 100755
--- a/nvbuild.sh
+++ b/nvbuild.sh
@@ -57,7 +57,7 @@ export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
 
 cd "$THIS_DIR"
 export PYTHON_BIN_PATH=/usr/bin/python$PYVER
-LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)
+LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l)
 if [[ $NOCONFIG -eq 0 ]]; then
   if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
@@ -79,4 +79,3 @@ export NOCLEAN
 export PYVER
 export LIBCUDA_FOUND
 bash bazel_build.sh
-
