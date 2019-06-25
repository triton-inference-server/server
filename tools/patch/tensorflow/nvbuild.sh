diff --git a/nvbuild.sh b/nvbuild.sh
index e54d9d2..3f0f7e0 100755
--- a/nvbuild.sh
+++ b/nvbuild.sh
@@ -64,7 +64,7 @@ export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
 THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
 cd "${THIS_DIR}/tensorflow-source"
 export PYTHON_BIN_PATH=/usr/bin/python$PYVER
-LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)
+LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l)
 if [[ $NOCONFIG -eq 0 ]]; then
   if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
