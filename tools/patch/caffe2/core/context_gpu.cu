--- pytorch/caffe2/core/context_gpu.cu	2018-11-02 12:29:48.223056985 -0700
+++ ../tensorrtserver/tools/patch/caffe2/core/context_gpu.cu	2018-10-12 12:44:37.280572118 -0700
@@ -152,7 +152,13 @@
         // Note: just for future reference, the 0 here is not a gpu id, it is
         // a reserved flag for cudaDeviceEnablePeerAccess that should always be
         // zero currently.
-        CUDA_ENFORCE(cudaDeviceEnablePeerAccess(j, 0));
+        // It is ok if peer access is already enabled...
+        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
+        if ((err != cudaErrorPeerAccessAlreadyEnabled) &&
+            (err != cudaSuccess)) {
+          CAFFE_THROW(cudaGetErrorString(err));
+        }
+        cudaGetLastError(); // reset cuda error code
       }
     }
   }
