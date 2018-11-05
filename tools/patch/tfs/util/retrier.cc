--- tensorflow_serving/util/retrier.cc	2018-05-15 13:11:26.068986416 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/util/retrier.cc	2018-10-12 12:44:37.280572118 -0700
@@ -42,7 +42,9 @@
   if (is_cancelled()) {
     LOG(INFO) << "Retrying of " << description << " was cancelled.";
   }
-  if (num_tries == max_num_retries + 1) {
+
+  // NVIDIA: Add max_num_retries>0 check to avoid spurious logging
+  if ((max_num_retries > 0) && (num_tries == max_num_retries + 1)) {
     LOG(INFO) << "Retrying of " << description
               << " exhausted max_num_retries: " << max_num_retries;
   }
