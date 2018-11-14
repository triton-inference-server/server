diff --git a/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc b/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc
index ea49918..397c939 100644
--- a/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc
+++ b/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc
@@ -357,6 +357,17 @@ void FileSystemStoragePathSource::SetAspiredVersionsCallback(
   aspired_versions_callback_ = callback;
 
   if (config_.file_system_poll_wait_seconds() >= 0) {
+    run_callback_once_ = false;
+    run_callback_done_ = false;
+
+    // If poll_wait_seconds == 0 then we need to poll just once to
+    // initialize from the model repository.
+    int64 poll_secs = config_.file_system_poll_wait_seconds();
+    if (poll_secs == 0) {
+      poll_secs = (int64)INT32_MAX;
+      run_callback_once_ = true;
+    }
+
     // Kick off a thread to poll the file system periodically, and call the
     // callback.
     PeriodicFunction::Options pf_options;
@@ -364,14 +375,19 @@ void FileSystemStoragePathSource::SetAspiredVersionsCallback(
         "FileSystemStoragePathSource_filesystem_polling_thread";
     fs_polling_thread_.reset(new PeriodicFunction(
         [this] {
-          Status status = this->PollFileSystemAndInvokeCallback();
-          if (!status.ok()) {
-            LOG(ERROR) << "FileSystemStoragePathSource encountered a "
-                          "file-system access error: "
-                       << status.error_message();
+          if (!this->run_callback_done_) {
+            Status status = this->PollFileSystemAndInvokeCallback();
+            if (!status.ok()) {
+              LOG(ERROR) << "FileSystemStoragePathSource encountered a "
+                            "file-system access error: "
+                         << status.error_message();
+            }
+          }
+          if (this->run_callback_once_) {
+            this->run_callback_done_ = true;
           }
         },
-        config_.file_system_poll_wait_seconds() * 1000000, pf_options));
+        poll_secs * 1000000, pf_options));
   }
 }
 
