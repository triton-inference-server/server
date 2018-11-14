diff --git a/tensorflow_serving/sources/storage_path/file_system_storage_path_source.h b/tensorflow_serving/sources/storage_path/file_system_storage_path_source.h
index fd9af2d..9f1964d 100644
--- a/tensorflow_serving/sources/storage_path/file_system_storage_path_source.h
+++ b/tensorflow_serving/sources/storage_path/file_system_storage_path_source.h
@@ -96,6 +96,9 @@ class FileSystemStoragePathSource : public Source<StoragePath> {
 
   AspiredVersionsCallback aspired_versions_callback_ GUARDED_BY(mu_);
 
+  bool run_callback_once_ GUARDED_BY(mu_);
+  bool run_callback_done_ GUARDED_BY(mu_);
+
   // A thread that periodically calls PollFileSystemAndInvokeCallback().
   std::unique_ptr<PeriodicFunction> fs_polling_thread_ GUARDED_BY(mu_);
 
