--- tensorflow_serving/model_servers/server_core.cc	2018-11-01 12:04:22.181086237 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/model_servers/server_core.cc	2018-11-02 13:00:23.162610533 -0700
@@ -342,8 +342,12 @@
     TF_RETURN_IF_ERROR(
         CreateStoragePathSource(source_config, router.get(), &source));
 
+    //
+    // NVIDIA. Ignore errors that occur if some servable fails to
+    // load. In that case we still want to continue and set up the
+    // source, router, manager...
     // Connect the adapters to the manager, and wait for the models to load.
-    TF_RETURN_IF_ERROR(ConnectAdaptersToManagerAndAwaitModelLoads(&adapters));
+    ConnectAdaptersToManagerAndAwaitModelLoads(&adapters);
 
     // Stow the source components.
     storage_path_source_and_router_ = {source.get(), router.get()};
