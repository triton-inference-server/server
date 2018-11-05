--- tensorflow/cc/saved_model/loader.cc	2018-10-19 10:25:04.454294410 -0700
+++ ../tensorrtserver/tools/patch/tensorflow/cc/saved_model/loader.cc	2018-11-02 11:47:57.254526547 -0700
@@ -19,6 +19,7 @@
 
 #include "tensorflow/cc/saved_model/constants.h"
 #include "tensorflow/cc/saved_model/reader.h"
+#include "tensorflow/core/graph/default_device.h"
 #include "tensorflow/core/lib/io/path.h"
 #include "tensorflow/core/lib/monitoring/counter.h"
 #include "tensorflow/core/lib/strings/str_util.h"
@@ -216,8 +217,20 @@
   TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(export_dir, tags,
                                                     &bundle->meta_graph_def));
 
+  // If visible_device_list starts with a '/' then it is being used to
+  // communicate the CPU/GPU that the graph runs on. This isn't
+  // foolproof since individual operations in the graph could specify
+  // a specific run location. [DLIS-43]
+  SessionOptions lsession_options = session_options;
+  const std::string& vdl =
+    lsession_options.config.gpu_options().visible_device_list();
+  if (!vdl.empty() && (vdl[0] == '/')) {
+    graph::SetDefaultDevice(vdl, bundle->meta_graph_def.mutable_graph_def());
+    lsession_options.config.mutable_gpu_options()->clear_visible_device_list();
+  }
+
   TF_RETURN_IF_ERROR(LoadMetaGraphIntoSession(
-      bundle->meta_graph_def, session_options, &bundle->session));
+      bundle->meta_graph_def, lsession_options, &bundle->session));
 
   std::vector<AssetFileDef> asset_file_defs;
   TF_RETURN_IF_ERROR(
