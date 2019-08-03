diff --git a/tensorflow/cc/saved_model/loader.cc b/tensorflow/cc/saved_model/loader.cc
index 85d3dd01fa..b385f8b857 100644
--- a/tensorflow/cc/saved_model/loader.cc
+++ b/tensorflow/cc/saved_model/loader.cc
@@ -19,6 +19,8 @@ limitations under the License.
 
 #include "tensorflow/cc/saved_model/constants.h"
 #include "tensorflow/cc/saved_model/reader.h"
+#include "tensorflow/core/graph/default_device.h"
+#include "tensorflow/core/grappler/utils.h"
 #include "tensorflow/core/lib/io/path.h"
 #include "tensorflow/core/lib/monitoring/counter.h"
 #include "tensorflow/core/lib/strings/str_util.h"
@@ -244,9 +245,28 @@ Status LoadSavedModelInternal(const SessionOptions& session_options,
                               SavedModelBundle* const bundle) {
   TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(export_dir, tags,
                                                     &bundle->meta_graph_def));
+  
+  // If allocator starts with a '/' then it is being used to
+  // communicate the CPU/GPU that the graph runs on.
+  SessionOptions lsession_options = session_options;
+  const std::string& alloc_type =
+    lsession_options.config.gpu_options().allocator_type();
+  if (!alloc_type.empty() && (alloc_type[0] == '/')) {
+    // Clear the device field from the graphdef so that the default device
+    // setting below will control which GPU the graph will run on.
+    for (tensorflow::NodeDef& node :
+      *bundle->meta_graph_def.mutable_graph_def()->mutable_node()) {
+      if (!tensorflow::grappler::NodeIsOnCpu(&node)) {
+        node.clear_device();
+      }
+    }
+    graph::SetDefaultDevice(alloc_type, bundle->meta_graph_def.mutable_graph_def());
+    lsession_options.config.mutable_gpu_options()->clear_allocator_type();
+  }
 
   TF_RETURN_IF_ERROR(LoadMetaGraphIntoSession(
-      bundle->meta_graph_def, session_options, &bundle->session));
+      bundle->meta_graph_def, lsession_options, &bundle->session));
+
 
   std::vector<AssetFileDef> asset_file_defs;
   TF_RETURN_IF_ERROR(
