--- tensorflow_serving/util/net_http/server/internal/evhttp_request.h	2018-11-01 12:04:22.185086318 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/util/net_http/server/internal/evhttp_request.h	2018-11-02 11:47:57.254526547 -0700
@@ -21,6 +21,7 @@
 #include <cstdint>
 #include <memory>
 
+#include "libevent/include/event2/keyvalq_struct.h"
 #include "tensorflow_serving/util/net_http/server/internal/server_support.h"
 #include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"
 #include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
@@ -55,6 +56,8 @@
   // evhttp_uridecode(path)
   const char* path = nullptr;  // owned by uri
 
+  struct evkeyvalq params;  // query parameters
+
   evkeyvalq* headers = nullptr;  // owned by raw request
 };
 
@@ -75,6 +78,10 @@
 
   absl::string_view http_method() const override;
 
+  bool QueryParam(absl::string_view key, std::string* str) const override;
+  evbuffer* InputBuffer() const override;
+  evbuffer* OutputBuffer() override;
+
   void WriteResponseBytes(const char* data, int64_t size) override;
 
   void WriteResponseString(absl::string_view data) override;
