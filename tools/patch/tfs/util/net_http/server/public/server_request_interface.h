--- tensorflow_serving/util/net_http/server/public/server_request_interface.h	2018-06-25 15:21:58.138237776 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/util/net_http/server/public/server_request_interface.h	2018-10-12 12:44:37.280572118 -0700
@@ -36,6 +36,8 @@
 
 #include "tensorflow_serving/util/net_http/server/public/response_code_enum.h"
 
+struct evbuffer;
+
 namespace tensorflow {
 namespace serving {
 namespace net_http {
@@ -61,6 +63,16 @@
   // Must be in Upper Case.
   virtual absl::string_view http_method() const = 0;
 
+  // Get the value for a query param. Return true if query param
+  // exists or false if it doesn't (in that case 'str' is undefined).
+  virtual bool QueryParam(absl::string_view key, std::string* str) const = 0;
+
+  // Get the evbuffer for the request body.
+  virtual evbuffer* InputBuffer() const = 0;
+
+  // Get the evbuffer for the response body.
+  virtual evbuffer* OutputBuffer() = 0;
+
   // Input/output byte-buffer types are subject to change!
   // I/O buffer choices:
   // - absl:ByteStream would work but it is not yet open-sourced
