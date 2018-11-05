--- tensorflow_serving/util/net_http/server/internal/evhttp_request.cc	2018-11-01 12:04:22.185086318 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/util/net_http/server/internal/evhttp_request.cc	2018-11-02 11:47:57.254526547 -0700
@@ -44,6 +44,8 @@
     evhttp_uri_free(decoded_uri);
   }
 
+  evhttp_clear_headers(&params);
+
   if (request && evhttp_request_is_owned(request)) {
     evhttp_request_free(request);
   }
@@ -98,6 +100,8 @@
     path = "/";
   }
 
+  evhttp_parse_query(uri, &params);
+
   headers = evhttp_request_get_input_headers(request);
 
   return true;
@@ -123,6 +127,26 @@
   return parsed_request_->method;
 }
 
+bool EvHTTPRequest::QueryParam(absl::string_view key, std::string* str) const {
+  std::string key_str(key.data(), key.size());
+  const char* val =
+    evhttp_find_header(&parsed_request_->params, key_str.c_str());
+  if (val != nullptr) {
+    *str = val;
+    return true;
+  }
+
+  return false;
+}
+
+evbuffer* EvHTTPRequest::InputBuffer() const {
+  return evhttp_request_get_input_buffer(parsed_request_->request);
+}
+
+evbuffer* EvHTTPRequest::OutputBuffer() {
+  return output_buf;
+}
+
 bool EvHTTPRequest::Initialize() {
   output_buf = evbuffer_new();
   return output_buf != nullptr;
