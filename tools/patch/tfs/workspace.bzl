--- tensorflow_serving/workspace.bzl	2018-07-13 13:15:37.431961006 -0700
+++ /home/david/dev/gitlab/dgx/tensorrtserver/tools/patch/tfs/workspace.bzl	2018-10-12 12:44:37.280572118 -0700
@@ -45,7 +45,7 @@
       ],
       sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
       strip_prefix = "rapidjson-1.1.0",
-      build_file = "third_party/rapidjson.BUILD"
+      build_file = "serving/third_party/rapidjson.BUILD"
   )
 
   # ===== libevent (libevent.org) dependencies =====
@@ -56,5 +56,5 @@
       ],
       sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
       strip_prefix = "libevent-release-2.1.8-stable",
-      build_file = "third_party/libevent.BUILD"
+      build_file = "serving/third_party/libevent.BUILD"
   )
