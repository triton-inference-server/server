# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package(
    default_visibility = ["//visibility:public"],
)

#
# HTTP service endpoint
#
cc_library(
    name = "http_server",
    srcs = ["http_server.cc"],
    hdrs = ["http_server.h"],
    deps = [
        "//src/core:all_cc_protos",
        "//src/core:libtrtserver_import",
        "@com_github_libevhtp//:libevhtp",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

#
# GRPC service endpoint
#
cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        "//src/core:all_cc_protos",
        "//src/core:libtrtserver_import",
        "//src/nvrpc:nvrpc",
        "@com_github_libevent_libevent//:libevent", # FIXME, because provider_utils mixes HTTP/GRPC
        "@grpc//:grpc++_unsecure",
    ],
)

#
# Inference server
#
cc_binary(
    name = "trtserver",
    srcs = ["main.cc"],
    deps = [
        ":grpc_server",
        ":http_server",
        "//src/core:libtrtserver_import",
        "@extern_lib//:libcaffe2",
        "@extern_lib//:libcaffe2_detectron_ops_gpu",
        "@extern_lib//:libcaffe2_gpu",
        "@extern_lib//:libc10",
        "@extern_lib//:libc10_cuda",
        "@extern_lib//:libmkl_core",
        "@extern_lib//:libmkl_gnu_thread",
        "@extern_lib//:libmkl_avx2",
        "@extern_lib//:libmkl_def",
        "@extern_lib//:libmkl_intel_lp64",
        "@extern_lib//:libmkl_rt",
        "@extern_lib//:libmkl_vml_def",
        "@extern_lib//:libonnxruntime",
        "@extern_lib//:libtorch",
        "@prometheus//pull:pull",
    ],
    linkopts = [
        "-pthread",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime"
    ],
)
