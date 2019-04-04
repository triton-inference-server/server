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

cc_library(
    name = "request_header",
    hdrs = ["request.h"],
    deps = [
        "//src/core:api_proto",
        "//src/core:request_status_proto",
        "//src/core:server_status_proto",
    ],
)

cc_library(
    name = "request",
    srcs = ["request.cc"],
    deps = [
        ":request_header",
    ],
)

cc_library(
    name = "request_common",
    hdrs = ["request_common.h"],
    srcs = ["request_common.cc"],
    deps = [
        ":request",
        "//src/core:constants",
        "//src/core:model_config",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "request_grpc",
    hdrs = ["request_grpc.h"],
    srcs = ["request_grpc.cc"],
    deps = [
        ":request_header",
        ":request_common",
        "//src/core:grpc_service_proto",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "request_http",
    srcs = ["request_http.cc"],
    hdrs = ["request_http.h"],
    deps = [
        ":request_header",
        ":request_common",
    ],
)

cc_library(
    name = "image_client_main",
    srcs = ["image_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "perf_client_main",
    srcs = ["perf_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:constants",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "simple_client_main",
    srcs = ["simple_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "simple_string_client_main",
    srcs = ["simple_string_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "ensemble_image_client_main",
    srcs = ["ensemble_image_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "simple_sequence_client_main",
    srcs = ["simple_sequence_client.cc"],
    deps = [
        ":request_grpc",
        ":request_http",
        "//src/core:model_config_proto",
    ],
)

cc_library(
    name = "simple_inprocess_main",
    srcs = ["simple_inprocess.cc"],
    deps = [
        "//src/core:request_inprocess_header",
    ],
)

cc_binary(
    name = "image_client",
    deps = [
        ":image_client_main",
    ],
    linkopts = [
        "-pthread",
        "-lcurl", "-lz",
        "-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui"
    ],
)

cc_binary(
    name = "perf_client",
    deps = [
        ":perf_client_main",
    ],
    linkopts = [
        "-pthread", "-lcurl"
    ],
)

cc_binary(
    name = "simple_client",
    deps = [
        ":simple_client_main",
    ],
    linkopts = [
        "-pthread", "-lcurl"
    ],
)

cc_binary(
    name = "simple_string_client",
    deps = [
        ":simple_string_client_main",
    ],
    linkopts = [
        "-pthread", "-lcurl"
    ],
)

cc_binary(
    name = "ensemble_image_client",
    deps = [
        ":ensemble_image_client_main",
    ],
    linkopts = [
        "-pthread", "-lcurl"
    ],
)

cc_binary(
    name = "simple_sequence_client",
    deps = [
        ":simple_sequence_client_main",
    ],
    linkopts = [
        "-pthread", "-lcurl"
    ],
)

cc_binary(
    name = "simple_inprocess",
    deps = [
        ":simple_inprocess_main",
        "//src/core:libtrtserver_import",
        "@extern_lib//:libcaffe2",                   # bazel is stupid...
        "@extern_lib//:libcaffe2_detectron_ops_gpu", # putting these on libtrtserver doesn't work
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
    ],
    linkopts = [
        "-pthread",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime"
    ],
)
