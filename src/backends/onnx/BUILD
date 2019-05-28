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

load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "onnx_backend_proto",
    srcs = ["onnx_backend.proto"],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":onnx_utils",
        "//src/core:server_header",
    ],
)

cc_library(
    name = "onnx_utils",
    srcs = ["onnx_utils.cc"],
    hdrs = ["onnx_utils.h"],
    deps = [
        "//src/core:model_config",
        "//src/core:server_header",
    ],
)

cc_library(
    name = "onnx_backend",
    srcs = ["onnx_backend.cc"],
    hdrs = ["onnx_backend.h"],
    deps = [
        ":loader",
        ":onnx_utils",
        "//src/core:constants",
        "//src/core:model_config_proto",
        "//src/core:model_config",
        "//src/core:server_header",
        "@local_config_cuda//cuda:cuda_headers"
    ],
)

cc_library(
    name = "onnx_backend_factory",
    srcs = ["onnx_backend_factory.cc"],
    hdrs = ["onnx_backend_factory.h"],
    deps = [
        ":loader",
        ":onnx_backend",
        ":onnx_backend_proto",
        "//src/core:constants",
        "//src/core:model_config",
        "//src/core:model_config_proto",
        "//src/core:server_header",
    ],
    alwayslink = 1,
)

filegroup(
    name = "onnx_testdata",
    srcs = glob(
        ["testdata/**/*"]),
)

cc_test(
    name = "onnx_backend_test",
    srcs = ["onnx_backend_test.cc"],
    linkopts = [
        "-L/opt/tensorrtserver/lib",
        "-lcaffe2_gpu",
        "-lcaffe2",
        "-ltorch",
        "-lonnxruntime",
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
        ":onnx_testdata"
    ],
    deps = [
        "//src/test:model_config_test_base",
        "//src/core:server",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)
