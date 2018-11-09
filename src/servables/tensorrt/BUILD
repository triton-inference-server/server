# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

load('@tf_serving//tensorflow_serving:serving.bzl', 'serving_proto_library')

serving_proto_library(
    name = "plan_bundle_proto",
    srcs = ["plan_bundle.proto"],
    cc_api_version = 2,
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

cc_library(
    name = "autofill",
    srcs = ["autofill.cc"],
    hdrs = ["autofill.h"],
    deps = [
        ":loader",
        ":plan_utils",
        "//src/core:autofill_header",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":logging",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "//src/core:logging",
    ],
)

cc_library(
    name = "plan_utils",
    srcs = ["plan_utils.cc"],
    hdrs = ["plan_utils.h"],
    deps = [
        "//src/core:model_config",
    ],
)

cc_library(
    name = "plan_bundle",
    srcs = ["plan_bundle.cc"],
    hdrs = ["plan_bundle.h"],
    deps = [
        ":loader",
        ":plan_utils",
        "//src/core:constants",
        "//src/core:infer",
        "//src/core:label_provider",
        "//src/core:model_config_proto",
        "//src/core:server_status",
        "//src/core:utils",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "plan_bundle_source_adapter",
    srcs = ["plan_bundle_source_adapter.cc"],
    hdrs = ["plan_bundle_source_adapter.h"],
    deps = [
        ":plan_bundle",
        ":plan_bundle_proto",
        "//src/core:constants",
        "//src/core:logging",
        "//src/core:model_config",
        "//src/core:model_config_proto",
        "//src/core:utils",
        "@tf_serving//tensorflow_serving/core:loader",
        "@tf_serving//tensorflow_serving/core:simple_loader",
        "@tf_serving//tensorflow_serving/core:source_adapter",
        "@tf_serving//tensorflow_serving/core:storage_path",
        "@tf_serving//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

filegroup(
    name = "tensorrt_testdata",
    srcs = glob(
        ["testdata/**/*"]),
)

cc_test(
    name = "plan_bundle_test",
    srcs = ["plan_bundle_test.cc"],
    linkopts = [
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime"
    ],
    data = [
        ":tensorrt_testdata"
    ],
    deps = [
        ":plan_bundle",
        "//src/core:constants",
        "//src/test:model_config_test_base",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)
