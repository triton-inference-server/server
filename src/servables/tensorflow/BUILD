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
    name = "graphdef_bundle_proto",
    srcs = ["graphdef_bundle.proto"],
    cc_api_version = 2,
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

serving_proto_library(
    name = "savedmodel_bundle_proto",
    srcs = ["savedmodel_bundle.proto"],
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
        ":tf_utils",
        "//src/core:autofill_header",
        "//src/core:logging",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tf_utils",
    srcs = ["tf_utils.cc"],
    hdrs = ["tf_utils.h"],
    deps = [
        "//src/core:model_config",
        "//src/core:model_config_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        "//src/core:logging",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
    ],
)

cc_library(
    name = "base_bundle",
    srcs = ["base_bundle.cc"],
    hdrs = ["base_bundle.h"],
    deps = [
        ":tf_utils",
        "//src/core:constants",
        "//src/core:infer",
        "//src/core:label_provider",
        "//src/core:logging",
        "//src/core:model_config_proto",
        "//src/core:server_status",
        "//src/core:utils",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "graphdef_bundle",
    srcs = ["graphdef_bundle.cc"],
    hdrs = ["graphdef_bundle.h"],
    deps = [
        ":base_bundle",
        ":tf_utils",
        "//src/core:model_config",
    ],
)

cc_library(
    name = "graphdef_bundle_source_adapter",
    srcs = ["graphdef_bundle_source_adapter.cc"],
    hdrs = ["graphdef_bundle_source_adapter.h"],
    deps = [
        ":graphdef_bundle",
        ":graphdef_bundle_proto",
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

cc_library(
    name = "savedmodel_bundle",
    srcs = ["savedmodel_bundle.cc"],
    hdrs = ["savedmodel_bundle.h"],
    deps = [
        ":base_bundle",
        ":loader",
        ":tf_utils",
        "//src/core:model_config",
    ],
)

cc_library(
    name = "savedmodel_bundle_source_adapter",
    srcs = ["savedmodel_bundle_source_adapter.cc"],
    hdrs = ["savedmodel_bundle_source_adapter.h"],
    deps = [
        ":savedmodel_bundle",
        ":savedmodel_bundle_proto",
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
    name = "tf_testdata",
    srcs = glob(
        ["testdata/**/*"]),
)

cc_test(
    name = "graphdef_bundle_test",
    srcs = ["graphdef_bundle_test.cc"],
    linkopts = [
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
         ":tf_testdata"
    ],
    deps = [
        ":graphdef_bundle",
        "//src/core:constants",
        "//src/test:model_config_test_base",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)

cc_test(
    name = "savedmodel_bundle_test",
    srcs = ["savedmodel_bundle_test.cc"],
    linkopts = [
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
         ":tf_testdata"
    ],
    deps = [
        ":savedmodel_bundle",
        "//src/core:constants",
        "//src/test:model_config_test_base",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)
