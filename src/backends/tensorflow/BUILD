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
    name = "graphdef_bundle_proto",
    srcs = ["graphdef_bundle.proto"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_proto_library(
    name = "savedmodel_bundle_proto",
    srcs = ["savedmodel_bundle.proto"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "autofill_header",
    hdrs = [
        "autofill.h",
    ],
    deps = [
        "//src/core:model_config",
    ],
)

cc_library(
    name = "autofill",
    srcs = ["autofill.cc"],
    deps = [
        ":autofill_header",
        ":loader",
        ":savedmodel_bundle_proto",
        ":tf_utils",
        "//src/core:server_header",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
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
        "//src/core:server_header",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        "//src/core:server_header",
        "@org_tensorflow//tensorflow/c:c_api",
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
        "//src/core:model_config_proto",
        "//src/core:server_header",
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
        "//src/core:server_header",
    ],
)

cc_library(
    name = "graphdef_backend_factory",
    srcs = ["graphdef_backend_factory.cc"],
    hdrs = ["graphdef_backend_factory.h"],
    deps = [
        ":graphdef_bundle",
        ":graphdef_bundle_proto",
        "//src/core:constants",
        "//src/core:model_config",
        "//src/core:model_config_proto",
        "//src/core:server_header",
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
        "//src/core:server_header",
    ],
)

cc_library(
    name = "savedmodel_backend_factory",
    srcs = ["savedmodel_backend_factory.cc"],
    hdrs = ["savedmodel_backend_factory.h"],
    deps = [
        ":savedmodel_bundle",
        ":savedmodel_bundle_proto",
        "//src/core:constants",
        "//src/core:model_config",
        "//src/core:model_config_proto",
        "//src/core:server_header",
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
        "-L/opt/tensorrtserver/lib",
        "-lcaffe2_gpu",
        "-lcaffe2",
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
         ":tf_testdata"
    ],
    deps = [
        "//src/test:model_config_test_base",
        "//src/core:server",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)

cc_test(
    name = "savedmodel_bundle_test",
    srcs = ["savedmodel_bundle_test.cc"],
    linkopts = [
        "-L/opt/tensorrtserver/lib",
        "-lcaffe2_gpu",
        "-lcaffe2",
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
         ":tf_testdata"
    ],
    deps = [
        "//src/test:model_config_test_base",
        "//src/core:server",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)
