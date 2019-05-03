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

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load('@com_google_protobuf//:protobuf.bzl', 'py_proto_library')

#
# C++, Python and GRPC protobuf-derived libraries
#

cc_library(
    name = "all_cc_protos",
    deps = [
        ":api_proto",
        ":grpc_service_proto",
        "model_config_proto",
        "request_status_proto",
        "server_status_proto",
    ],
)

cc_proto_library(
    name = "api_proto",
    srcs = ["api.proto"],
)

py_proto_library(
    name = "api_proto_py_pb2",
    srcs = ["api.proto"],
    srcs_version = "PY2AND3",
)

cc_proto_library(
    name = "grpc_service_proto",
    srcs = ["grpc_service.proto"],
    use_grpc_plugin = True,
    default_runtime="@com_google_protobuf//:protobuf",
    protoc="@com_google_protobuf//:protoc",
    deps = [
        ":api_proto",
        ":request_status_proto",
        ":server_status_proto",
    ],
)

py_proto_library(
    name = "grpc_service_proto_py_pb2",
    srcs = ["grpc_service.proto"],
    use_grpc_plugin = True,
    srcs_version = "PY2AND3",
    default_runtime="@com_google_protobuf//:protobuf_python",
    protoc="@com_google_protobuf//:protoc",
    deps = [
        ":api_proto_py_pb2",
        ":request_status_proto_py_pb2",
        ":server_status_proto_py_pb2",
    ],
)

cc_proto_library(
    name = "model_config_proto",
    srcs = ["model_config.proto"],
)

py_proto_library(
    name = "model_config_proto_py_pb2",
    srcs = ["model_config.proto"],
    srcs_version = "PY2AND3",
)

cc_proto_library(
    name = "request_status_proto",
    srcs = ["request_status.proto"],
)

py_proto_library(
    name = "request_status_proto_py_pb2",
    srcs = ["request_status.proto"],
    srcs_version = "PY2AND3",
)

cc_proto_library(
    name = "server_status_proto",
    srcs = ["server_status.proto"],
    deps = [
        ":model_config_proto",
    ],
)

py_proto_library(
    name = "server_status_proto_py_pb2",
    srcs = ["server_status.proto"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_config_proto_py_pb2",
    ],
)

#
# Simple dependencies required by clients, custom backends
#

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "model_config",
    srcs = ["model_config.cc"],
    hdrs = ["model_config.h"],
    deps = [
        ":constants",
        ":model_config_proto",
    ],
)

cc_library(
    name = "model_config_cuda",
    srcs = ["model_config_cuda.cc"],
    hdrs = ["model_config_cuda.h"],
    deps = [
        ":model_config_proto",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

#
# Inferface for in-process access to the server
#
cc_library(
    name = "request_inprocess_header",
    hdrs = ["request_inprocess.h"],
    deps = [
        ":constants",
        "//src/clients/c++:request",
    ],
)

#
# Server headers
#

cc_library(
    name = "server_header",
    hdrs = [
        "autofill.h",
        "backend.h",
        "constants.h",
        "dynamic_batch_scheduler.h",
        "ensemble_scheduler.h",
        "ensemble_utils.h",
        "filesystem.h",
        "label_provider.h",
        "logging.h",
        "metric_model_reporter.h",
        "metrics.h",
        "model_config.h",
        "model_config_cuda.h",
        "model_config_utils.h",
        "model_repository_manager.h",
        "profile.h",
        "provider.h",
        "provider_utils.h",
        "request_status.h",
        "scheduler.h",
        "sequence_batch_scheduler.h",
        "server.h",
        "server_status.h",
        "status.h",
    ],
    deps = [
        ":all_cc_protos",
        ":request_inprocess_header",
        "//src/servables/caffe2:autofill_header",
        "//src/servables/tensorflow:autofill_header",
        "//src/servables/tensorrt:autofill_header",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
        "@prometheus//core:core",
    ],
)

#
# Server
#

cc_library(
    name = "server",
    srcs = [
        "autofill.cc",
        "backend.cc",
        "dynamic_batch_scheduler.cc",
        "ensemble_scheduler.cc",
        "ensemble_utils.cc",
        "filesystem.cc",
        "label_provider.cc",
        "logging.cc",
        "metric_model_reporter.cc",
        "metrics.cc",
        "model_config_utils.cc",
        "model_repository_manager.cc",
        "profile.cc",
        "provider.cc",
        "provider_utils.cc",
        "request_inprocess.cc",
        "request_status.cc",
        "sequence_batch_scheduler.cc",
        "server.cc",
        "server_status.cc",
        "status.cc",
    ],
    deps = [
        ":all_cc_protos",
        ":constants",
        ":model_config",
        ":model_config_cuda",
        ":server_header",
        "//src/clients/c++:request",
        "//src/clients/c++:request_common",
        "//src/operations/tensorflow:all_custom_ops",
        "//src/servables/caffe2:autofill",
        "//src/servables/caffe2:netdef_backend_factory",
        "//src/servables/tensorflow:autofill",
        "//src/servables/tensorflow:graphdef_backend_factory",
        "//src/servables/tensorflow:savedmodel_backend_factory",
        "//src/servables/tensorrt:autofill",
        "//src/servables/tensorrt:plan_backend_factory",
        "//src/servables/custom:custom_backend_factory",
        "//src/servables/ensemble:ensemble_backend_factory",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/contrib:contrib_kernels",
        "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
        "@org_tensorflow//tensorflow/contrib/tensorrt:trt_engine_op_op_lib",
        "@org_tensorflow//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
        "@org_tensorflow//tensorflow/contrib/tensorrt:trt_shape_function",
        "@org_tensorflow//tensorflow/core:lib",
        "@prometheus//core:core",
        "@tf_serving//tensorflow_serving/config:model_server_config_proto",
        "@tf_serving//tensorflow_serving/core:servable_state_monitor",
        "@tf_serving//tensorflow_serving/core:availability_preserving_policy",
        "@tf_serving//tensorflow_serving/model_servers:server_core",
    ],
)

cc_binary(
    name = "libtrtserver.so",
    deps = [
        ":server",
        ":libtrtserver.ldscript"
    ],
    linkopts = [
        "-pthread",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
        "-Wl,--version-script", "$(location :libtrtserver.ldscript)"
    ],
    linkshared = 1,
)

cc_import(
    name = "libtrtserver_import",
    shared_library = ":libtrtserver.so",
    hdrs = [
        "autofill.h",
        "backend.h",
        "constants.h",
        "dynamic_batch_scheduler.h",
        "ensemble_scheduler.h",
        "ensemble_utils.h",
        "filesystem.h",
        "label_provider.h",
        "logging.h",
        "metric_model_reporter.h",
        "metrics.h",
        "model_config.h",
        "model_config_cuda.h",
        "model_config_utils.h",
        "model_repository_manager.h",
        "profile.h",
        "provider.h",
        "provider_utils.h",
        "request_status.h",
        "scheduler.h",
        "sequence_batch_scheduler.h",
        "server.h",
        "server_status.h",
        "status.h",
    ],
)
