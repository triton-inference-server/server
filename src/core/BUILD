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

cc_library(
    name = "autofill_header",
    hdrs = ["autofill.h"],
    deps = [
        ":model_config",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/config:platform_config_proto",
    ],
)

cc_library(
    name = "autofill",
    srcs = ["autofill.cc"],
    deps = [
        ":autofill_header",
        ":constants",
        ":logging",
        ":model_config",
        ":model_config_proto",
        "//src/servables/caffe2:autofill",
        "//src/servables/tensorflow:autofill",
        "//src/servables/tensorrt:autofill",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "backend",
    hdrs = ["backend.h"],
    srcs = ["backend.cc"],
    deps = [
        ":constants",
        ":dynamic_batch_scheduler",
        ":label_provider",
        ":logging",
        ":metrics",
        ":model_config_proto",
        ":scheduler",
        ":sequence_batch_scheduler",
        ":model_config_utils",
        "@com_github_libevent_libevent//:libevent",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "provider_header",
    hdrs = ["provider.h"],
    deps = [
        ":api_proto",
        ":grpc_service_proto",
        "@com_github_libevent_libevent//:libevent",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "provider",
    srcs = ["provider.cc"],
    deps = [
        ":backend",
        ":constants",
        ":provider_header",
        ":logging",
        ":model_config",
        ":model_config_utils",
        "@com_github_libevent_libevent//:libevent",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "label_provider",
    srcs = ["label_provider.cc"],
    hdrs = ["label_provider.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        ":constants",
        ":logging",
        "@prometheus//core:core",
        "@prometheus//pull:pull",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
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

cc_library(
    name = "model_repository_manager",
    srcs = ["model_repository_manager.cc"],
    hdrs = ["model_repository_manager.h"],
    deps = [
        ":constants",
        ":logging",
        ":model_config",
        ":model_config_proto",
        ":model_config_utils",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/config:model_server_config_proto",
        "@tf_serving//tensorflow_serving/config:platform_config_proto",
    ],
)

cc_library(
    name = "profile",
    srcs = ["profile.cc"],
    hdrs = ["profile.h"],
    deps = [
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "scheduler",
    hdrs = ["scheduler.h"],
    deps = [
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "dynamic_batch_scheduler",
    srcs = ["dynamic_batch_scheduler.cc"],
    hdrs = ["dynamic_batch_scheduler.h"],
    deps = [
        ":api_proto",
        ":constants",
        ":provider_header",
        ":logging",
        ":model_config",
        ":model_config_proto",
        ":scheduler",
        ":server_status_header",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "sequence_batch_scheduler",
    srcs = ["sequence_batch_scheduler.cc"],
    hdrs = ["sequence_batch_scheduler.h"],
    deps = [
        ":constants",
        ":provider_header",
        ":logging",
        ":model_config",
        ":model_config_proto",
        ":scheduler",
        ":server_status_header",
        ":model_config_utils",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "server",
    srcs = ["server.cc"],
    deps = [
        ":api_proto",
        ":constants",
        ":grpc_server",
        ":http_server",
        ":logging",
        ":model_config",
        ":model_config_proto",
        ":model_repository_manager",
        ":profile",
        ":provider",
        ":request_status",
        ":request_status_proto",
        ":server_header",
        ":server_status_header",
        ":server_status_proto",
        ":model_config_utils",
        "//src/servables/caffe2:netdef_bundle_source_adapter",
        "//src/servables/tensorflow:graphdef_bundle_source_adapter",
        "//src/servables/tensorflow:savedmodel_bundle_source_adapter",
        "//src/servables/tensorrt:plan_bundle_source_adapter",
        "//src/servables/custom:custom_bundle_source_adapter",
        "//src/servables/ensemble:ensemble_bundle_source_adapter",
        "@com_github_libevent_libevent//:libevent",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/config:model_server_config_proto",
        "@tf_serving//tensorflow_serving/core:servable_state_monitor",
        "@tf_serving//tensorflow_serving/core:availability_preserving_policy",
        "@tf_serving//tensorflow_serving/model_servers:server_core",
    ],
)

cc_library(
    name = "server_header",
    hdrs = ["server.h"],
    deps = [
        ":api_proto",
        ":provider",
        ":model_config_proto",
        ":request_status_proto",
        ":server_status_header",
        ":server_status_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/model_servers:server_core",
    ],
)

cc_library(
    name = "http_server",
    srcs = ["http_server.cc"],
    hdrs = ["http_server.h"],
    deps = [
        ":constants",
        ":logging",
        ":provider",
        ":request_status",
        ":request_status_proto",
        ":server_header",
        "@com_github_libevhtp//:libevhtp",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        ":constants",
        ":grpc_service_proto",
        ":logging",
        ":provider",
        ":request_status",
        ":request_status_proto",
        ":server_header",
        "//src/nvrpc:nvrpc",
        "@grpc//:grpc++_unsecure",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "server_status_header",
    hdrs = ["server_status.h"],
    deps = [
        ":model_config_proto",
        ":model_repository_manager",
        ":server_status_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/core:servable_state_monitor",
    ],
)

cc_library(
    name = "server_status",
    srcs = ["server_status.cc"],
    deps = [
        ":backend",
        ":constants",
        ":provider_header",
        ":logging",
        ":metrics",
        ":server_status_header",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/core:servable_state",
    ],
)

cc_library(
    name = "request_status",
    srcs = ["request_status.cc"],
    hdrs = ["request_status.h"],
    deps = [
        ":request_status_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "model_config_utils",
    srcs = ["model_config_utils.cc"],
    hdrs = ["model_config_utils.h"],
    deps = [
        ":autofill",
        ":constants",
        ":logging",
        ":model_config",
        ":model_config_proto",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:lib",
        "@tf_serving//tensorflow_serving/config:platform_config_proto",
    ],
)

cc_library(
    name = "ensemble_utils",
    srcs = ["ensemble_utils.cc"],
    hdrs = ["ensemble_utils.h"],
    deps = [
        ":logging",
        ":model_config_proto",
        ":model_config_utils",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)