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

workspace(name = "inference_server")

local_repository(
  name = "org_tensorflow",
  path = "/opt/tensorflow/tensorflow-source",
)

local_repository(
  name = "tf_serving",
  path = __workspace_dir__ + "/serving/",
)

new_local_repository(
    name = "extern_lib",
    path = "/opt/tensorrtserver/lib",
    build_file_content = """
cc_library(
    name = "libcaffe2",
    srcs = ["libcaffe2.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libcaffe2_gpu",
    srcs = ["libcaffe2_gpu.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libcaffe2_detectron_ops_gpu",
    srcs = ["libcaffe2_detectron_ops_gpu.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libc10",
    srcs = ["libc10.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libc10_cuda",
    srcs = ["libc10_cuda.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_core",
    srcs = ["libmkl_core.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_gnu_thread",
    srcs = ["libmkl_gnu_thread.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_avx2",
    srcs = ["libmkl_avx2.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_def",
    srcs = ["libmkl_def.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_intel_lp64",
    srcs = ["libmkl_intel_lp64.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_rt",
    srcs = ["libmkl_rt.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libmkl_vml_def",
    srcs = ["libmkl_vml_def.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libonnxruntime",
    srcs = ["libonnxruntime.so"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "libtorch",
    srcs = ["libtorch.so"],
    visibility = ["//visibility:public"],
)
""",
)

# Need prometheus for metrics
http_archive(
    name = "prometheus",
    strip_prefix = "prometheus-cpp-0.5.0",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/v0.5.0.tar.gz"],
)
load("@prometheus//:repositories.bzl", "load_civetweb")
load_civetweb()

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

new_http_archive(
    name = "com_github_libevhtp",
    urls = [
        "https://github.com/criticalstack/libevhtp/archive/1.2.18.zip",
    ],
    sha256 = "3194dc6eb4e8d6aa1e7dd3dc60bfbe066f38f9a0b5881463f0e149badd82a7bb",
    strip_prefix = "libevhtp-1.2.18",
    build_file = "third_party/libevhtp.BUILD",
)

load('@tf_serving//tensorflow_serving:workspace.bzl', 'tf_serving_workspace')
tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.18.0")
