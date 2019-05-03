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

filegroup(
    name = "test_testdata",
    srcs = glob(
        ["testdata/**/*"]),
)

cc_library(
    name = "model_config_test_base",
    testonly = 1,
    hdrs = ["model_config_test_base.h"],
    srcs = ["model_config_test_base.cc"],
    data = [
        ":test_testdata",
    ],
    deps = [
        "//src/core:server_header",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "testmain",
    testonly = 1,
    srcs = ["testmain.cc"],
    deps = [
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "caffe2plan_main",
    srcs = ["caffe2plan.cc"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "caffe2plan",
    deps = [
        ":caffe2plan_main",
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = [
        "-lnvinfer",
        "-lnvcaffe_parser",
    ],
)
