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

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

cc_library(
    name = "addsub_kernels",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
    copts = cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "addsub_base",
    srcs = ["addsub.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":addsub_kernels",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/backends/custom:custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libaddsub.so",
    deps = [
        ":addsub_base",
        ":addsub_kernels",
    ],
    linkshared = 1,
)
