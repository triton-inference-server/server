# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# libevhtp library.
# from https://github.com/criticalstack/libevhtp

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

include_files = [
    "libevhtp/include/evhtp.h",
    "libevhtp/include/internal.h",
    "libevhtp/include/numtoa.h",
    "libevhtp/include/evhtp/config.h",
    "libevhtp/include/evhtp/evhtp.h",
    "libevhtp/include/evhtp/log.h",
    "libevhtp/include/evhtp/parser.h",
    "libevhtp/include/evhtp/sslutils.h",
    "libevhtp/include/evhtp/thread.h",
]

lib_files = [
    "libevhtp/build/libevhtp.a",
]

genrule(
    name = "libevhtp-srcs",
    srcs = ["@com_github_libevent_libevent//:libevent-files"],
    outs = include_files + lib_files,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/libevhtp",
        "export PATH=$$(pwd)/$(@D)/../com_github_libevent_libevent/libevent:$$PATH",
        "cd $$(pwd)/external/com_github_libevhtp/build/",
        "cmake -D EVHTP_DISABLE_SSL:BOOL=ON -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON ..",
        "make",
        "mkdir -p $$INSTALL_DIR/build",
        "mkdir -p $$INSTALL_DIR/include",
        "cp libevhtp.a $$INSTALL_DIR/build/.",
        "cd ..",
        "cp -r include/ $$INSTALL_DIR/",
        "cp build/include/evhtp/config.h $$INSTALL_DIR/include/evhtp/.",
        #"rm -rf $$TMP_DIR",
    ]),
)

cc_library(
    name = "libevhtp",
    srcs = [
        "libevhtp/build/libevhtp.a",
    ],
    hdrs = include_files,
    includes = ["libevhtp/include"],
    linkstatic = 1,
)

filegroup(
    name = "libevhtp-files",
    srcs = include_files + lib_files,
)