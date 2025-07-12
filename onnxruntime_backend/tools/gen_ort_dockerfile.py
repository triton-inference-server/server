#!/usr/bin/env python3
# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import platform
import re

FLAGS = None

ORT_TO_TRTPARSER_VERSION_MAP = {
    "1.9.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
    "1.10.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
}

OPENVINO_VERSION_MAP = {
    "2024.0.0": (
        "2024.0",  # OpenVINO short version
        "2024.0.0.14509.34caeefd078",  # OpenVINO version with build number
    ),
    "2024.1.0": (
        "2024.1",  # OpenVINO short version
        "2024.1.0.15008.f4afc983258",  # OpenVINO version with build number
    ),
    "2024.4.0": (
        "2024.4",  # OpenVINO short version
        "2024.4.0.16579.c3152d32c9c",  # OpenVINO version with build number
    ),
    "2024.5.0": (
        "2024.5",  # OpenVINO short version
        "2024.5.0.17288.7975fa5da0c",  # OpenVINO version with build number
    ),
    "2025.0.0": (
        "2025.0",  # OpenVINO short version
        "2025.0.0.17942.1f68be9f594",  # OpenVINO version with build number
    ),
    "2025.1.0": (
        "2025.1",  # OpenVINO short version
        "2025.1.0.18503.6fec06580ab",  # OpenVINO version with build number
    ),
    "2025.2.0": (
        "2025.2",  # OpenVINO short version
        "2025.2.0.19140.c01cd93e24d",  # OpenVINO version with build number
    ),
}


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def dockerfile_common():
    df = """
ARG BASE_IMAGE={}
ARG ONNXRUNTIME_VERSION={}
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BUILD_CONFIG={}
""".format(
        FLAGS.triton_container, FLAGS.ort_version, FLAGS.ort_build_config
    )

    if FLAGS.ort_openvino is not None:
        df += """
ARG ONNXRUNTIME_OPENVINO_VERSION={}
""".format(
            FLAGS.ort_openvino
        )

    df += """
FROM ${BASE_IMAGE}
WORKDIR /workspace
"""
    return df


def dockerfile_for_linux(output_file):
    df = dockerfile_common()
    df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# The Onnx Runtime dockerfile is the collection of steps in
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles

"""
    # Consider moving rhel logic to its own function e.g., dockerfile_for_rhel
    # if the changes become more substantial.
    if target_platform() == "rhel":
        df += """
# The manylinux container defaults to Python 3.7, but some feature installation
# requires a higher version.
ARG PYVER=3.12
ENV PYTHONPATH=/opt/python/v
RUN ln -sf /opt/python/cp${PYVER/./}* ${PYTHONPATH}

ENV PYBIN=${PYTHONPATH}/bin
ENV PYTHON_BIN_PATH=${PYBIN}/python${PYVER} \
    PATH=${PYBIN}:${PATH}

RUN yum install -y \
        wget \
        zip \
        ca-certificates \
        curl \
        python3-pip \
        git \
        gnupg \
        gnupg1 \
        openssl-devel

RUN pip3 install patchelf==0.17.2
"""
    else:
        if os.getenv("CCACHE_REMOTE_ONLY") and os.getenv("CCACHE_REMOTE_STORAGE"):
            df += """
ENV CCACHE_REMOTE_ONLY="true" \\
    CCACHE_REMOTE_STORAGE="{}" \\
    CMAKE_CXX_COMPILER_LAUNCHER="ccache" \\
    CMAKE_C_COMPILER_LAUNCHER="ccache" \\
    CMAKE_CUDA_COMPILER_LAUNCHER="ccache" \\
    VERBOSE=1

RUN apt-get update \\
      && apt-get install -y --no-install-recommends ccache && ccache -p \\
      && rm -rf /var/lib/apt/lists/*
""".format(
                os.getenv("CCACHE_REMOTE_STORAGE")
            )

        df += """

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev \
        python3-pip \
        git \
        gnupg \
        gnupg1

RUN pip3 install patchelf==0.17.2

# Install dependencies from
# onnxruntime/dockerfiles/scripts/install_common_deps.sh.
RUN apt update -q=2 \\
    && apt install -y gpg wget \\
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \\
    && . /etc/os-release \\
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \\
    && apt-get update -q=2 \\
    && apt-get install -y --no-install-recommends cmake=3.28.3* cmake-data=3.28.3* \\
    && cmake --version

"""

    if FLAGS.ort_openvino is not None:
        df += """
# Install OpenVINO
ARG ONNXRUNTIME_OPENVINO_VERSION
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}
"""
        df += """
ARG OPENVINO_SHORT_VERSION={}
ARG OPENVINO_VERSION_WITH_BUILD_NUMBER={}
""".format(
            OPENVINO_VERSION_MAP[FLAGS.ort_openvino][0],
            OPENVINO_VERSION_MAP[FLAGS.ort_openvino][1],
        )

        # Openvino changed the filename of the toolkit in 2025.0.0 so we need to detect this for
        # the release we want to install
        openvino_folder_name = "UNKNOWN_FOLDER_NAME"
        openvino_toolkit_filename = "UNKNOWN_FILENAME"
        if OPENVINO_VERSION_MAP[FLAGS.ort_openvino][0].split(".")[0] >= "2025":
            openvino_folder_name = (
                "openvino_toolkit_ubuntu24_${OPENVINO_VERSION_WITH_BUILD_NUMBER}_x86_64"
            )
            openvino_toolkit_filename = openvino_folder_name + ".tgz"
        else:
            openvino_folder_name = "l_openvino_toolkit_ubuntu24_${OPENVINO_VERSION_WITH_BUILD_NUMBER}_x86_64"
            openvino_toolkit_filename = openvino_folder_name + ".tgz"

        df += """
# Step 1: Download and install core components
# Ref: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html#step-1-download-and-install-the-openvino-core-components
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/${{OPENVINO_SHORT_VERSION}}/linux/{} --output openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    tar -xf openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    mkdir -p ${{INTEL_OPENVINO_DIR}} && \
    mv {}/* ${{INTEL_OPENVINO_DIR}} && \
    rm openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    (cd ${{INTEL_OPENVINO_DIR}}/install_dependencies && \
        ./install_openvino_dependencies.sh -y) && \
    ln -s ${{INTEL_OPENVINO_DIR}} ${{INTEL_OPENVINO_DIR}}/../openvino_`echo ${{ONNXRUNTIME_OPENVINO_VERSION}} | awk '{{print substr($0,0,4)}}'`

# Step 2: Configure the environment
# Ref: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html#step-2-configure-the-environment
ENV OpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
ENV LD_LIBRARY_PATH=$INTEL_OPENVINO_DIR/runtime/lib/intel64:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=$INTEL_OPENVINO_DIR/runtime/lib/intel64/pkgconfig
ENV PYTHONPATH=$INTEL_OPENVINO_DIR/python/python3.12:$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH
""".format(
            openvino_toolkit_filename, openvino_folder_name
        )

    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    if FLAGS.ort_version == "1.8.1":
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

RUN git clone -b tensorrt-8.0 --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
       """
    # Use the tensorrt-8.5ea branch to use Tensor RT 8.5a to use the built-in tensorrt parser
    elif FLAGS.ort_version == "1.12.1":
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

RUN git clone -b tensorrt-8.5ea --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
       """
    else:
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

# Cherry-pick commit: https://github.com/microsoft/onnxruntime/commit/9dad9af9f9b48c05814d0c2d067d0565e8da6ce8
RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime \\
    && cd onnxruntime \\
    && git cherry-pick -n 9dad9af9f9b48c05814d0c2d067d0565e8da6ce8 \\
    && sed -i 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt
        """

    if FLAGS.onnx_tensorrt_tag != "":
        df += """
    RUN (cd /workspace/onnxruntime/cmake/external/onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
            FLAGS.onnx_tensorrt_tag
        )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        elif target_platform() == "igpu":
            ep_flags += ' --cudnn_home "/usr/include"'
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.ort_version >= "1.12.1":
                ep_flags += " --use_tensorrt_builtin_parser"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)

    if os.name == "posix":
        if os.getuid() == 0:
            ep_flags += " --allow_running_as_root"

    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU"

    if target_platform() == "igpu":
        ep_flags += (
            " --skip_tests --cmake_extra_defines 'onnxruntime_BUILD_UNIT_TESTS=OFF'"
        )
        if os.getenv("CUDA_ARCH_LIST") is not None:
            print(f"[INFO] Defined CUDA_ARCH_LIST: {os.getenv('CUDA_ARCH_LIST')}")
            cuda_archs = (
                os.getenv("CUDA_ARCH_LIST")
                .replace("PTX", "")
                .replace(" ", "-real;")
                .replace(".", "")
            )
            cuda_archs = re.sub(r"-real;$", "", cuda_archs)
            print(f"[INFO] Set ONNX Runtime to use CUDA architectures to: {cuda_archs}")
        else:
            cuda_archs = "87"
    else:
        if os.getenv("CUDA_ARCH_LIST") is not None:
            print(f"[INFO] Defined CUDA_ARCH_LIST: {os.getenv('CUDA_ARCH_LIST')}")
            cuda_archs = (
                os.getenv("CUDA_ARCH_LIST")
                .replace("PTX", "")
                .replace(" ", "-real;")
                .replace(".", "")
            )
            cuda_archs = re.sub(r"-real;$", "", cuda_archs)
            print(f"[INFO] Set ONNX Runtime to use CUDA architectures to: {cuda_archs}")
        else:
            cuda_archs = "75;80;86;90;100;120"

    df += """
WORKDIR /workspace/onnxruntime
ARG COMMON_BUILD_ARGS="--config ${{ONNXRUNTIME_BUILD_CONFIG}} --skip_submodule_sync --parallel --build_shared_lib \
    --compile_no_warning_as_error --build_dir /workspace/build --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='{}'  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 "
""".format(
        cuda_archs
    )

    df += """
RUN ./build.sh ${{COMMON_BUILD_ARGS}} --update --build {}
""".format(
        ep_flags
    )

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime

RUN mkdir -p /opt/onnxruntime && \
    cp /workspace/onnxruntime/LICENSE /opt/onnxruntime && \
    cat /workspace/onnxruntime/cmake/external/onnx/VERSION_NUMBER > /opt/onnxruntime/ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
RUN mkdir -p /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
       /opt/onnxruntime/include

RUN mkdir -p /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_shared.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime.so \
       /opt/onnxruntime/lib
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/bin
"""
    else:
        df += """
RUN mkdir -p /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnxruntime_perf_test \
       /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnx_test_runner \
       /opt/onnxruntime/bin && \
    (cd /opt/onnxruntime/bin && chmod a+x *)
"""

    if FLAGS.enable_gpu:
        df += """
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_cuda.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_tensorrt.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_openvino is not None:
        df += """
# OpenVino specific headers and libraries
RUN cp -r ${INTEL_OPENVINO_DIR}/docs/licensing /opt/onnxruntime/LICENSE.openvino

RUN cp /workspace/onnxruntime/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h \
       /opt/onnxruntime/include

RUN apt-get update && apt-get install -y --no-install-recommends libtbb12

RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_openvino.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp /usr/lib/x86_64-linux-gnu/libtbb.so.* /opt/onnxruntime/lib

RUN OV_SHORT_VERSION=`echo ${ONNXRUNTIME_OPENVINO_VERSION} | awk '{ split($0,a,"."); print substr(a[1],3) a[2] a[3] }'` && \
    (cd /opt/onnxruntime/lib && \
        chmod a-x * && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so)
"""
    # Linking compiled ONNX Runtime libraries to their corresponding versioned libraries
    df += """
RUN cd /opt/onnxruntime/lib \
        && ln -s libonnxruntime.so libonnxruntime.so.1 \
        && ln -s libonnxruntime.so.1 libonnxruntime.so.${ONNXRUNTIME_VERSION}
"""
    df += """
RUN cd /opt/onnxruntime/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# For testing copy ONNX custom op library and model
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/test
"""
    else:
        df += """
RUN mkdir -p /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libcustom_op_library.so \
       /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/testdata/custom_op_library/custom_op_test.onnx \
       /opt/onnxruntime/test
"""

    with open(output_file, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()

    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    if FLAGS.ort_version == "1.8.1":
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO

RUN git clone -b tensorrt-8.0 --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
"""
    else:
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
RUN git clone -b rel-%ONNXRUNTIME_VERSION% --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    cd onnxruntime && git submodule update --init --recursive
"""

    if FLAGS.onnx_tensorrt_tag != "":
        df += """
    RUN (cd \\workspace\\onnxruntime\\cmake\\external\\onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
            FLAGS.onnx_tensorrt_tag
        )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)
    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU"

    df += """
WORKDIR /workspace/onnxruntime
ARG VS_DEVCMD_BAT="\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
RUN powershell Set-Content 'build.bat' -value 'call %VS_DEVCMD_BAT%',(Get-Content 'build.bat')
RUN build.bat --cmake_generator "Visual Studio 17 2022" --config Release --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=75;80;86;90;100;120" --skip_submodule_sync --parallel --build_shared_lib --compile_no_warning_as_error --skip_tests --update --build --build_dir /workspace/build {}
""".format(
        ep_flags
    )

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime
RUN copy \\workspace\\onnxruntime\\LICENSE \\opt\\onnxruntime
RUN copy \\workspace\\onnxruntime\\cmake\\external\\onnx\\VERSION_NUMBER \\opt\\onnxruntime\\ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_c_api.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_session_options_config_keys.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\cpu\\cpu_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_perf_test.exe \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnx_test_runner.exe \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.lib \\opt\\onnxruntime\\lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.lib \\opt\\onnxruntime\\lib
"""

    if FLAGS.enable_gpu:
        df += """
WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.lib \\opt\\onnxruntime\\lib
WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.dll \\opt\\onnxruntime\\bin
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.dll \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.lib \\opt\\onnxruntime\\lib
"""
    with open(output_file, "w") as dfile:
        dfile.write(df)


def preprocess_gpu_flags():
    if target_platform() == "windows":
        # Default to CUDA based on CUDA_PATH envvar and TensorRT in
        # C:/tensorrt
        if "CUDA_PATH" in os.environ:
            if FLAGS.cuda_home is None:
                FLAGS.cuda_home = os.environ["CUDA_PATH"]
            elif FLAGS.cuda_home != os.environ["CUDA_PATH"]:
                print("warning: --cuda-home does not match CUDA_PATH envvar")

        if FLAGS.cudnn_home is None:
            FLAGS.cudnn_home = FLAGS.cuda_home

        version = None
        m = re.match(r".*v([1-9]?[0-9]+\.[0-9]+)$", FLAGS.cuda_home)
        if m:
            version = m.group(1)

        if FLAGS.cuda_version is None:
            FLAGS.cuda_version = version
        elif FLAGS.cuda_version != version:
            print("warning: --cuda-version does not match CUDA_PATH envvar")

        if (FLAGS.cuda_home is None) or (FLAGS.cuda_version is None):
            print("error: windows build requires --cuda-version and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = "/tensorrt"
    else:
        if "CUDNN_VERSION" in os.environ:
            if FLAGS.cudnn_home is None:
                FLAGS.cudnn_home = "/usr"

        if FLAGS.cuda_home is None:
            FLAGS.cuda_home = "/usr/local/cuda"

        if (FLAGS.cuda_home is None) or (FLAGS.cudnn_home is None):
            print("error: linux build requires --cudnn-home and --cuda-home")

        if FLAGS.tensorrt_home is None:
            if target_platform() == "rhel":
                if platform.machine().lower() == "aarch64":
                    FLAGS.tensorrt_home = "/usr/local/cuda/targets/sbsa-linux/"
                else:
                    FLAGS.tensorrt_home = "/usr/local/cuda/targets/x86_64-linux/"
            else:
                FLAGS.tensorrt_home = "/usr/src/tensorrt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-container",
        type=str,
        required=True,
        help="Triton base container to use for ORT build.",
    )
    parser.add_argument("--ort-version", type=str, required=True, help="ORT version.")
    parser.add_argument(
        "--output", type=str, required=True, help="File to write Dockerfile to."
    )
    parser.add_argument(
        "--enable-gpu", action="store_true", required=False, help="Enable GPU support"
    )
    parser.add_argument(
        "--ort-build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
        help="ORT build configuration.",
    )
    parser.add_argument(
        "--target-platform",
        required=False,
        default=None,
        help='Target for build, can be "linux", "windows", "rhel", or "igpu". If not specified, build targets the current platform.',
    )

    parser.add_argument(
        "--cuda-version", type=str, required=False, help="Version for CUDA."
    )
    parser.add_argument(
        "--cuda-home", type=str, required=False, help="Home directory for CUDA."
    )
    parser.add_argument(
        "--cudnn-home", type=str, required=False, help="Home directory for CUDNN."
    )
    parser.add_argument(
        "--ort-openvino",
        type=str,
        required=False,
        help="Enable OpenVino execution provider using specified OpenVINO version.",
    )
    parser.add_argument(
        "--ort-tensorrt",
        action="store_true",
        required=False,
        help="Enable TensorRT execution provider.",
    )
    parser.add_argument(
        "--tensorrt-home", type=str, required=False, help="Home directory for TensorRT."
    )
    parser.add_argument(
        "--onnx-tensorrt-tag", type=str, default="", help="onnx-tensorrt repo tag."
    )
    parser.add_argument("--trt-version", type=str, default="", help="TRT version.")

    FLAGS = parser.parse_args()
    if FLAGS.enable_gpu:
        preprocess_gpu_flags()

    # if a tag is provided by the user, then simply use it
    # if the tag is empty - check whether there is an entry in the ORT_TO_TRTPARSER_VERSION_MAP
    # map corresponding to ort version + trt version combo. If yes then use it
    # otherwise we leave it empty and use the defaults from ort
    if (
        FLAGS.onnx_tensorrt_tag == ""
        and FLAGS.ort_version in ORT_TO_TRTPARSER_VERSION_MAP.keys()
    ):
        trt_version = re.match(r"^[0-9]+\.[0-9]+", FLAGS.trt_version)
        if (
            trt_version
            and trt_version.group(0)
            == ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][0]
        ):
            FLAGS.onnx_tensorrt_tag = ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][1]

    if target_platform() == "windows":
        # OpenVINO EP not yet supported for windows build
        if FLAGS.ort_openvino is not None:
            print("warning: OpenVINO not supported for windows, ignoring")
            FLAGS.ort_openvino = None
        dockerfile_for_windows(FLAGS.output)
    else:
        dockerfile_for_linux(FLAGS.output)
