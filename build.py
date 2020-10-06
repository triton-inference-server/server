#!/usr/bin/env python3
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import logging
import docker
import os.path
import multiprocessing
import pathlib
import shutil
import subprocess
import sys
import traceback
from distutils.dir_util import copy_tree

#
# Build Triton Inference Server.
#
EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['pytorch', 'tensorrt', 'custom', 'ensemble', 'caffe2']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali'
]
FLAGS = None

# Map from container version to corresponding component versions
# container-version -> (upstream container version, ort version, ort openvino version)
CONTAINER_VERSION_MAP = {
    '20.08': ('20.08', '1.4.0', '2020.2'),  # 2.2.0
    '20.09': ('20.09', '1.4.0', '2020.2'),  # 2.3.0
    '20.10dev': ('20.09', '1.5.1', '2020.4'),  # 2.4.0dev
    '20.10': ('20.10', '1.5.1', '2020.4'),  # 2.4.0
    '20.11dev': ('20.09', '1.5.1', '2020.4')  # 2.5.0dev
}


def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print('<failed to log>', file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    fail_if(True, msg)


def fail_if(p, msg):
    if p:
        print('error: {}'.format(msg), file=sys.stderr)
        sys.exit(1)


def mkdir(path):
    log_verbose('mkdir: {}'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    log_verbose('rmdir: {}'.format(path))
    shutil.rmtree(path, ignore_errors=True)


def cpdir(src, dest):
    log_verbose('cpdir: {} -> {}'.format(src, dest))
    copy_tree(src, dest, preserve_symlinks=1)


def untar(targetdir, tarfile):
    log_verbose('untar {} into {}'.format(tarfile, targetdir))
    p = subprocess.Popen(['tar', '--strip-components=1', '-xf', tarfile],
                         cwd=targetdir)
    p.wait()
    fail_if(p.returncode != 0,
            'untar {} into {} failed'.format(tarfile, targetdir))


def gitclone(cwd, repo, tag, subdir):
    log_verbose('git clone of repo "{}" at tag "{}"'.format(repo, tag))
    p = subprocess.Popen([
        'git', 'clone', '--recursive', '--single-branch', '--depth=1', '-b',
        tag, 'https://github.com/triton-inference-server/{}.git'.format(repo),
        subdir
    ],
                         cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0,
            'git clone of repo "{}" at tag "{}" failed'.format(repo, tag))


def prebuild_command():
    p = subprocess.Popen(FLAGS.container_prebuild_command.split())
    p.wait()
    fail_if(p.returncode != 0, 'container prebuild cmd failed')


def cmake(cwd, args):
    log_verbose('cmake {}'.format(args))
    p = subprocess.Popen([
        'cmake',
    ] + args, cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'cmake failed')


def makeinstall(cwd, target='install'):
    log_verbose('make {}'.format(target))
    verbose_flag = 'VERBOSE=1' if FLAGS.verbose else 'VERBOSE=0'
    p = subprocess.Popen(
        ['make', '-j',
         str(FLAGS.build_parallel), verbose_flag, target],
        cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'make {} failed'.format(target))


def cmake_enable(flag):
    return 'ON' if flag else 'OFF'


def core_cmake_args(components, backends, install_dir):
    cargs = [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    cargs.append('-DTRITON_ENABLE_LOGGING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_logging)))
    cargs.append('-DTRITON_ENABLE_STATS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_stats)))
    cargs.append('-DTRITON_ENABLE_METRICS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_metrics)))
    cargs.append('-DTRITON_ENABLE_METRICS_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu_metrics)))
    cargs.append('-DTRITON_ENABLE_TRACING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_tracing)))

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))
    cargs.append('-DTRITON_MIN_COMPUTE_CAPABILITY={}'.format(
        FLAGS.min_compute_capability))

    cargs.append('-DTRITON_ENABLE_GRPC:BOOL={}'.format(
        cmake_enable('grpc' in FLAGS.endpoint)))
    cargs.append('-DTRITON_ENABLE_HTTP:BOOL={}'.format(
        cmake_enable('http' in FLAGS.endpoint)))

    cargs.append('-DTRITON_ENABLE_GCS:BOOL={}'.format(
        cmake_enable('gcs' in FLAGS.filesystem)))
    cargs.append('-DTRITON_ENABLE_S3:BOOL={}'.format(
        cmake_enable('s3' in FLAGS.filesystem)))

    cargs.append('-DTRITON_ENABLE_TENSORFLOW={}'.format(
        cmake_enable(('tensorflow1' in backends) or
                     ('tensorflow2' in backends))))

    for be in (CORE_BACKENDS + NONCORE_BACKENDS):
        if not be.startswith('tensorflow'):
            cargs.append('-DTRITON_ENABLE_{}={}'.format(
                be.upper(), cmake_enable(be in backends)))
        if (be in CORE_BACKENDS) and (be in backends):
            if be == 'pytorch':
                cargs += pytorch_cmake_args()
            elif be == 'tensorrt':
                pass
            elif be == 'custom':
                pass
            elif be == 'ensemble':
                pass
            elif be == 'caffe2':
                pass
            else:
                fail('unknown core backend {}'.format(be))

    cargs.append(
        '-DTRITON_EXTRA_LIB_PATHS=/opt/tritonserver/lib;/opt/tritonserver/lib/pytorch'
    )
    cargs.append('/workspace/build')
    return cargs


def backend_repo(be):
    if (be == 'tensorflow1') or (be == 'tensorflow2'):
        return 'tensorflow_backend'
    return '{}_backend'.format(be)


def backend_cmake_args(images, components, be, install_dir):
    if be == 'onnxruntime':
        args = onnxruntime_cmake_args()
    elif be == 'tensorflow1':
        args = tensorflow_cmake_args(1, images)
    elif be == 'tensorflow2':
        args = tensorflow_cmake_args(2, images)
    elif be == 'python':
        args = []
    elif be == 'dali':
        args = dali_cmake_args()
    elif be in EXAMPLE_BACKENDS:
        args = []
    else:
        fail('unknown backend {}'.format(be))

    cargs = args + [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))

    cargs.append('..')
    return cargs


def pytorch_cmake_args():
    return [
        '-DTRITON_PYTORCH_INCLUDE_PATHS=/opt/tritonserver/include/torch;/opt/tritonserver/include/torch/torch/csrc/api/include;/opt/tritonserver/include/torchvision;/usr/include/python3.6',
    ]


def onnxruntime_cmake_args():
    return [
        '-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON',
        '-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON',
        '-DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/opt/tritonserver/include/onnxruntime',
        '-DTRITON_ONNXRUNTIME_LIB_PATHS=/opt/tritonserver/backends/onnxruntime'
    ]


def tensorflow_cmake_args(ver, images):
    # If a specific TF image is specified use it, otherwise pull from
    # NGC.
    image_name = "tensorflow{}".format(ver)
    if image_name in images:
        image = images[image_name]
    else:
        image = 'nvcr.io/nvidia/tensorflow:{}-tf{}-py3'.format(
            FLAGS.upstream_container_version, ver)
    return [
        '-DTRITON_TENSORFLOW_VERSION={}'.format(ver),
        '-DTRITON_TENSORFLOW_DOCKER_IMAGE={}'.format(image)
    ]


def dali_cmake_args():
    return [
        '-DTRITON_DALI_SKIP_DOWNLOAD=OFF',
    ]


def create_dockerfile_buildbase(ddir, dockerfile_name, argmap):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG PYTORCH_IMAGE={}

ARG ONNX_RUNTIME_VERSION={}
ARG ONNX_RUNTIME_OPENVINO_VERSION={}

############################################################################
## PyTorch stage: Use PyTorch container for Caffe2 and libtorch
############################################################################
FROM ${{PYTORCH_IMAGE}} AS tritonserver_pytorch

# Must rebuild in the pytorch container to disable some features that
# are not relevant for inferencing and so that OpenCV libraries are
# not included in the server (which will likely conflict with custom
# backends using opencv). The uninstalls seem excessive but is the
# recommendation from pytorch CONTRIBUTING.md.
WORKDIR /opt/pytorch
RUN (conda uninstall -y pytorch || true) && \
    (conda uninstall -y ninja || true) && \
    pip uninstall -y torch && \
    pip uninstall -y torch
RUN cd pytorch && \
    python setup.py clean && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0+PTX" \
    CUDA_HOME="/usr/local/cuda" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    USE_DISTRIBUTED=OFF USE_OPENMP=OFF USE_NCCL=OFF USE_SYSTEM_NCCL=OFF \
    USE_OPENCV=OFF USE_LEVELDB=OFF USE_LMDB=OFF USE_REDIS=OFF \
    BUILD_TEST=OFF \
    pip install --no-cache-dir -v .

############################################################################
## Onnx Runtime stage: Build Onnx Runtime on CUDA 10, CUDNN 7
############################################################################
FROM ${{BASE_IMAGE}} AS tritonserver_onnx

# Onnx Runtime release version from top of file
ARG ONNX_RUNTIME_VERSION

WORKDIR /workspace

# Get release version of Onnx Runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone -b rel-${{ONNX_RUNTIME_VERSION}} --recursive https://github.com/Microsoft/onnxruntime && \
    (cd onnxruntime && \
            git submodule update --init --recursive)

ARG SCRIPT_DIR=/workspace/onnxruntime/tools/ci_build/github/linux/docker/scripts

# Copy patches into container...
COPY build/onnxruntime /tmp/trtis/build/onnxruntime

RUN sed -i "s/backend-test-tools.*//" ${{SCRIPT_DIR}}/install_onnx.sh
RUN cp -r ${{SCRIPT_DIR}} /tmp/scripts && \
    ${{SCRIPT_DIR}}/install_ubuntu.sh -p 3.6 -o 18.04 && ${{SCRIPT_DIR}}/install_deps.sh -p 3.6

ENV PATH /usr/bin:$PATH
RUN cmake --version

# Install OpenVINO
# https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/linux/docker/Dockerfile.ubuntu_openvino
ARG ONNX_RUNTIME_OPENVINO_VERSION
# Nested text replacement to skip installing CMake via distribution
# as it downgrades the version (need >= 3.11.0)
RUN sed -i 's/\\.\\/install_dependencies\\.sh/sed -i "s\\/cmake \\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/" install_dependencies\\.sh\\n\\.\\/install_dependencies\\.sh/' /tmp/scripts/install_openvino.sh
RUN /tmp/scripts/install_openvino.sh -o ${{ONNX_RUNTIME_OPENVINO_VERSION}}
ENV INTEL_OPENVINO_DIR /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}
ENV LD_LIBRARY_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_OPENVINO_DIR/deployment_tools/:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH

ENV PYTHONPATH $INTEL_OPENVINO_DIR/tools:$PYTHONPATH
ENV IE_PLUGINS_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64

RUN wget https://github.com/intel/compute-runtime/releases/download/19.15.12831/intel-gmmlib_19.1.1_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.15.12831/intel-igc-core_1.0.2-1787_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.15.12831/intel-igc-opencl_1.0.2-1787_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.15.12831/intel-opencl_19.15.12831_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.15.12831/intel-ocloc_19.15.12831_amd64.deb && \
    sudo dpkg -i *.deb && rm -rf *.deb

# Allow configure to pick up GDK and CuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by NVidia's base image)
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so

# Build files will be in /workspace/build
ARG COMMON_BUILD_ARGS="--skip_submodule_sync --parallel --build_shared_lib --use_openmp"
RUN mkdir -p /workspace/build
RUN python3 /workspace/onnxruntime/tools/ci_build/build.py --build_dir /workspace/build \
            --config Release $COMMON_BUILD_ARGS \
            --use_cuda \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cudnn-$(echo $CUDNN_VERSION | cut -d. -f1-2)/cuda \
            --use_tensorrt \
            --tensorrt_home /usr/src/tensorrt \
            --use_openvino CPU_FP32 \
            --update \
            --build

# Record version of ONNX installed for ORT testing,
# different versions are installed, but the last one is the latest for ORT
RUN echo "import onnx; print(onnx.__version__)" | python3 > /workspace/ort_onnx_version.txt

############################################################################
## Final stage: Install and arrange all dependencies needed for build
############################################################################
FROM ${{BASE_IMAGE}} AS tritonserver_build

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

# libgoogle-glog0v5 is needed by caffe2 libraries.
# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip is needed by python backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            docker.io \
            git \
            libgoogle-glog0v5 \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            libcurl4-openssl-dev \
            libb64-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            rapidjson-dev \
            software-properties-common \
            unzip \
            wget \
            zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# grpcio-tools grpcio-channelz are needed by python backend
RUN pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker && \
    pip3 install grpcio-tools grpcio-channelz

# Server build requires recent version of CMake (FetchContent required)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake

# BASE_IMAGE can be a tritonserver image so we must remove any
# existing triton install before copying in the new components.
RUN rm -fr /opt/*

# Caffe2 libraries
COPY --from=tritonserver_pytorch \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so \
     /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10.so \
     /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10_cuda.so \
     /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_avx2.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_core.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_def.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_gnu_thread.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_intel_lp64.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_rt.so /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/libmkl_vml_def.so /opt/tritonserver/lib/pytorch/

# LibTorch and Torchvision headers and libraries
COPY --from=tritonserver_pytorch /opt/conda/lib/python3.6/site-packages/torch/include \
     /opt/tritonserver/include/torch
COPY --from=tritonserver_pytorch /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch.so \
      /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch_cpu.so \
      /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch_cuda.so \
      /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_nvrtc.so \
     /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/pytorch/vision/torchvision/csrc \
    /opt/tritonserver/include/torchvision/torchvision/
COPY --from=tritonserver_pytorch /opt/pytorch/vision/build/libtorchvision.so \
    /opt/tritonserver/lib/pytorch/
COPY --from=tritonserver_pytorch /opt/pytorch/pytorch/LICENSE \
    /opt/tritonserver/lib/pytorch/
RUN cd /opt/tritonserver/lib/pytorch && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# Onnx Runtime headers and library
# Put include files to same directory as ONNX Runtime changed the include path
# https://github.com/microsoft/onnxruntime/pull/1461
ARG ONNX_RUNTIME_VERSION
COPY --from=tritonserver_onnx /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
     /opt/tritonserver/include/onnxruntime/
COPY --from=tritonserver_onnx /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
     /opt/tritonserver/include/onnxruntime/
COPY --from=tritonserver_onnx /workspace/onnxruntime/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h \
     /opt/tritonserver/include/onnxruntime/
COPY --from=tritonserver_onnx /workspace/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h \
     /opt/tritonserver/include/onnxruntime/
COPY --from=tritonserver_onnx /workspace/onnxruntime/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h \
     /opt/tritonserver/include/onnxruntime/
COPY --from=tritonserver_onnx /workspace/build/Release/libonnxruntime.so.${{ONNX_RUNTIME_VERSION}} \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /workspace/onnxruntime/LICENSE \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /workspace/ort_onnx_version.txt \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /workspace/build/Release/onnxruntime_perf_test \
     /opt/tritonserver/backends/onnxruntime/
RUN cd /opt/tritonserver/backends/onnxruntime && \
    ln -sf libonnxruntime.so.${{ONNX_RUNTIME_VERSION}} libonnxruntime.so

# Minimum OpenVINO libraries required by ONNX Runtime to link and to run
# with OpenVINO Execution Provider
ARG ONNX_RUNTIME_OPENVINO_VERSION
COPY --from=tritonserver_onnx /workspace/build/Release/external/ngraph/lib/libovep_ngraph.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libinference_engine.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libngraph.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/plugins.xml \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${{ONNX_RUNTIME_OPENVINO_VERSION}}/LICENSE \
     /opt/tritonserver/backends/onnxruntime/LICENSE.openvino
RUN cd /opt/tritonserver/backends/onnxruntime && \
    ln -sf libtbb.so.2 libtbb.so && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

WORKDIR /workspace
RUN rm -fr *
COPY . .

# Copy ONNX custom op library and model (Needed for testing)
COPY --from=tritonserver_onnx /workspace/build/Release/libcustom_op_library.so \
    /workspace/qa/L0_custom_ops/
COPY --from=tritonserver_onnx /workspace/build/Release/testdata/custom_op_library/custom_op_test.onnx \
    /workspace/qa/L0_custom_ops/

ENTRYPOINT []

ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'], argmap['PYTORCH_IMAGE'],
           argmap['ONNX_RUNTIME_VERSION'],
           argmap['ONNX_RUNTIME_OPENVINO_VERSION'])

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_build(ddir, dockerfile_name, argmap):
    df = '''
FROM tritonserver_builder_image AS build
FROM tritonserver_buildbase
COPY --from=build /tmp/tritonbuild /tmp/tritonbuild
'''
    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile(ddir, dockerfile_name, argmap):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${{BUILD_IMAGE}} AS tritonserver_build

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${{BASE_IMAGE}}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
LABEL com.nvidia.tritonserver.version="${{TRITON_SERVER_VERSION}}"

ENV PATH /opt/tritonserver/bin:${{PATH}}

# Need to include pytorch in LD_LIBRARY_PATH since Torchvision loads custom
# ops from that path
ENV LD_LIBRARY_PATH /opt/tritonserver/lib/pytorch/:$LD_LIBRARY_PATH

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# Needed by Caffe2 libraries to avoid:
# Intel MKL FATAL ERROR: Cannot load libmkl_intel_thread.so
ENV MKL_THREADING_LAYER GNU

# Create a user that can be used to run the triton-server as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true && \
    if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
        useradd $TRITON_SERVER_USER; \
    fi && \
    [ `id -u $TRITON_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# libgoogle-glog0v5 is needed by caffe2 libraries.
# libcurl is needed for GCS
RUN apt-get update && \
    if [ $(cat /etc/os-release | grep 'VERSION_ID="16.04"' | wc -l) -ne 0 ]; then \
        apt-get install -y --no-install-recommends \
                libb64-0d \
                libcurl3-dev \
                libgoogle-glog0v5 \
                libre2-1v5; \
    elif [ $(cat /etc/os-release | grep 'VERSION_ID="18.04"' | wc -l) -ne 0 ]; then \
        apt-get install -y --no-install-recommends \
                libb64-0d \
                libcurl4-openssl-dev \
                libgoogle-glog0v5 \
                libre2-4; \
    else \
        echo "Ubuntu version must be either 16.04 or 18.04" && \
        exit 1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
COPY --chown=1000:1000 LICENSE .
COPY --chown=1000:1000 VERSION .
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/lib/libtritonserver.so lib/
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/lib/pytorch lib/pytorch
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/backends/onnxruntime/* backends/onnxruntime/
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])

    if len(FLAGS.backend) > 0:
        df += '''
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/backends backends
'''

    df += '''
# Get ONNX version supported
RUN export ONNX_VERSION=`cat backends/onnxruntime/ort_onnx_version.txt`

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

COPY --chown=1000:1000 nvidia_entrypoint.sh /opt/tritonserver
ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
'''

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def container_build(images):
    # Set the docker build-args based on container version
    if FLAGS.container_version in CONTAINER_VERSION_MAP:
        if FLAGS.upstream_container_version is not None:
            upstream_container_version = FLAGS.upstream_container_version
        else:
            upstream_container_version = CONTAINER_VERSION_MAP[
                FLAGS.container_version][0]
        onnx_runtime_version = CONTAINER_VERSION_MAP[FLAGS.container_version][1]
        onnx_runtime_openvino_version = CONTAINER_VERSION_MAP[
            FLAGS.container_version][2]
    else:
        fail('unsupported container version {}'.format(FLAGS.container_version))

    # The build and install directories within the container.
    build_dir = os.path.join(os.sep, 'tmp', 'tritonbuild')
    install_dir = os.path.join(os.sep, 'tmp', 'tritonbuild', 'install')

    # We can't use docker module for building container because it
    # doesn't stream output and it also seems to handle cache-from
    # incorrectly which leads to excessive rebuilds in the multistage
    # build.
    if 'base' in images:
        base_image = images['base']
    else:
        base_image = 'nvcr.io/nvidia/tritonserver:{}-py3'.format(
            upstream_container_version)

    if 'pytorch' in images:
        pytorch_image = images['pytorch']
    else:
        pytorch_image = 'nvcr.io/nvidia/pytorch:{}-py3'.format(
            upstream_container_version)

    dockerfileargmap = {
        'TRITON_VERSION': FLAGS.version,
        'TRITON_CONTAINER_VERSION': FLAGS.container_version,
        'BASE_IMAGE': base_image,
        'PYTORCH_IMAGE': pytorch_image,
        'ONNX_RUNTIME_VERSION': onnx_runtime_version,
        'ONNX_RUNTIME_OPENVINO_VERSION': onnx_runtime_openvino_version
    }

    cachefrommap = [
        'tritonserver_pytorch', 'tritonserver_pytorch_cache0',
        'tritonserver_pytorch_cache1', 'tritonserver_onnx',
        'tritonserver_onnx_cache0', 'tritonserver_onnx_cache1',
        'tritonserver_buildbase', 'tritonserver_buildbase_cache0',
        'tritonserver_buildbase_cache1'
    ]

    cachefromargs = ['--cache-from={}'.format(k) for k in cachefrommap]
    commonargs = [
        'docker', 'build', '--pull', '-f',
        os.path.join(FLAGS.build_dir, 'Dockerfile.buildbase')
    ]

    log_verbose('buildbase container {}'.format(commonargs + cachefromargs))
    create_dockerfile_buildbase(FLAGS.build_dir, 'Dockerfile.buildbase',
                                dockerfileargmap)
    try:
        # First build Dockerfile.buildbase. Because of the way Docker
        # does caching with multi-stage images, we must build each
        # stage separately to make sure it is cached (specifically
        # this is needed for CI builds where the build starts with a
        # clean docker cache each time).

        # PyTorch
        p = subprocess.Popen(commonargs + cachefromargs + [
            '-t', 'tritonserver_pytorch', '--target', 'tritonserver_pytorch',
            '.'
        ])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_pytorch failed')

        # ONNX Runtime
        p = subprocess.Popen(
            commonargs + cachefromargs +
            ['-t', 'tritonserver_onnx', '--target', 'tritonserver_onnx', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_onnx failed')

        # Final buildbase image
        p = subprocess.Popen(commonargs + cachefromargs +
                             ['-t', 'tritonserver_buildbase', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_buildbase failed')

        # Before attempting to run the new image, make sure any
        # previous 'tritonserver_builder' container is removed.
        client = docker.from_env(timeout=3600)

        try:
            existing = client.containers.get('tritonserver_builder')
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # ignore

        # Next run build.py inside the container with the same flags
        # as was used to run this instance, except:
        #
        # --container-version is removed so that a direct build is
        # performed within the container
        #
        # --build-dir is added/overridden to 'build_dir'
        #
        # --install-dir is added/overridden to 'install_dir'
        #
        # --container-prebuild-command needs to be quoted correctly
        runargs = []
        skip_arg = False
        for a in sys.argv[1:]:
            if skip_arg:
                skip_arg = False
                continue
            if a == '--container-version':
                skip_arg = True
                continue
            elif a.startswith('--container-version='):
                continue
            runargs.append(a)

        runargs += ['--build-dir', build_dir]
        runargs += ['--install-dir', install_dir]

        for idx, arg in enumerate(runargs):
            if arg == '--container-prebuild-command':
                runargs[idx + 1] = '"{}"'.format(runargs[idx + 1])
            elif arg.startswith('--container-prebuild-command='):
                runargs[idx] = '--container-prebuild-command="{}"'.format(
                    runargs[idx][len('--container-prebuild-command='):])

        log_verbose('run {}'.format(runargs))
        container = client.containers.run(
            'tritonserver_buildbase',
            './build.py {}'.format(' '.join(runargs)),
            detach=True,
            name='tritonserver_builder',
            volumes={
                '/var/run/docker.sock': {
                    'bind': '/var/run/docker.sock',
                    'mode': 'rw'
                }
            },
            working_dir='/workspace')
        if FLAGS.verbose:
            for ln in container.logs(stream=True):
                log_verbose(ln)
        ret = container.wait()
        fail_if(ret['StatusCode'] != 0,
                'tritonserver_builder failed: {}'.format(ret))

        # It is possible to copy the install artifacts from the
        # container at this point (and, for example put them in the
        # specified install directory on the host). But for container
        # build we just want to use the artifacts in the server base
        # container which is created below.
        #mkdir(FLAGS.install_dir)
        #tarfilename = os.path.join(FLAGS.install_dir, 'triton.tar')
        #install_tar, stat_tar = container.get_archive(install_dir)
        #with open(tarfilename, 'wb') as taroutfile:
        #    for d in install_tar:
        #        taroutfile.write(d)
        #untar(FLAGS.install_dir, tarfilename)

        # Build is complete, save the container as the
        # tritonserver_build image. We must to this in two steps:
        #
        #   1. Commit the container as image
        #   "tritonserver_builder_image". This image can't be used
        #   directly because it binds the /var/run/docker.sock mount
        #   and so you would need to always run with that mount
        #   specified... so it can be used this way but very
        #   inconvenient.
        #
        #   2. Perform a docker build to create "tritonserver_build"
        #   from "tritonserver_builder_image" that is essentially
        #   identical but removes the mount.
        try:
            client.images.remove('tritonserver_builder_image', force=True)
        except docker.errors.ImageNotFound:
            pass  # ignore

        container.commit('tritonserver_builder_image', 'latest')
        container.remove(force=True)

        create_dockerfile_build(FLAGS.build_dir, 'Dockerfile.build',
                                dockerfileargmap)
        p = subprocess.Popen([
            'docker', 'build', '-t', 'tritonserver_build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile.build'), '.'
        ])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_build failed')

        # Final base image... this is a multi-stage build that uses
        # the install artifacts from the tritonserver_build container.
        create_dockerfile(FLAGS.build_dir, 'Dockerfile', dockerfileargmap)
        p = subprocess.Popen([
            'docker', 'build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile')
        ] + ['-t', 'tritonserver', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver failed')

    except Exception as e:
        logging.error(traceback.format_exc())
        fail('container build failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument('-q',
                          '--quiet',
                          action="store_true",
                          required=False,
                          help='Disable console output.')
    group_qv.add_argument('-v',
                          '--verbose',
                          action="store_true",
                          required=False,
                          help='Enable verbose output.')

    parser.add_argument(
        '--build-dir',
        type=str,
        required=True,
        help=
        'Build directory. All repo clones and builds will be performed in this directory.'
    )
    parser.add_argument(
        '--install-dir',
        required=False,
        default=None,
        help='Install directory, default is <builddir>/opt/tritonserver.')
    parser.add_argument(
        '--build-type',
        required=False,
        default='Release',
        help=
        'Build type, one of "Release", "Debug", "RelWithDebInfo" or "MinSizeRel". Default is "Release".'
    )
    parser.add_argument(
        '-j',
        '--build-parallel',
        type=int,
        required=False,
        default=None,
        help='Build parallelism. Defaults to 2 * number-of-cores.')

    parser.add_argument('--version',
                        type=str,
                        required=False,
                        default='0.0.0',
                        help='The Triton version.')
    parser.add_argument(
        '--container-version',
        type=str,
        required=False,
        help=
        'The Triton container version. When this argument is given Docker will be used for the build.'
    )
    parser.add_argument(
        '--upstream-container-version',
        type=str,
        required=False,
        help=
        'The upstream container version to use when performing a container build. If not specified the upstream container version will be chosen automaticallly based on --container-version.'
    )
    parser.add_argument(
        '--container-prebuild-command',
        type=str,
        required=False,
        help=
        'When performing a container build, this command will be executed within the container just before the build it performed.'
    )
    parser.add_argument(
        '--image',
        action='append',
        required=False,
        help=
        'Use specified Docker image in build as <image-name>,<full-image-name>. <image-name> can be "base", "tensorflow1", "tensorflow2", or "pytorch".'
    )

    parser.add_argument('--enable-logging',
                        action="store_true",
                        required=False,
                        help='Enable logging.')
    parser.add_argument('--enable-stats',
                        action="store_true",
                        required=False,
                        help='Enable statistics collection.')
    parser.add_argument('--enable-metrics',
                        action="store_true",
                        required=False,
                        help='Enable metrics reporting.')
    parser.add_argument('--enable-gpu-metrics',
                        action="store_true",
                        required=False,
                        help='Include GPU metrics in reported metrics.')
    parser.add_argument('--enable-tracing',
                        action="store_true",
                        required=False,
                        help='Enable tracing.')
    parser.add_argument('--enable-nvtx',
                        action="store_true",
                        required=False,
                        help='Enable NVTX.')
    parser.add_argument('--enable-gpu',
                        action="store_true",
                        required=False,
                        help='Enable GPU support.')
    parser.add_argument(
        '--min-compute-capability',
        type=str,
        required=False,
        default='6.0',
        help='Minimum CUDA compute capability supported by server.')

    parser.add_argument(
        '-e',
        '--endpoint',
        action='append',
        required=False,
        help=
        'Include specified endpoint in build. Allowed values are "grpc" and "http".'
    )
    parser.add_argument(
        '--filesystem',
        action='append',
        required=False,
        help=
        'Include specified filesystem in build. Allowed values are "gcs" and "s3".'
    )
    parser.add_argument(
        '-b',
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>[:<repo-tag>]. <repo-tag> indicates the git tag/branch to use for the build, default is "main".'
    )
    parser.add_argument(
        '-r',
        '--repo-tag',
        action='append',
        required=False,
        help=
        'The git tag/branch to use for a component of the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", or "backend".'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image is None:
        FLAGS.image = []
    if FLAGS.repo_tag is None:
        FLAGS.repo_tag = []
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.endpoint is None:
        FLAGS.endpoint = []
    if FLAGS.filesystem is None:
        FLAGS.filesystem = []

    # Initialize map of docker images.
    images = {}
    for img in FLAGS.image:
        parts = img.split(',')
        fail_if(
            len(parts) != 2,
            '--image must specific <image-name>,<full-image-registry>')
        fail_if(
            parts[0] not in ['base', 'pytorch', 'tensorflow1', 'tensorflow2'],
            'unsupported value for --image')
        log('image "{}": "{}"'.format(parts[0], parts[1]))
        images[parts[0]] = parts[1]

    # If --container-version is specified then we perform the actual
    # build within a build container and then from that create a
    # tritonserver container holding the results of the build.
    if FLAGS.container_version is not None:
        container_build(images)
        sys.exit(0)

    # If there is a container pre-build command assume this invocation
    # is being done within the build container and so run the
    # pre-build command.
    if (FLAGS.container_prebuild_command):
        prebuild_command()

    log('Building Triton Inference Server')

    if FLAGS.install_dir is None:
        FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")
    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2
    if FLAGS.version is None:
        FLAGS.version = '0.0.0'

    # Initialize map of common components and repo-tag for each.
    components = {'common': 'main', 'core': 'main', 'backend': 'main'}
    for be in FLAGS.repo_tag:
        parts = be.split(':')
        fail_if(
            len(parts) != 2,
            '--repo-tag must specific <component-name>:<repo-tag>')
        fail_if(
            parts[0] not in components,
            '--repo-tag <component-name> must be "common", "core", or "backend"'
        )
        components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    for be in FLAGS.backend:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append('main')
        log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        backends[parts[0]] = parts[1]

    # Build the core server. For now the core is contained in this
    # repo so we just build in place
    if True:
        repo_build_dir = os.path.join(FLAGS.build_dir, 'tritonserver', 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, 'tritonserver',
                                        'install')

        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              core_cmake_args(components, backends, repo_install_dir))
        makeinstall(repo_build_dir, target='server')

        core_install_dir = FLAGS.install_dir
        mkdir(core_install_dir)
        cpdir(repo_install_dir, core_install_dir)

    # Build each backend...
    for be in backends:
        # Core backends are not built separately from core so skip...
        if (be in CORE_BACKENDS):
            continue

        repo_build_dir = os.path.join(FLAGS.build_dir, be, 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, be, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, backend_repo(be), backends[be], be)
        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              backend_cmake_args(images, components, be, repo_install_dir))
        makeinstall(repo_build_dir)

        backend_install_dir = os.path.join(FLAGS.install_dir, 'backends', be)
        rmdir(backend_install_dir)
        mkdir(backend_install_dir)
        cpdir(os.path.join(repo_install_dir, 'backends', be),
              backend_install_dir)
