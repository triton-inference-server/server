# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#
# Multistage build.
#

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.08-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:20.08-py3
ARG TENSORFLOW1_IMAGE=nvcr.io/nvidia/tensorflow:20.08-tf1-py3
ARG TENSORFLOW2_IMAGE=nvcr.io/nvidia/tensorflow:20.08-tf2-py3

############################################################################
## PyTorch stage: Use PyTorch container for Caffe2 and libtorch
############################################################################
FROM ${PYTORCH_IMAGE} AS tritonserver_pytorch

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
FROM ${BASE_IMAGE} AS tritonserver_onnx

# Onnx Runtime release version
ARG ONNX_RUNTIME_VERSION=1.4.0

WORKDIR /workspace

# Get release version of Onnx Runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone -b rel-${ONNX_RUNTIME_VERSION} --recursive https://github.com/Microsoft/onnxruntime && \
    (cd onnxruntime && \
            git submodule update --init --recursive)

ARG SCRIPT_DIR=/workspace/onnxruntime/tools/ci_build/github/linux/docker/scripts

# Copy patches into container...
COPY build/onnxruntime /tmp/trtis/build/onnxruntime

RUN sed -i "s/backend-test-tools.*//" ${SCRIPT_DIR}/install_onnx.sh
RUN cp -r ${SCRIPT_DIR} /tmp/scripts && \
    ${SCRIPT_DIR}/install_ubuntu.sh -p 3.6 -o 18.04 && ${SCRIPT_DIR}/install_deps.sh -p 3.6

ENV PATH /usr/bin:$PATH
RUN cmake --version

# Install OpenVINO
# https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/linux/docker/Dockerfile.ubuntu_openvino
ARG OPENVINO_VERSION=2020.2
# Nested text replacement to skip installing CMake via distribution
# as it downgrades the version (need >= 3.11.0)
RUN sed -i 's/\.\/install_dependencies\.sh/sed -i "s\/cmake \\\\\\\\\/\\\\\\\\\/" install_dependencies\.sh\n\.\/install_dependencies\.sh/' /tmp/scripts/install_openvino.sh
RUN /tmp/scripts/install_openvino.sh -o ${OPENVINO_VERSION}
ENV INTEL_OPENVINO_DIR /data/dldt/openvino_${OPENVINO_VERSION}
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
## TensorFlow stage: Use TensorFlow container
############################################################################
FROM ${TENSORFLOW1_IMAGE} AS tritonserver_tf1
FROM ${TENSORFLOW2_IMAGE} AS tritonserver_tf2

############################################################################
## Build stage: Build inference server
############################################################################
FROM ${BASE_IMAGE} AS tritonserver_build

ARG TRITON_VERSION=2.4.0dev
ARG TRITON_CONTAINER_VERSION=20.10dev

# libgoogle-glog0v5 is needed by caffe2 libraries.
# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip is needed by python backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            git \
            libgoogle-glog0v5 \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            rapidjson-dev \
            libb64-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            software-properties-common && \
    if [ $(cat /etc/os-release | grep 'VERSION_ID="16.04"' | wc -l) -ne 0 ]; then \
        apt-get install -y --no-install-recommends \
                libcurl3-dev; \
    elif [ $(cat /etc/os-release | grep 'VERSION_ID="18.04"' | wc -l) -ne 0 ]; then \
        apt-get install -y --no-install-recommends \
                libcurl4-openssl-dev \
                zlib1g-dev; \
    else \
        echo "Ubuntu version must be either 16.04 or 18.04" && \
        exit 1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for protobuf code generation in Python
RUN pip3 install --upgrade wheel setuptools && \
    pip3 install grpcio-tools

# Server build requires recent version of CMake (FetchContent required)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake

# TensorFlow libraries. Install the monolithic libtensorflow_trtis and
# create links from libtensorflow_framework.so and
# libtensorflow_cc.so.  Custom TF operations link against
# libtensorflow_framework.so so it must be present (and that
# functionality is provided by libtensorflow_trtis.so).
COPY --from=tritonserver_tf1 \
     /usr/local/lib/tensorflow/libtensorflow_trtis.so.1 \
     /opt/tritonserver/backends/tensorflow1/
RUN cd /opt/tritonserver/backends/tensorflow1 && \
    patchelf --set-rpath '$ORIGIN' libtensorflow_trtis.so.1 && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_trtis.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_framework.so.1 && \
    ln -sf libtensorflow_framework.so.1 libtensorflow_framework.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_cc.so.1 && \
    ln -sf libtensorflow_cc.so.1 libtensorflow_cc.so

COPY --from=tritonserver_tf2 \
     /usr/local/lib/tensorflow/libtensorflow_triton.so.2 \
     /opt/tritonserver/backends/tensorflow2/
RUN cd /opt/tritonserver/backends/tensorflow2 && \
    patchelf --set-rpath '$ORIGIN' libtensorflow_triton.so.2 && \
    ln -sf libtensorflow_triton.so.2 libtensorflow_triton.so && \
    ln -sf libtensorflow_triton.so.2 libtensorflow_framework.so.2 && \
    ln -sf libtensorflow_framework.so.2 libtensorflow_framework.so && \
    ln -sf libtensorflow_triton.so.2 libtensorflow_cc.so.2 && \
    ln -sf libtensorflow_cc.so.2 libtensorflow_cc.so

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
RUN cd /opt/tritonserver/lib/pytorch && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# Onnx Runtime headers and library
# Put include files to same directory as ONNX Runtime changed the include path
# https://github.com/microsoft/onnxruntime/pull/1461
ARG ONNX_RUNTIME_VERSION=1.4.0
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
COPY --from=tritonserver_onnx /workspace/build/Release/libonnxruntime.so.${ONNX_RUNTIME_VERSION} \
     /opt/tritonserver/backends/onnxruntime/
RUN cd /opt/tritonserver/backends/onnxruntime && \
    ln -sf libonnxruntime.so.${ONNX_RUNTIME_VERSION} libonnxruntime.so

# Minimum OpenVINO libraries required by ONNX Runtime to link and to run
# with OpenVINO Execution Provider
ARG OPENVINO_VERSION=2020.2
COPY --from=tritonserver_onnx /workspace/build/Release/external/ngraph/lib/libovep_ngraph.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libngraph.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/plugins.xml \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.so \
     /opt/tritonserver/backends/onnxruntime/
COPY --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 \
     /opt/tritonserver/backends/onnxruntime/
RUN cd /opt/tritonserver/backends/onnxruntime && \
    ln -sf libtbb.so.2 libtbb.so && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# Copy entire repo into container even though some is not needed for
# build itself... because we want to be able to copyright check on
# files that aren't directly needed for build.
WORKDIR /workspace
RUN rm -fr *
COPY . .

# Copy ONNX custom op library and model (Needed for testing)
COPY --from=tritonserver_onnx /workspace/build/Release/libcustom_op_library.so \
    /workspace/qa/L0_custom_ops/
COPY --from=tritonserver_onnx /workspace/build/Release/testdata/custom_op_library/custom_op_test.onnx \
    /workspace/qa/L0_custom_ops/

# Build the server.
#
# - Need to find CUDA stubs if they are available since some backends
# may need to link against them. This is identical to the logic in TF
# container nvbuild.sh
ARG TRITON_COMMON_REPO_TAG=imant-summary
ARG TRITON_CORE_REPO_TAG=main
ARG TRITON_BACKEND_REPO_TAG=main

RUN LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l) && \
    if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs; \
        ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    rm -fr builddir && mkdir -p builddir && \
    (cd builddir && \
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
                  -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
                  -DTRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG} \
                  -DTRITON_ENABLE_GRPC=ON \
                  -DTRITON_ENABLE_HTTP=ON \
                  -DTRITON_ENABLE_METRICS=ON \
                  -DTRITON_ENABLE_METRICS_GPU=ON \
                  -DTRITON_ENABLE_STATS=ON \
                  -DTRITON_ENABLE_TRACING=ON \
                  -DTRITON_ENABLE_GCS=ON \
                  -DTRITON_ENABLE_S3=ON \
                  -DTRITON_ENABLE_CUSTOM=ON \
                  -DTRITON_ENABLE_TENSORFLOW=ON \
                  -DTRITON_ENABLE_TENSORRT=ON \
                  -DTRITON_ENABLE_CAFFE2=ON \
                  -DTRITON_ENABLE_ONNXRUNTIME=ON \
                  -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON \
                  -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON \
                  -DTRITON_ENABLE_PYTORCH=ON \
                  -DTRITON_ENABLE_ENSEMBLE=ON \
                  -DTRITON_ONNXRUNTIME_INCLUDE_PATHS="/opt/tritonserver/include/onnxruntime" \
                  -DTRITON_PYTORCH_INCLUDE_PATHS="/opt/tritonserver/include/torch;/opt/tritonserver/include/torch/torch/csrc/api/include;/opt/tritonserver/include/torchvision;/usr/include/python3.6" \
                  -DTRITON_EXTRA_LIB_PATHS="/opt/tritonserver/lib;/opt/tritonserver/backends/tensorflow1;/opt/tritonserver/backends/tensorflow2;/opt/tritonserver/lib/pytorch" \
                  ../build && \
            make -j16 server && \
            mkdir -p /opt/tritonserver/include && \
            cp -r server/install/bin /opt/tritonserver/. && \
            cp -r server/install/lib /opt/tritonserver/. && \
            cp -r server/install/include/triton /opt/tritonserver/include/.) && \
    (cd /opt/tritonserver && ln -sf /workspace/qa qa)

# Build the backends.
#
ARG TRITON_EXAMPLE_BACKEND_TAG=main
RUN for BE in identity repeat square; do \
        rm -fr /tmp/triton_backends && mkdir -p /tmp/triton_backends && \
            (cd /tmp/triton_backends && \
                 git clone --single-branch --depth=1 -b ${TRITON_EXAMPLE_BACKEND_TAG} \
                     https://github.com/triton-inference-server/${BE}_backend.git) && \
            (cd /tmp/triton_backends/${BE}_backend && \
                 mkdir build && cd build && \
                 cmake -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
                       -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
                       -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
                       -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} .. && \
                 make -j16 install && \
                 mkdir -p /opt/tritonserver/backends && \
                 cp -r install/backends/${BE} /opt/tritonserver/backends/.); \
    done

ARG TRITON_ONNXRUNTIME_BACKEND_TAG=main
RUN rm -fr /tmp/triton_backends && mkdir -p /tmp/triton_backends && \
    (cd /tmp/triton_backends && \
         git clone --single-branch --depth=1 -b ${TRITON_ONNXRUNTIME_BACKEND_TAG} \
             https://github.com/triton-inference-server/onnxruntime_backend.git) && \
    (cd /tmp/triton_backends/onnxruntime_backend && \
         mkdir build && cd build && \
         cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
               -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
               -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
               -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} \
               -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON \
               -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON \
               -DTRITON_ONNXRUNTIME_INCLUDE_PATHS="/opt/tritonserver/include/onnxruntime" \
               -DTRITON_ONNXRUNTIME_LIB_PATHS="/opt/tritonserver/backends/onnxruntime" .. && \
         make -j16 install && \
         mkdir -p /opt/tritonserver/backends && \
         cp -r install/backends/onnxruntime /opt/tritonserver/backends/.)

ARG TRITON_TENSORFLOW1_BACKEND_TAG=main
RUN rm -fr /tmp/triton_backends && mkdir -p /tmp/triton_backends && \
    (cd /tmp/triton_backends && \
         git clone --single-branch --depth=1 -b ${TRITON_TENSORFLOW1_BACKEND_TAG} \
             https://github.com/triton-inference-server/tensorflow_backend.git) && \
    (cd /tmp/triton_backends/tensorflow_backend && \
         mkdir build && cd build && \
         cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
               -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
               -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
               -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} \
               -DTRITON_TENSORFLOW_VERSION="1" .. \
               -DTRITON_TENSORFLOW_LIB_PATHS="/opt/tritonserver/backends/tensorflow1" .. && \
         make -j16 install && \
         mkdir -p /opt/tritonserver/backends && \
         cp -r install/backends/tensorflow1 /opt/tritonserver/backends/.)

ARG TRITON_TENSORFLOW2_BACKEND_TAG=main
RUN rm -fr /tmp/triton_backends && mkdir -p /tmp/triton_backends && \
    (cd /tmp/triton_backends && \
         git clone --single-branch --depth=1 -b ${TRITON_TENSORFLOW2_BACKEND_TAG} \
             https://github.com/triton-inference-server/tensorflow_backend.git) && \
    (cd /tmp/triton_backends/tensorflow_backend && \
         mkdir build && cd build && \
         cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
               -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
               -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
               -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} \
               -DTRITON_TENSORFLOW_VERSION="2" .. \
               -DTRITON_TENSORFLOW_LIB_PATHS="/opt/tritonserver/backends/tensorflow2" .. && \
         make -j16 install && \
         mkdir -p /opt/tritonserver/backends && \
         cp -r install/backends/tensorflow2 /opt/tritonserver/backends/.)

ARG TRITON_PYTHON_BACKEND_TAG=main
RUN rm -fr /tmp/triton_backends && mkdir -p /tmp/triton_backends && \
    (cd /tmp/triton_backends && \
         git clone --single-branch --depth=1 -b ${TRITON_PYTHON_BACKEND_TAG} \
             https://github.com/triton-inference-server/python_backend.git) && \
    (cd /tmp/triton_backends/python_backend && \
         mkdir build && cd build && \
         cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
               -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
               -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
               -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} .. && \
         make -j16 install && \
         mkdir -p /opt/tritonserver/backends && \
         cp -r install/backends/python /opt/tritonserver/backends/.)

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV PATH /opt/tritonserver/bin:${PATH}

COPY nvidia_entrypoint.sh /opt/tritonserver
ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRITON_VERSION=2.4.0dev
ARG TRITON_CONTAINER_VERSION=20.10dev

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}

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

ARG OPENVINO_VERSION=2020.2
WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
COPY --chown=1000:1000 LICENSE .
COPY --chown=1000:1000 --from=tritonserver_onnx /data/dldt/openvino_${OPENVINO_VERSION}/LICENSE LICENSE.openvino
COPY --chown=1000:1000 --from=tritonserver_onnx /workspace/onnxruntime/LICENSE LICENSE.onnxruntime
# TF1 and TF2 use the same license
COPY --chown=1000:1000 --from=tritonserver_tf1 /opt/tensorflow/tensorflow-source/LICENSE LICENSE.tensorflow
COPY --chown=1000:1000 --from=tritonserver_pytorch /opt/pytorch/pytorch/LICENSE LICENSE.pytorch
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/lib lib
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/backends backends

# Get ONNX version supported
COPY --chown=1000:1000 --from=tritonserver_onnx /workspace/ort_onnx_version.txt ort_onnx_version.txt
RUN export ONNX_VERSION=`cat ort_onnx_version.txt` && rm -f ort_onnx_version.txt

# Perf test provided by ONNX Runtime, can be used to test run model with ONNX Runtime directly
COPY --chown=1000:1000 --from=tritonserver_onnx /workspace/build/Release/onnxruntime_perf_test /opt/onnxruntime/

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

