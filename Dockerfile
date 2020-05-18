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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.03-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:20.03-py3
ARG TENSORFLOW_IMAGE=nvcr.io/nvidia/tensorflow:20.03-tf1-py3

############################################################################
## PyTorch stage: Use PyTorch container for Caffe2 and libtorch
############################################################################
FROM ${PYTORCH_IMAGE} AS tritonserver_pytorch

# Must rebuild in the pytorch container to disable some features that
# are not relevant for inferencing and so that OpenCV libraries are
# not included in the server (which will likely conflict with custom
# backends using opencv).
WORKDIR /opt/pytorch
RUN pip uninstall -y torch
RUN cd pytorch && \
    pip install --upgrade setuptools && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CUDA_HOME="/usr/local/cuda" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    USE_DISTRIBUTED=0 USE_MIOPEN=0 USE_NCCL=0 \
    USE_OPENCV=0 USE_LEVELDB=0 USE_LMDB=0 USE_REDIS=0 \
    BUILD_TEST=0 \
    pip install --no-cache-dir -v .

############################################################################
## Onnx Runtime stage: Build Onnx Runtime on CUDA 10, CUDNN 7
############################################################################
FROM ${BASE_IMAGE} AS tritonserver_onnx

# Currently the prebuilt Onnx Runtime library is built on CUDA 9, thus it
# needs to be built from source

# Onnx Runtime release version
ARG ONNX_RUNTIME_VERSION=1.2.0

# Get release version of Onnx Runtime
WORKDIR /workspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone -b rel-${ONNX_RUNTIME_VERSION} --recursive https://github.com/Microsoft/onnxruntime && \
    (cd onnxruntime && \
            git submodule update --init --recursive)

ENV PATH="/opt/cmake/bin:${PATH}"
ARG SCRIPT_DIR=/workspace/onnxruntime/tools/ci_build/github/linux/docker/scripts

RUN sed -i "s/backend-test-tools.*//" ${SCRIPT_DIR}/install_onnx.sh
RUN cp -r ${SCRIPT_DIR} /tmp/scripts && \
    ${SCRIPT_DIR}/install_ubuntu.sh -p 3.6 -o 18.04 && ${SCRIPT_DIR}/install_deps.sh -p 3.6

# Install OpenVINO
# https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/linux/docker/Dockerfile.ubuntu_openvino
ARG OPENVINO_VERSION=2019_R3.1
RUN /tmp/scripts/install_openvino.sh -o ${OPENVINO_VERSION}
ENV INTEL_CVSDK_DIR /data/dldt/openvino_2019.3.376
ENV INTEL_OPENVINO_DIR /data/dldt/openvino_2019.3.376
ENV InferenceEngine_DIR /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/build

ENV LD_LIBRARY_PATH $INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_CVSDK_DIR/deployment_tools/inference_engine/temp/omp/lib:$INTEL_CVSDK_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH

ENV PATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PATH
ENV PYTHONPATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$INTEL_CVSDK_DIR/tools:$PYTHONPATH
ENV IE_PLUGINS_PATH $INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/intel64

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

############################################################################
## TensorFlow stage: Use TensorFlow container
############################################################################
FROM ${TENSORFLOW_IMAGE} AS tritonserver_tf

############################################################################
## Build stage: Build inference server
############################################################################
FROM ${BASE_IMAGE} AS tritonserver_build

ARG TRITON_VERSION=1.14.0dev
ARG TRITON_CONTAINER_VERSION=20.06dev

# libgoogle-glog0v5 is needed by caffe2 libraries.
# libcurl4-openSSL-dev is needed for GCS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            cmake \
            git \
            libgoogle-glog0v5 \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            rapidjson-dev \
            libb64-dev \
            patchelf \
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

# TensorFlow libraries. Install the monolithic libtensorflow_trtis and
# create links from libtensorflow_framework.so and
# libtensorflow_cc.so.  Custom TF operations link against
# libtensorflow_framework.so so it must be present (and that
# functionality is provided by libtensorflow_trtis.so).
COPY --from=tritonserver_tf \
     /usr/local/lib/tensorflow/libtensorflow_trtis.so.1 \
     /opt/tritonserver/lib/tensorflow/
RUN cd /opt/tritonserver/lib/tensorflow && \
    patchelf --set-rpath '$ORIGIN' libtensorflow_trtis.so.1 && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_trtis.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_framework.so.1 && \
    ln -sf libtensorflow_framework.so.1 libtensorflow_framework.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_cc.so.1 && \
    ln -sf libtensorflow_cc.so.1 libtensorflow_cc.so

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

# LibTorch headers and libraries
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
RUN cd /opt/tritonserver/lib/pytorch && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# Onnx Runtime headers and library
# Put include files to same directory as ONNX Runtime changed the include path
# https://github.com/microsoft/onnxruntime/pull/1461
ARG ONNX_RUNTIME_VERSION=1.2.0
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
     /opt/tritonserver/lib/onnx/
RUN cd /opt/tritonserver/lib/onnx && \
    ln -sf libonnxruntime.so.${ONNX_RUNTIME_VERSION} libonnxruntime.so

# Minimum OpenVINO libraries required by ONNX Runtime to link and to run
# with OpenVINO Execution Provider
COPY --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libinference_engine.so \
     /opt/tritonserver/lib/onnx/
COPY --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.so \
     /opt/tritonserver/lib/onnx/
COPY --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/plugins.xml \
     /opt/tritonserver/lib/onnx/
COPY --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so \
     /opt/tritonserver/lib/onnx/
COPY --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 \
     /opt/tritonserver/lib/onnx/
RUN cd /opt/tritonserver/lib/onnx && \
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

# Build the server.
#
# - Need to find CUDA stubs if they are available since some backends
# may need to link against them. This is identical to the logic in TF
# container nvbuild.sh
RUN LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l) && \
    if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs; \
        ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    rm -fr builddir && mkdir -p builddir && \
    (cd builddir && \
            cmake -DCMAKE_BUILD_TYPE=Release \
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
                  -DTRITON_ENABLE_TENSORRT=OFF \
                  -DTRITON_ENABLE_CAFFE2=ON \
                  -DTRITON_ENABLE_ONNXRUNTIME=ON \
                  -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF \
                  -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON \
                  -DTRITON_ENABLE_PYTORCH=ON \
                  -DTRITON_ENABLE_ENSEMBLE=OFF \
                  -DTRITON_ONNXRUNTIME_INCLUDE_PATHS="/opt/tritonserver/include/onnxruntime" \
                  -DTRITON_PYTORCH_INCLUDE_PATHS="/opt/tritonserver/include/torch" \
                  -DTRITON_EXTRA_LIB_PATHS="/opt/tritonserver/lib;/opt/tritonserver/lib/tensorflow;/opt/tritonserver/lib/pytorch;/opt/tritonserver/lib/onnx" \
                  ../build && \
            make -j16 server && \
            mkdir -p /opt/tritonserver/include && \
            cp -r server/install/bin /opt/tritonserver/. && \
            cp -r server/install/lib /opt/tritonserver/. && \
            cp -r server/install/include /opt/tritonserver/include/tritonserver) && \
    (cd /opt/tritonserver && ln -sf /workspace/qa qa) && \
    (cd /opt/tritonserver/lib && chmod ugo-w+rx *) && \
    (cd /opt/tritonserver/lib/tensorflow && chmod ugo-w+rx *) && \
    (cd /opt/tritonserver/lib/pytorch && chmod ugo-w+rx *) && \
    (cd /opt/tritonserver/lib/onnx && chmod ugo-w+rx *)

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

ARG TRITON_VERSION=1.14.0dev
ARG TRITON_CONTAINER_VERSION=20.06dev

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}

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
RUN if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
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
COPY --chown=1000:1000 --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/LICENSE LICENSE.openvino
COPY --chown=1000:1000 --from=tritonserver_onnx /workspace/onnxruntime/LICENSE LICENSE.onnxruntime
COPY --chown=1000:1000 --from=tritonserver_tf /opt/tensorflow/tensorflow-source/LICENSE LICENSE.tensorflow
COPY --chown=1000:1000 --from=tritonserver_pytorch /opt/pytorch/pytorch/LICENSE LICENSE.pytorch
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/lib lib
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/include include

# Install ONNX-Runtime-OpenVINO dependencies to use it in base container
COPY --chown=1000:1000 --from=tritonserver_onnx /workspace/build/Release/openvino_* \
     /opt/openvino_scripts/
COPY --chown=1000:1000 --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/deployment_tools/model_optimizer \
     /opt/openvino_scripts/openvino_2019.3.376/deployment_tools/model_optimizer/
COPY --chown=1000:1000 --from=tritonserver_onnx /data/dldt/openvino_2019.3.376/tools \
     /opt/openvino_scripts/openvino_2019.3.376/tools
ENV INTEL_CVSDK_DIR /opt/openvino_scripts/openvino_2019.3.376
ENV PYTHONPATH /opt/openvino_scripts:$INTEL_CVSDK_DIR:$INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$INTEL_CVSDK_DIR/tools:$PYTHONPATH

# ONNX Runtime requires Python3 and additional packages to
# convert ONNX models to OpenVINO models in its OpenVINO execution accelerator.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade wheel setuptools test-generator==0.1.1 && \
    (cd $INTEL_CVSDK_DIR/deployment_tools/model_optimizer && \
        pip3 install -r requirements_onnx.txt)

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
