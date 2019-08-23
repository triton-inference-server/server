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

#
# Multistage build.
#

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrtserver:19.07-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3
ARG TENSORFLOW_IMAGE=nvcr.io/nvidia/tensorflow:19.07-py3

############################################################################
## TensorFlow stage: Use TensorFlow container to build
############################################################################
FROM ${TENSORFLOW_IMAGE} AS trtserver_tf

# Modify the TF model loader to allow us to set the default GPU for
# multi-GPU support
COPY tools/patch/tensorflow /tmp/trtis/tools/patch/tensorflow
RUN sha1sum -c /tmp/trtis/tools/patch/tensorflow/checksums && \
    patch -i /tmp/trtis/tools/patch/tensorflow/cc/saved_model/loader.cc \
          /opt/tensorflow/tensorflow-source/tensorflow/cc/saved_model/loader.cc && \
    patch -i /tmp/trtis/tools/patch/tensorflow/BUILD \
          /opt/tensorflow/tensorflow-source/tensorflow/BUILD && \
    patch -i /tmp/trtis/tools/patch/tensorflow/tf_version_script.lds \
          /opt/tensorflow/tensorflow-source/tensorflow/tf_version_script.lds && \
    patch -i /tmp/trtis/tools/patch/tensorflow/nvbuild.sh \
          /opt/tensorflow/nvbuild.sh && \
    patch -i /tmp/trtis/tools/patch/tensorflow/nvbuildopts \
          /opt/tensorflow/nvbuildopts && \
    patch -i /tmp/trtis/tools/patch/tensorflow/bazel_build.sh \
          /opt/tensorflow/bazel_build.sh

# Copy tensorflow_backend_tf into TensorFlow so it builds into the
# monolithic libtensorflow_cc library. We want tensorflow_backend_tf
# to build against the TensorFlow protobuf since it interfaces with
# that code.
COPY src/backends/tensorflow/tensorflow_backend_tf.* \
     /opt/tensorflow/tensorflow-source/tensorflow/

# Build TensorFlow library for TRTIS
WORKDIR /opt/tensorflow
RUN ./nvbuild.sh --python3.6

############################################################################
## PyTorch stage: Use PyTorch container for Caffe2 and libtorch
############################################################################
FROM ${PYTORCH_IMAGE} AS trtserver_caffe2

# Copy netdef_backend_c2 into Caffe2 core so it builds into the
# libtorch library. We want netdef_backend_c2 to build against the
# Caffe2 protobuf since it interfaces with that code.
COPY src/backends/caffe2/netdef_backend_c2.* \
     /opt/pytorch/pytorch/caffe2/core/

# Build same as in pytorch container... except for the NO_DISTRIBUTED
# line where we turn off features not needed for trtserver This will
# build the libraries needed by the Caffe2 NetDef backend and the
# PyTorch libtorch backend.
WORKDIR /opt/pytorch
RUN pip uninstall -y torch
RUN cd pytorch && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
     CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
     USE_DISTRIBUTED=0 USE_MIOPEN=0 USE_NCCL=0 \
     USE_OPENCV=0 USE_LEVELDB=0 USE_LMDB=0 USE_REDIS=0 \
     BUILD_TEST=0 \
     pip install --no-cache-dir -v .

############################################################################
## Onnx Runtime stage: Build Onnx Runtime on CUDA 10, CUDNN 7
############################################################################
FROM ${BASE_IMAGE} AS trtserver_onnx

# Currently the prebuilt Onnx Runtime library is built on CUDA 9, thus it
# needs to be built from source

# Onnx Runtime release version
ARG ONNX_RUNTIME_VERSION=0.4.0

# Get release version of Onnx Runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends git

# Check out stable commit on master until new release
# to support cloud-based filesystems
RUN git clone --recursive https://github.com/Microsoft/onnxruntime && \
    (cd onnxruntime && \
            git checkout c0acb8b6c3b2e3e174627dcb3100009e97c2293d && \
            git submodule update)

ENV PATH="/opt/cmake/bin:${PATH}"
ARG SCRIPT_DIR=/workspace/onnxruntime/tools/ci_build/github/linux/docker/scripts

# Modify install dependencies to corresponding packages in Ubuntu 18.04
RUN sed -i "s/libicu55/libicu60/" ${SCRIPT_DIR}/install_ubuntu.sh && \
    sed -i "s/libpng16/libpng/" ${SCRIPT_DIR}/install_ubuntu.sh && \
    sed -i "s/libprotobuf9v5/libprotobuf10/" ${SCRIPT_DIR}/install_ubuntu.sh && \
    sed -i "s/libcurl3/libcurl4/" ${SCRIPT_DIR}/install_ubuntu.sh && \
    sed -i "s/3\.5/3.6/" ${SCRIPT_DIR}/install_ubuntu.sh
RUN cp -r ${SCRIPT_DIR} /tmp/scripts && \
    ${SCRIPT_DIR}/install_ubuntu.sh && ${SCRIPT_DIR}/install_deps.sh

# Allow configure to pick up GDK and CuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by NVidia's base image)
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so

# Build and Install LLVM
ARG LLVM_VERSION=6.0.1
RUN cd /tmp && \
    wget --no-verbose http://releases.llvm.org/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz && \
    xz -d llvm-$LLVM_VERSION.src.tar.xz && \
    tar xvf llvm-$LLVM_VERSION.src.tar && \
    cd llvm-$LLVM_VERSION.src && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -- -j$(nproc) && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-$LLVM_VERSION -DBUILD_TYPE=Release -P cmake_install.cmake && \
    cd /tmp && \
    rm -rf llvm*

ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

# Build files will be in /workspace/build
ARG COMMON_BUILD_ARGS="--skip_submodule_sync --parallel --build_shared_lib --use_openmp"
RUN mkdir -p /workspace/build
RUN python3 /workspace/onnxruntime/tools/ci_build/build.py --build_dir /workspace/build \
            --config Release $COMMON_BUILD_ARGS \
            --use_cuda \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cudnn-$(echo $CUDNN_VERSION | cut -d. -f1-2)/cuda \
            --update \
            --build

############################################################################
## Build stage: Build inference server
############################################################################
FROM ${BASE_IMAGE} AS trtserver_build

ARG TRTIS_VERSION=1.5.0dev
ARG TRTIS_CONTAINER_VERSION=19.08dev

# libgoogle-glog0v5 is needed by caffe2 libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            autoconf \
            automake \
            build-essential \
            cmake \
            git \
            libgoogle-glog0v5 \
            libre2-dev \
            libssl-dev \
            libtool

# libcurl4-openSSL-dev is needed for GCS
RUN if [ $(cat /etc/os-release | grep 'VERSION_ID="16.04"' | wc -l) -ne 0 ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
                libcurl3-dev; \
    elif [ $(cat /etc/os-release | grep 'VERSION_ID="18.04"' | wc -l) -ne 0 ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
                libcurl4-openssl-dev \
                zlib1g-dev; \
    else \
        echo "Ubuntu version must be either 16.04 or 18.04" && \
        exit 1; \
    fi

# TensorFlow libraries. Install the monolithic libtensorflow_cc and
# create a link libtensorflow_framework.so -> libtensorflow_cc.so so
# that custom tensorflow operations work correctly. Custom TF
# operations link against libtensorflow_framework.so so it must be
# present (and its functionality is provided by libtensorflow_cc.so).
COPY --from=trtserver_tf \
     /usr/local/lib/tensorflow/libtensorflow_cc.so /opt/tensorrtserver/lib/
RUN cd /opt/tensorrtserver/lib && \
    ln -sf libtensorflow_cc.so libtensorflow_framework.so.1 && \
    ln -sf libtensorflow_cc.so libtensorflow_framework.so && \
    ln -sf libtensorflow_cc.so libtensorflow_cc.so.1

# Caffe2 libraries
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10_cuda.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_avx2.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_core.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_def.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_gnu_thread.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_intel_lp64.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_rt.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_vml_def.so /opt/tensorrtserver/lib/

# LibTorch headers and library
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/include \
     /opt/tensorrtserver/include/torch
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch.so \
      /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/lib/libthnvrtc.so \
      /opt/tensorrtserver/lib/

# Onnx Runtime headers and library
ARG ONNX_RUNTIME_VERSION=0.4.0
COPY --from=trtserver_onnx /workspace/onnxruntime/include/onnxruntime \
     /opt/tensorrtserver/include/onnxruntime/
COPY --from=trtserver_onnx /workspace/build/Release/libonnxruntime.so.${ONNX_RUNTIME_VERSION} \
     /opt/tensorrtserver/lib/
RUN cd /opt/tensorrtserver/lib && \
    ln -sf libonnxruntime.so.${ONNX_RUNTIME_VERSION} libonnxruntime.so

# Copy entire repo into container even though some is not needed for
# build itself... because we want to be able to copyright check on
# files that aren't directly needed for build.
WORKDIR /workspace
RUN rm -fr *
COPY . .

# Build the server.
#
# - Need to find CUDA stubs if they are available since some backends
# may need to link against them. This is identical to the login in TF
# container nvbuild.sh
RUN LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l) && \
    if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs; \
        ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    echo $LD_LIBRARY_PATH && \
    rm -fr builddir && mkdir -p builddir && \
    (cd builddir && \
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DTRTIS_ENABLE_METRICS=ON \
                  -DTRTIS_ENABLE_GCS=ON\
                  -DTRTIS_ENABLE_S3=ON\
                  -DTRTIS_ENABLE_CUSTOM=ON \
                  -DTRTIS_ENABLE_TENSORFLOW=ON \
                  -DTRTIS_ENABLE_TENSORRT=ON \
                  -DTRTIS_ENABLE_CAFFE2=ON \
                  -DTRTIS_ENABLE_ONNXRUNTIME=ON \
                  -DTRTIS_ENABLE_PYTORCH=ON \
                  -DTRTIS_ONNXRUNTIME_INCLUDE_PATHS="/opt/tensorrtserver/include/onnxruntime" \
                  -DTRTIS_PYTORCH_INCLUDE_PATHS="/opt/tensorrtserver/include/torch" \
                  -DTRTIS_EXTRA_LIB_PATHS="/opt/tensorrtserver/lib" \
                  ../build && \
            make -j16 trtis && \
            mkdir -p /opt/tensorrtserver/include && \
            cp -r trtis/install/bin /opt/tensorrtserver/. && \
            cp -r trtis/install/lib /opt/tensorrtserver/. && \
            cp -r trtis/install/include /opt/tensorrtserver/include/trtserver) && \
    (cd /opt/tensorrtserver && ln -sf /workspace/qa qa)

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRTIS_VERSION=1.5.0dev
ARG TRTIS_CONTAINER_VERSION=19.08dev

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}
LABEL com.nvidia.tensorrtserver.version="${TENSORRT_SERVER_VERSION}"

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# Needed by Caffe2 libraries to avoid:
# Intel MKL FATAL ERROR: Cannot load libmkl_intel_thread.so
ENV MKL_THREADING_LAYER GNU

# Create a user that can be used to run the tensorrt-server as
# non-root. Make sure that this user to given ID 1000.
ENV TENSORRT_SERVER_USER=tensorrt-server
RUN id -u $TENSORRT_SERVER_USER > /dev/null 2>&1 || \
    useradd $TENSORRT_SERVER_USER && \
    [ `id -u $TENSORRT_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TENSORRT_SERVER_USER` -eq 1000 ]

# libgoogle-glog0v5 is needed by caffe2 libraries.
# libcurl is needed for GCS
RUN if [ $(cat /etc/os-release | grep 'VERSION_ID="16.04"' | wc -l) -ne 0 ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
                libcurl3-dev \
                libgoogle-glog0v5 \
                libre2-1v5; \
    elif [ $(cat /etc/os-release | grep 'VERSION_ID="18.04"' | wc -l) -ne 0 ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
                libcurl4-openssl-dev \
                libgoogle-glog0v5 \
                libre2-4; \
    else \
        echo "Ubuntu version must be either 16.04 or 18.04" && \
        exit 1; \
    fi

WORKDIR /opt/tensorrtserver
RUN rm -fr /opt/tensorrtserver/*
COPY LICENSE .
COPY --from=trtserver_onnx /workspace/onnxruntime/LICENSE LICENSE.onnxruntime
COPY --from=trtserver_tf /opt/tensorflow/tensorflow-source/LICENSE LICENSE.tensorflow
COPY --from=trtserver_caffe2 /opt/pytorch/pytorch/LICENSE LICENSE.pytorch
COPY --from=trtserver_build /opt/tensorrtserver/bin/trtserver bin/
COPY --from=trtserver_build /opt/tensorrtserver/lib lib
COPY --from=trtserver_build /opt/tensorrtserver/include include

RUN chmod ugo-w+rx /opt/tensorrtserver/lib/*.so

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
