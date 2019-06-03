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

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrtserver:19.06-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:19.06-py3
ARG TENSORFLOW_IMAGE=nvcr.io/nvidia/tensorflow:19.06-py3

############################################################################
## Caffe2 stage: Use PyTorch container to get Caffe2 backend
############################################################################
FROM ${PYTORCH_IMAGE} AS trtserver_caffe2

# We cannot just pull libraries from the PyTorch container... we need
# to:
#   - copy over netdef_backend_c2 interface so it can build with other
#     C2 sources

# Copy netdef_backend_c2 into Caffe2 core so it builds into the
# libcaffe2 library. We want netdef_backend_c2 to build against the
# Caffe2 protobuf since it interfaces with that code.
COPY src/backends/caffe2/netdef_backend_c2.* \
     /opt/pytorch/pytorch/caffe2/core/

# Build same as in pytorch container... except for the NO_DISTRIBUTED
# line where we turn off features not needed for trtserver
# This will build both the caffe2 libraries needed by the Caffe2 NetDef backend
# and the LibTorch library needed by the PyTorch backend.
WORKDIR /opt/pytorch
RUN pip uninstall -y torch
RUN cd pytorch && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
      CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
      NCCL_INCLUDE_DIR="/usr/include/" \
      NCCL_LIB_DIR="/usr/lib/" \
      NO_DISTRIBUTED=1 NO_TEST=1 NO_MIOPEN=1 USE_MKLDNN=0 USE_OPENCV=OFF USE_LEVELDB=OFF \
      python setup.py install && python setup.py clean

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
RUN git clone -b rel-${ONNX_RUNTIME_VERSION} --recursive https://github.com/Microsoft/onnxruntime

ENV PATH="/opt/cmake/bin:${PATH}"
ARG SCRIPT_DIR=/workspace/onnxruntime/tools/ci_build/github/linux/docker/scripts
RUN ${SCRIPT_DIR}/install_ubuntu.sh && ${SCRIPT_DIR}/install_deps.sh

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
## Build stage: Build inference server based on TensorFlow container
############################################################################
FROM ${TENSORFLOW_IMAGE} AS trtserver_build

ARG TRTIS_VERSION=1.3.0dev
ARG TRTIS_CONTAINER_VERSION=19.06dev
ARG PYVER=3.5

# The TFServing release branch must match the TF release used by
# TENSORFLOW_IMAGE
ARG TFS_BRANCH=r1.12

# libcurl and libopencv are needed to build some testing
# applications. libgoogle-glog0v5 is needed by caffe2 libraries.
# libopencv is needed by image preprocessing custom backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            automake \
            libgoogle-glog0v5 \
            libopencv-dev \
            libopencv-core-dev \
            libtool

# Use the PYVER version of python
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python`echo $PYVER | cut -c1-1` && \
    ln -s /usr/bin/python$PYVER /usr/bin/python && \
    ln -s /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

# Caffe2 library requirements...
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so \
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

# LibTorch library
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch.so \
      /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/include/torch/csrc/api/include/. \
      /usr/local/include/.
COPY --from=trtserver_caffe2 /opt/conda/lib/python3.6/site-packages/torch/include/. \
      /usr/local/include/.

# Onnx Runtime library
ARG ONNX_RUNTIME_VERSION=0.4.0
COPY --from=trtserver_onnx /workspace/onnxruntime/include/onnxruntime /usr/local/include/
COPY --from=trtserver_onnx /workspace/build/Release/libonnxruntime.so.${ONNX_RUNTIME_VERSION} /opt/tensorrtserver/lib/
RUN ln -s /opt/tensorrtserver/lib/libonnxruntime.so.${ONNX_RUNTIME_VERSION} /opt/tensorrtserver/lib/libonnxruntime.so

# Copy entire repo into container even though some is not needed for
# build itself... because we want to be able to copyright check on
# files that aren't directly needed for build.
WORKDIR /workspace
RUN rm -fr *
COPY . .

# Pull the TFS release that matches the version of TF being used.
RUN git clone --single-branch -b ${TFS_BRANCH} https://github.com/tensorflow/serving.git

# Modify the TF model loader to allow us to set the default GPU
RUN sha1sum -c tools/patch/tensorflow/checksums && \
    patch -i tools/patch/tensorflow/cc/saved_model/loader.cc \
          /opt/tensorflow/tensorflow-source/tensorflow/cc/saved_model/loader.cc

# TFS modifications. Use a checksum to detect if the TFS file has
# changed... if it has need to verify our patch is still valid and
# update the patch/checksum as necessary.
RUN sha1sum -c tools/patch/tfs/checksums && \
    patch -i tools/patch/tfs/model_servers/server_core.cc \
          /workspace/serving/tensorflow_serving/model_servers/server_core.cc && \
    patch -i tools/patch/tfs/sources/storage_path/file_system_storage_path_source.cc \
          /workspace/serving/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc && \
    patch -i tools/patch/tfs/sources/storage_path/file_system_storage_path_source.h \
          /workspace/serving/tensorflow_serving/sources/storage_path/file_system_storage_path_source.h && \
    patch -i tools/patch/tfs/sources/storage_path/file_system_storage_path_source.proto \
          /workspace/serving/tensorflow_serving/sources/storage_path/file_system_storage_path_source.proto && \
    patch -i tools/patch/tfs/util/retrier.cc \
          /workspace/serving/tensorflow_serving/util/retrier.cc && \
    patch -i tools/patch/tfs/util/BUILD \
          /workspace/serving/tensorflow_serving/util/BUILD && \
    patch -i tools/patch/tfs/workspace.bzl \
          /workspace/serving/tensorflow_serving/workspace.bzl

ENV TF_NEED_GCP 1
ENV TF_NEED_S3 1

# Build the server and any testing artifacts
RUN (cd /opt/tensorflow && ./nvbuild.sh --python$PYVER --configonly) && \
    mv .bazelrc .bazelrc.orig && \
    cat .bazelrc.orig /opt/tensorflow/tensorflow-source/.tf_configure.bazelrc > .bazelrc && \
    bazel build -c opt \
          src/servers/trtserver \
          src/custom/... \
          src/test/... && \
    (cd /opt/tensorrtserver && ln -s /workspace/qa qa) && \
    mkdir -p /opt/tensorrtserver/include && \
    cp bazel-out/k8-opt/genfiles/src/core/api.pb.h /opt/tensorrtserver/include/. && \
    cp bazel-out/k8-opt/genfiles/src/core/model_config.pb.h /opt/tensorrtserver/include/. && \
    cp bazel-out/k8-opt/genfiles/src/core/request_status.pb.h /opt/tensorrtserver/include/. && \
    cp bazel-out/k8-opt/genfiles/src/core/server_status.pb.h /opt/tensorrtserver/include/. && \
    mkdir -p /opt/tensorrtserver/bin && \
    cp bazel-bin/src/servers/trtserver /opt/tensorrtserver/bin/. && \
    cp bazel-bin/src/test/caffe2plan /opt/tensorrtserver/bin/. && \
    mkdir -p /opt/tensorrtserver/lib && \
    cp bazel-bin/src/core/libtrtserver.so /opt/tensorrtserver/lib/. && \
    mkdir -p /opt/tensorrtserver/custom && \
    cp bazel-bin/src/custom/addsub/libaddsub.so /opt/tensorrtserver/custom/. && \
    cp bazel-bin/src/custom/identity/libidentity.so /opt/tensorrtserver/custom/. && \
    cp bazel-bin/src/custom/image_preprocess/libimagepreprocess.so /opt/tensorrtserver/custom/. && \
    cp bazel-bin/src/custom/param/libparam.so /opt/tensorrtserver/custom/. && \
    cp bazel-bin/src/custom/sequence/libsequence.so /opt/tensorrtserver/custom/. && \
    bazel clean --expunge && \
    rm -rf /root/.cache/bazel && \
    rm -rf /tmp/*

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}
ENV PYVER ${PYVER}

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRTIS_VERSION=1.3.0dev
ARG TRTIS_CONTAINER_VERSION=19.06dev
ARG PYVER=3.5

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}
LABEL com.nvidia.tensorrtserver.version="${TENSORRT_SERVER_VERSION}"

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}
ENV PYVER ${PYVER}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# Create a user that can be used to run the tensorrt-server as
# non-root. Make sure that this user to given ID 1000.
ENV TENSORRT_SERVER_USER=tensorrt-server
RUN id -u $TENSORRT_SERVER_USER > /dev/null 2>&1 || \
    useradd $TENSORRT_SERVER_USER && \
    [ `id -u $TENSORRT_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TENSORRT_SERVER_USER` -eq 1000 ]

# libgoogle-glog0v5 is needed by caffe2 libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libgoogle-glog0v5

WORKDIR /opt/tensorrtserver
RUN rm -fr /opt/tensorrtserver/*
COPY LICENSE .
COPY --from=trtserver_build /workspace/serving/LICENSE LICENSE.tfserving
COPY --from=trtserver_build /opt/tensorflow/tensorflow-source/LICENSE LICENSE.tensorflow
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
