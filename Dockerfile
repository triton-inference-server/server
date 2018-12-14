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

#
# Multistage build.
#

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrtserver:18.11-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:18.11-py3
ARG TENSORFLOW_IMAGE=nvcr.io/nvidia/tensorflow:18.11-py3

############################################################################
## Caffe2 stage: Use PyTorch container to get Caffe2 backend
############################################################################
FROM ${PYTORCH_IMAGE} AS trtserver_caffe2

ARG BUILD_CLIENTS_ONLY=0

# We cannot just pull libraries from the PyTorch container... we need
# to:
#   - copy over netdef_bundle_c2 interface so it can build with other
#     C2 sources
#   - need to patch to delegate logging to the inference server.

# Copy netdef_bundle_c2 into Caffe2 core so it builds into the
# libcaffe2 library. We want netdef_bundle_c2 to build against the
# Caffe2 protobuf since it interfaces with that code.
COPY src/servables/caffe2/netdef_bundle_c2.* \
     /opt/pytorch/pytorch/caffe2/core/

# Modify the C2 logging library to delegate logging to the trtserver
# logger. Use a checksum to detect if the C2 logging file has
# changed... if it has need to verify our patch is still valid and
# update the patch/checksum as necessary.
COPY tools/patch/caffe2 /tmp/patch/caffe2
RUN sha1sum -c /tmp/patch/caffe2/checksums && \
    patch -i /tmp/patch/caffe2/core/logging.cc \
          /opt/pytorch/pytorch/caffe2/core/logging.cc && \
    patch -i /tmp/patch/caffe2/core/logging_is_not_google_glog.h \
          /opt/pytorch/pytorch/caffe2/core/logging_is_not_google_glog.h && \
    patch -i /tmp/patch/caffe2/core/context_gpu.cu \
          /opt/pytorch/pytorch/caffe2/core/context_gpu.cu

# Build same as in pytorch container... except for the NO_DISTRIBUTED
# line where we turn off features not needed for trtserver
WORKDIR /opt/pytorch
RUN pip uninstall -y torch
RUN bash -c 'if [ "$BUILD_CLIENTS_ONLY" != "1" ]; then \
               cd pytorch && \
               TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
                CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
                NCCL_INCLUDE_DIR="/usr/include/" \
                NCCL_LIB_DIR="/usr/lib/" \
                NO_DISTRIBUTED=1 NO_TEST=1 NO_MIOPEN=1 USE_OPENCV=OFF USE_LEVELDB=OFF \
                python setup.py install && python setup.py clean; \
             else \
               mkdir -p /opt/conda/lib/python3.6/site-packages/torch/lib; \
               mkdir -p /opt/conda/lib; \
               touch /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so; \
               touch /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2.so; \
               touch /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so; \
               touch /opt/conda/lib/python3.6/site-packages/torch/lib/libc10.so; \
               touch /opt/conda/lib/libmkl_avx2.so; \
               touch /opt/conda/lib/libmkl_core.so; \
               touch /opt/conda/lib/libmkl_def.so; \
               touch /opt/conda/lib/libmkl_gnu_thread.so; \
               touch /opt/conda/lib/libmkl_intel_lp64.so; fi'

############################################################################
## Build stage: Build inference server based on TensorFlow container
############################################################################
FROM ${TENSORFLOW_IMAGE} AS trtserver_build

ARG TRTIS_VERSION=0.10.0dev
ARG TRTIS_CONTAINER_VERSION=19.01dev
ARG PYVER=3.5
ARG BUILD_CLIENTS_ONLY=0

# The TFServing release branch must match the TF release used by
# TENSORFLOW_IMAGE
ARG TFS_BRANCH=r1.12

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            automake \
            libcurl3-dev \
            libopencv-dev \
            libopencv-core-dev \
            libtool

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python$PYVER get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade setuptools

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
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_avx2.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_core.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_def.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_gnu_thread.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_intel_lp64.so /opt/tensorrtserver/lib/

# Copy entire repo into container even though some is not needed for
# build itself... because we want to be able to copyright check on
# files that aren't directly needed for build.
WORKDIR /workspace
RUN rm -fr *
COPY . .

# Pull the TFS release that matches the version of TF being used.
RUN git clone --single-branch -b ${TFS_BRANCH} https://github.com/tensorflow/serving.git

# Modify the TF logging library to delegate logging to the trtserver
# logger. Use a checksum to detect if the TF logging file has
# changed... if it has need to verify our patch is still valid and
# update the patch/checksum as necessary.
RUN sha1sum -c tools/patch/tensorflow/checksums && \
    patch -i tools/patch/tensorflow/cc/saved_model/loader.cc \
          /opt/tensorflow/tensorflow/cc/saved_model/loader.cc && \
    patch -i tools/patch/tensorflow/core/platform/default/logging.cc \
          /opt/tensorflow/tensorflow/core/platform/default/logging.cc

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
    patch -i tools/patch/tfs/util/net_http/server/internal/evhttp_request.cc \
          /workspace/serving/tensorflow_serving/util/net_http/server/internal/evhttp_request.cc && \
    patch -i tools/patch/tfs/util/net_http/server/internal/evhttp_request.h \
          /workspace/serving/tensorflow_serving/util/net_http/server/internal/evhttp_request.h && \
    patch -i tools/patch/tfs/util/net_http/server/public/BUILD \
          /workspace/serving/tensorflow_serving/util/net_http/server/public/BUILD && \
    patch -i tools/patch/tfs/util/net_http/server/public/server_request_interface.h \
          /workspace/serving/tensorflow_serving/util/net_http/server/public/server_request_interface.h && \
    patch -i tools/patch/tfs/workspace.bzl \
          /workspace/serving/tensorflow_serving/workspace.bzl

ENV TF_NEED_GCP 1
ENV TF_NEED_S3 1

# Build the server, clients and any testing artifacts
RUN (cd /opt/tensorflow && ./nvbuild.sh --python$PYVER --configonly) && \
    (cd tools && mv bazel.rc bazel.orig && \
     cat bazel.orig /opt/tensorflow/.tf_configure.bazelrc > bazel.rc) && \
    bash -c 'if [ "$BUILD_CLIENTS_ONLY" != "1" ]; then \
               bazel build -c opt --config=cuda \
                     src/servers/trtserver \
                     src/custom/... \
                     src/clients/... \
                     src/test/...; \
             else \
               bazel build -c opt src/clients/...; \
             fi' && \
    (cd /opt/tensorrtserver && ln -s /workspace/qa qa) && \
    mkdir -p /opt/tensorrtserver/bin && \
    cp bazel-bin/src/clients/c++/image_client /opt/tensorrtserver/bin/. && \
    cp bazel-bin/src/clients/c++/perf_client /opt/tensorrtserver/bin/. && \
    cp bazel-bin/src/clients/c++/simple_client /opt/tensorrtserver/bin/. && \
    mkdir -p /opt/tensorrtserver/lib && \
    cp bazel-bin/src/clients/c++/librequest.so /opt/tensorrtserver/lib/. && \
    cp bazel-bin/src/clients/c++/librequest.a /opt/tensorrtserver/lib/. && \
    mkdir -p /opt/tensorrtserver/custom && \
    cp bazel-bin/src/custom/addsub/libaddsub.so /opt/tensorrtserver/custom/. && \
    mkdir -p /opt/tensorrtserver/pip && \
    bazel-bin/src/clients/python/build_pip /opt/tensorrtserver/pip/. && \
    bash -c 'if [ "$BUILD_CLIENTS_ONLY" != "1" ]; then \
               cp bazel-bin/src/servers/trtserver /opt/tensorrtserver/bin/.; \
               cp bazel-bin/src/test/caffe2plan /opt/tensorrtserver/bin/.; \
             fi' && \
    bazel clean --expunge && \
    rm -rf /root/.cache/bazel && \
    rm -rf /tmp/*

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}
ENV PYVER ${PYVER}

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRTIS_VERSION=0.10.0dev
ARG TRTIS_CONTAINER_VERSION=19.01dev
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

WORKDIR /opt/tensorrtserver
RUN rm -fr /opt/tensorrtserver/*
COPY LICENSE .
COPY --from=trtserver_build /workspace/serving/LICENSE LICENSE.tfserving
COPY --from=trtserver_build /opt/tensorflow/LICENSE LICENSE.tensorflow
COPY --from=trtserver_caffe2 /opt/pytorch/pytorch/LICENSE LICENSE.pytorch
COPY --from=trtserver_build /opt/tensorrtserver/bin/trtserver bin/
COPY --from=trtserver_build /opt/tensorrtserver/lib lib

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
