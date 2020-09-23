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

ARG TRITON_VERSION=2.4.0dev
ARG TRITON_CONTAINER_VERSION=20.10dev

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.08-py3
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${BUILD_IMAGE} AS tritonserver_build

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

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

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
COPY --chown=1000:1000 LICENSE .
COPY --chown=1000:1000 VERSION .
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/lib/libtritonserver.so lib/
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/backends backends
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/lib/pytorch lib/pytorch
COPY --chown=1000:1000 --from=tritonserver_build /opt/tritonserver/backends/onnxruntime/* backends/onnxruntime/

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

