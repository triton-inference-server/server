#!/bin/bash
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

# Git clone repo from github
mkdir ${HOME}/trtis && cd ${HOME}/trtis && \
  git clone --single-branch --depth=1 \
    https://github.com/NVIDIA/tensorrt-inference-server && \
  cd tensorrt-inference-server

TRTIS_VERSION=`cat VERSION`

# Install dependencies
apt-get update && \
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
            libtool \
            libboost-dev \
            libh2o-dev \
            libh2o-evloop-dev \
            libnuma-dev \
            libwslay-dev \
            libuv1-dev && \
    if [ $(cat /etc/os-release | grep 'VERSION_ID="18.04"' | wc -l) -ne 0 ]; then \
        apt-get install -y --no-install-recommends \
                libcurl4-openssl-dev \
                zlib1g-dev; \
    else \
        echo "Ubuntu version must be 18.04" && \
        exit 1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# TensorFlow libraries. Install the monolithic libtensorflow_trtis and
# create a link libtensorflow_framework.so -> libtensorflow_trtis.so so
# that custom tensorflow operations work correctly. Custom TF
# operations link against libtensorflow_framework.so so it must be
# present (and its functionality is provided by libtensorflow_trtis.so).
# TODO Copy libtensorflow_trtis.so.1 from artifact store
(cd /opt/tensorrtserver/lib && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_framework.so.1 && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_framework.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_trtis.so && \
    ln -sf libtensorflow_trtis.so.1 libtensorflow_cc.so)

LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l) && \
    if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs; \
        ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    echo $LD_LIBRARY_PATH && \
    cd ${HOME}/trtis/tensorrt-inference-server
    rm -fr builddir && mkdir -p builddir && \
    (cd builddir && \
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DTRTIS_ENABLE_METRICS=OFF \
                  -DTRTIS_ENABLE_TRACING=OFF \
                  -DTRTIS_ENABLE_GCS=OFF \
                  -DTRTIS_ENABLE_S3=OFF \
                  -DTRTIS_ENABLE_CUSTOM=ON \
                  -DTRTIS_ENABLE_TENSORFLOW=ON \
                  -DTRTIS_ENABLE_TENSORRT=OFF \
                  -DTRTIS_ENABLE_CAFFE2=OFF \
                  -DTRTIS_ENABLE_ONNXRUNTIME=OFF \
                  -DTRTIS_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
                  -DTRTIS_ENABLE_PYTORCH=OFF \
                  -DTRTIS_EXTRA_LIB_PATHS="/opt/tensorrtserver/lib" \
                  ../build && \
            make -j16 trtis && \
            mkdir -p /opt/tensorrtserver/include && \
            cp -r trtis/install/bin /opt/tensorrtserver/. && \
            cp -r trtis/install/lib /opt/tensorrtserver/. && \
            cp -r trtis/install/include /opt/tensorrtserver/include/trtserver)

# Write version of jetpack installed
if [ -f /etc/nv_tegra_release ]; then
  JETSON_L4T_STRING=$(head -n 1 /etc/nv_tegra_release)
  JETSON_L4T_RELEASE=$(echo $JETSON_L4T_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
  JETSON_L4T_REVISION=$(echo $JETSON_L4T_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
  JETSON_L4T="$JETSON_L4T_RELEASE.$JETSON_L4T_REVISION"
else
  JETSON_L4T_STRING=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core)
  JETSON_L4T=$(echo $JETSON_L4T_STRING | cut -f 1 -d '-')
fi

# https://developer.nvidia.com/embedded/jetpack-archive
case $JETSON_L4T in
    "32.3.1") JETPACK_VERSION="4.3" ;;
    "32.2.1") JETPACK_VERSION="4.2.2" ;;
    "32.2.0" | "32.2") JETPACK_VERSION="4.2.1" ;;
    "32.1.0" | "32.1") JETPACK_VERSION="4.2" ;;
    "31.1.0" | "31.1") JETPACK_VERSION="4.1.1" ;;
    "31.0.2") JETPACK_VERSION="4.1" ;;
    "31.0.1") JETPACK_VERSION="4.0" ;;
    "28.2.1") JETPACK_VERSION="3.3 | 3.2.1" ;;
    "28.2.0" | "28.2") JETPACK_VERSION="3.2" ;;
    "28.1.0" | "28.1") JETPACK_VERSION="3.1" ;;
    "27.1.0" | "27.1") JETPACK_VERSION="3.0" ;;
    "24.2.1") JETPACK_VERSION="3.0 | 2.3.1" ;;
    "24.2.0" | "24.2") JETPACK_VERSION="2.3" ;;
    "24.1.0" | "24.1") JETPACK_VERSION="2.2.1 | 2.2" ;;
    "23.2.0" | "23.2") JETPACK_VERSION="2.1" ;;
    "23.1.0" | "23.1") JETPACK_VERSION="2.0" ;;
    "21.5.0" | "21.5") JETPACK_VERSION="2.3.1 | 2.3" ;;
    "21.4.0" | "21.4") JETPACK_VERSION="2.2 | 2.1 | 2.0 | DP 1.2" ;;
    "21.3.0" | "21.3") JETPACK_VERSION="DP 1.1" ;;
    "21.2.0" | "21.2") JETPACK_VERSION="DP 1.0" ;;
    *) JETPACK_VERSION="UNKNOWN" ;;
esac
# Export Jetson Jetpack installed
export JETPACK_VERSION

# TODO publish this to a fixed data repository
tar -zcvf tensorrtserver${TRTIS_VERSION}-jetpack-${JETPACK_VERSION}.tgz /opt/tensorrtserver
