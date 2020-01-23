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

TRTIS_VERSION=1.10.0jetson-dev

# Git clone repo from github
mkdir $HOME/trtis && cd ${HOME}/trtis && \
  git clone --recursive \
    https://github.com/NVIDIA/tensorrt-inference-server && \
  cd tensorrt-inference-server

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

# TODO publish this to a fixed data repository 
tar -zcvf tensorrtserver${TRTIS_VERSION}.tgz /opt/tensorrtserver
