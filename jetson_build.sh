TRTIS_VERSION=1.10.0jetson-dev

# Git clone repo from github
cd ${HOME}/trtis && \
  git clone git clone --recursive \
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

LIBCUDA_FOUND=$(ldconfig -p | grep -v compat | awk '{print $1}' | grep libcuda.so | wc -l) && \
    if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs; \
        ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    echo $LD_LIBRARY_PATH && \
    rm -fr builddir && mkdir -p builddir && \
    (cd builddir && \
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DTRTIS_ENABLE_METRICS=OFF \
                  -DTRTIS_ENABLE_TRACING=OFF \
                  -DTRTIS_ENABLE_GCS=OFF \
                  -DTRTIS_ENABLE_S3=OFF \
                  -DTRTIS_ENABLE_CUSTOM=ON \
                  -DTRTIS_ENABLE_TENSORFLOW=OFF \
                  -DTRTIS_ENABLE_TENSORRT=OFF \
                  -DTRTIS_ENABLE_CAFFE2=OFF \
                  -DTRTIS_ENABLE_ONNXRUNTIME=OFF \
                  -DTRTIS_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
                  -DTRTIS_ENABLE_PYTORCH=OFF \
                  ../build && \
            make -j16 trtis && \
            mkdir -p /opt/tensorrtserver/include && \
            cp -r trtis/install/bin /opt/tensorrtserver/. && \
            cp -r trtis/install/lib /opt/tensorrtserver/. && \
            cp -r trtis/install/include /opt/tensorrtserver/include/trtserver)

tar -zcvf tensorrtserver${TRTIS_VERSION}.tgz /opt/tensorrtserver
