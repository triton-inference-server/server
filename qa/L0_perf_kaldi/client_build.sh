TRITON_VERSION=20.03

cd /workspace

git clone --single-branch --depth=1 -b r${TRITON_VERSION} \
    https://github.com/NVIDIA/triton-inference-server.git

# git clone asr_kaldi from gitlab
cp -r asr_kaldi/kaldi-asr-client triton-inference-server/src/clients/c++

(cd triton-inference-server/src/clients/c++ && \
    echo "add_subdirectory(kaldi-asr-client)" >> "CMakeLists.txt")

# Client dependencies
apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        rapidjson-dev

pip3 install --upgrade wheel setuptools grpcio-tools

# Build client library and kaldi perf client
(cd triton-inference-server/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install \
          -DTRTIS_ENABLE_GRPC_V2=ON && \
    make -j16 trtis-clients)
