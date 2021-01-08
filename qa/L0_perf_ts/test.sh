#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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


if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

DEBIAN_FRONTEND=noninteractive

# Build perf_analyzer from source
apt-get update &&  \
apt-get install -y --no-install-recommends \
        software-properties-common \
        autoconf \
        automake \
        build-essential \
        curl \
        git \
        libb64-dev \
        libopencv-dev \
        libopencv-core-dev \
        libssl-dev \
        libtool \
        pkg-config \
        python3 \
        python3-pip \
        python3-dev \
        rapidjson-dev  \
        vim \
        wget && \
pip3 install --upgrade wheel setuptools && \
pip3 install --upgrade grpcio-tools && \
pip3 install --upgrade pip && \
pip3 install --upgrade requests
# Build expects "python" executable (not python3).
rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    cmake --version
rm -rf server
git clone --single-branch --depth=1 -b ${BENCHMARK_REPO_BRANCH} \
                  https://github.com/triton-inference-server/server.git;
(cd server
mkdir builddir && cd builddir && \
cmake -DCMAKE_BUILD_TYPE=Release \
      -DTRITON_COMMON_REPO_TAG:STRING=main \
      -DTRITON_CORE_REPO_TAG:STRING=main \
      -DTRITON_ENABLE_GPU=OFF \
      -DTRITON_ENABLE_METRICS_GPU=OFF \
      -DTRITON_ENABLE_GRPC=ON \
      -DTRITON_ENABLE_HTTP=ON ../build && \
make -j16 client && \
cp client/install/bin/perf_analyzer /usr/bin/)

REPODIR=/data/inferenceserver/${REPO_VERSION}
REPORTER=../common/reporter.py

rm -f *.log *.csv *.tjson *.json log4j.properties
rm -rf model_store
rm -rf serve

RET=0

# Create model archive. Using default handler for image classification
MODEL_NAME="resnet50_fp32_libtorch"
mkdir model_store
torch-model-archiver --model-name resnet50 --version 1.0 --serialized-file ${REPODIR}/perf_model_store/${MODEL_NAME}/1/model.pt \
--export-path model_store --handler image_classifier -f
# Suppressing the logging for better performance
echo "log4j.rootLogger = OFF" >> log4j.properties
# Run server
torchserve --start --ncs --model-store=model_store --models model_store/resnet50.mar --log-config log4j.properties

sleep 5

# Get the input image to be used for generating requests
STATIC_BATCH=1
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
echo "{\"data\":[{\"TORCHSERVE_INPUT\" : [\"kitten_small.jpg\"]}]}" >> data.json
NAME=${MODEL_NAME}_sbatch${STATIC_BATCH}
PERF_ANALYZER_ARGS="-m resnet50 --service-kind torchserve -i http -u localhost:8080 -b ${STATIC_BATCH} -p 5000 --input-data data.json"

# Run client
# To warmup the model
perf_analyzer ${PERF_ANALYZER_ARGS}
# Collect data
perf_analyzer ${PERF_ANALYZER_ARGS} -f ${NAME}.csv >> ${NAME}.log 2>&1
if (( $? != 0 )); then
    RET=1
fi

torchserve --stop

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${NAME}.tjson
echo -e "\"s_benchmark_name\":\"resnet50\"," >> ${NAME}.tjson
echo -e "\"s_server\":\"torchserve\"," >> ${NAME}.tjson
echo -e "\"s_protocol\":\"http\"," >> ${NAME}.tjson
echo -e "\"s_framework\":\"libtorch\"," >> ${NAME}.tjson
echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${NAME}.tjson
echo -e "\"l_concurrency\":1," >> ${NAME}.tjson
echo -e "\"l_batch_size\":1," >> ${NAME}.tjson
echo -e "\"l_instance_count\":1}]" >> ${NAME}.tjson


if [ -f $REPORTER ]; then
    set +e

    URL_FLAG=
    if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
        URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
    fi

    $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
    if (( $? != 0 )); then
        RET=1
    fi

    set -e
fi

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
