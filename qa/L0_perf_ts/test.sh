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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

# TODO: DLIS-3777 following key update is required only while base image 
#       is not updated accordingly
apt-key del 7fa2af80
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/$(uname -m)/3bf863cc.pub

apt update
apt install -y libb64-dev curl
apt install -y python3 python3-pip python3-dev
pip3 install --upgrade requests

REPODIR=/data/inferenceserver/${REPO_VERSION}
PERF_ANALYZER=/perf_bin/perf_analyzer
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
$PERF_ANALYZER ${PERF_ANALYZER_ARGS}
# Collect data
$PERF_ANALYZER ${PERF_ANALYZER_ARGS} -f ${NAME}.csv >> ${NAME}.log 2>&1
if (( $? != 0 )); then
    RET=1
fi

torchserve --stop

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${NAME}.tjson
echo -e "\"s_benchmark_name\":\"preprocess+resnet50\"," >> ${NAME}.tjson
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

    python $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
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
