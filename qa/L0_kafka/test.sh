#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
CLIENT_TEST=kafka_test.py
CLIENT_LOG="client.log"
TEST_RESULT_FILE="test_results.txt"
EXPECTED_NUM_TESTS=3

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
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

export CUDA_VISIBLE_DEVICES=0

TF_VERSION=${TF_VERSION:=1}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -rf kafka_2.13-3.2.1/
rm -f kafka_2.13-3.2.1.tgz

apt update
apt install -y default-jre
pip install numpy
pip install kafka-python
pip install tritonclient[all]

# Get and install kafka test server
# Set up derived from https://kafka.apache.org/quickstart
wget https://dlcdn.apache.org/kafka/3.2.1/kafka_2.13-3.2.1.tgz
tar -xzf kafka_2.13-3.2.1.tgz
kafka_2.13-3.2.1/bin/zookeeper-server-start.sh kafka_2.13-3.2.1/config/zookeeper.properties &
ZOO_PID="$!"
# Allow zookeeper to launch
sleep 3
kafka_2.13-3.2.1/bin/kafka-server-start.sh kafka_2.13-3.2.1/config/server.properties &
KAFKA_PID="$!"
sleep 3
kafka_2.13-3.2.1/bin/kafka-topics.sh --create --topic inputs --bootstrap-server localhost:9092
sleep 2
kafka_2.13-3.2.1/bin/kafka-topics.sh --create --topic output --bootstrap-server localhost:9092
sleep 2

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export WSLENV=$WSLENV:TRITONSERVER_DELAY_SCHEDULER
else
    MODELDIR=${MODELDIR:=`pwd`/models}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends
fi

rm -f *.log
rm -fr $MODELDIR && mkdir -p $MODELDIR

RET=0

SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="python"}
export BACKENDS

# Setup non-variable-size model repository
rm -fr *.log *.serverlog
for BACKEND in $BACKENDS; do
    TMP_MODEL_DIR="$DATADIR/qa_model_repository/${BACKEND}_float32_float32_float32"
    if [ "$BACKEND" == "python" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_model="${DATADIR}/qa_model_repository/onnx_float32_float32_float32"
        python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
        mkdir -p models/$python_model/1/
        cat $onnx_model/config.pbtxt | sed 's/platform:.*/backend:\ "python"/g' | sed 's/onnx/python/g' > models/$python_model/config.pbtxt
        cp $onnx_model/output0_labels.txt models/$python_model
        cp ../python_models/add_sub/model.py models/$python_model/1/
    else
        cp -r $TMP_MODEL_DIR models/. 
    fi
    (cd models/$(basename $TMP_MODEL_DIR) && \
          sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
done

SERVER_ARGS="--kafka-port=localhost:9092 --kafka-consumer-topics=inputs --kafka-producer-topic=output  --model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
set +e
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# produce to inputs, consume from outputs
python3 $CLIENT_TEST inputs output >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e
kill -9 $KAFKA_PID
sleep 2
kill -9 $ZOO_PID
sleep 2

kill_server

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Kafka Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Kafka Test FAILED\n***"
    cat $SERVER_LOG
fi

exit $RET

