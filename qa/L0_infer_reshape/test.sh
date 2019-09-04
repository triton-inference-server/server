#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
INFER_TEST=infer_reshape_test.py


SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_ARGS=--model-repository=`pwd`/models
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG
rm -fr models && mkdir models
cp -r /data/inferenceserver/$1/qa_reshape_model_repository/* models/. && \
    cp -r /data/inferenceserver/$1/qa_ensemble_model_repository/qa_reshape_model_repository/* models/.
for i in \
        nobatch_zero_3_float32 \
        nobatch_zero_4_float32 \
        zero_1_float32 \
        zero_2_float32 \
        zero_3_float32 \
        zero_4_float32 \
        nobatch_zero_1_int32 \
        nobatch_zero_2_int32 \
        nobatch_zero_3_int32 \
        zero_1_int32 \
        zero_2_int32 \
        zero_3_int32 ; do
    cp -r models/graphdef_${i} models/custom_${i}
    rm -fr models/custom_${i}/1/*
    cp libidentity.so models/custom_${i}/1/.
    (cd models/custom_${i} && \
                sed -i "s/^platform:.*/platform: \"custom\"/" config.pbtxt && \
                sed -i "s/^name:.*/name: \"custom_${i}\"/" config.pbtxt && \
                echo "default_model_filename: \"libidentity.so\"" >> config.pbtxt && \
                echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt)
done

create_nop_modelfile `pwd`/libidentity.so `pwd`/models

RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# python unittest seems to swallow ImportError and still return 0
# exit code. So need to explicitly check CLIENT_LOG to make sure
# we see some running tests
python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed To Run\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
