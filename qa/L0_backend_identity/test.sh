#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_PY=./identity_test.py
CLIENT_LOG="./client.log"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/all_models --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -fr *.log ./all_models

cp -r ./models ./all_models
cp -r ./models/identity_fp32 ./all_models/identity_bytes
(cd all_models/identity_bytes && \
          sed -i "s/^name:.*/name: \"identity_bytes\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_STRING/g" config.pbtxt)
cp -r ./models/identity_fp32 ./all_models/identity_nobatch_int8
(cd all_models/identity_nobatch_int8 && \
          sed -i "s/^name:.*/name: \"identity_nobatch_int8\"/" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 0/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_INT8/g" config.pbtxt)
cp -r ./models/identity_fp32 ./all_models/identity_uint32
(cd all_models/identity_uint32 && \
          sed -i "s/^name:.*/name: \"identity_uint32\"/" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 3000000 }" >> config.pbtxt)
cp -r ./models/identity_fp32 ./all_models/identity_bf16
(cd all_models/identity_bf16 && \
          sed -i "s/^name:.*/name: \"identity_bf16\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_BF16/g" config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

for PROTOCOL in http grpc; do
    set +e
    python $CLIENT_PY -i $PROTOCOL -v >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed: Client test had a non-zero return code."
        RET=1
    fi
    set -e
done

kill $SERVER_PID
wait $SERVER_PID

# Validate the byte_sizes reported by backend
OLDIFS=$IFS; IFS=','
for i in "byte_size = 0, 8", \
         "byte_size = 7, 2", \
         "byte_size = 16, 6", \
         "byte_size = 20, 2", \
         "byte_size = 160, 2" \
         ; do set -- $i; \
    # $SERVER_LOG is recorded as a binary file. Using -a option
    # to correctly grep the pattern in the server log.
    if [[ $(cat $SERVER_LOG | grep -a $1 | wc -l) -ne $2 ]]; then
        echo -e "\n***\n*** Test Failed $1 $2\n***"
        RET=1
    fi
done
IFS=$OLDIFS

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
