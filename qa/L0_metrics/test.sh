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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
  REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
  echo -e "Repository version must be specified"
  echo -e "\n***\n*** Test Failed\n***"
  exit 1
fi

MODELDIR=`pwd`/models
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
OPTDIR=${OPTDIR:="/opt"}
SERVER=${OPTDIR}/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${MODELDIR}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG

RET=0

# Prepare a libtorch float32 model with basic config
rm -rf $MODELDIR
model=libtorch_float32_float32_float32
mkdir -p $MODELDIR/${model}/1 && \
  cp -r $DATADIR/${model}/1/* $MODELDIR/${model}/1/. && \
  cp $DATADIR/${model}/config.pbtxt $MODELDIR/${model}/. && \
  (cd $MODELDIR/${model} && \
  sed -i "s/label_filename:.*//" config.pbtxt && \
  echo "instance_group [{ kind: KIND_GPU }]" >> config.pbtxt)

set +e
export CUDA_VISIBLE_DEVICES=0,1,2
run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER\n***" 
  cat $SERVER_LOG
  exit 1
fi

num_gpus=`curl -s localhost:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 3 ]; then
  echo "Found $num_gpus GPU(s) instead of 3 GPUs being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

export CUDA_VISIBLE_DEVICES=0
run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER\n***"
  cat $SERVER_LOG
  exit 1
fi

num_gpus=`curl -s localhost:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 1 ]; then
  echo "Found $num_gpus GPU(s) instead of 1 GPU being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi
kill $SERVER_PID
wait $SERVER_PID


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
