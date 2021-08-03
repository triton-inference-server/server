#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
source ../common.sh
source ../../common/util.sh

SERVER=/opt/tritonserver/bin/tritonserver
BASE_SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
SERVER_ARGS=$BASE_SERVER_ARGS
SERVER_LOG="./inference_server.log"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}

RET=0

rm -fr ./models
rm -rf *.tar.gz
install_build_deps
install_conda

# Create a model with python 3.7 version
create_conda_env "3.7" "python-3-7"
conda install numpy=1.20.1 -y
conda install tensorflow=2.1.0 -y
create_python_backend_stub
conda-pack -o python3.7.tar.gz
path_to_conda_pack=`pwd`/python3.7.tar.gz
mkdir -p models/python_3_7/1/
cp ../../python_models/python_version/config.pbtxt ./models/python_3_7
(cd models/python_3_7 && \
          sed -i "s/^name:.*/name: \"python_3_7\"/" config.pbtxt && \
          echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}">> config.pbtxt)
cp ../../python_models/python_version/model.py ./models/python_3_7/1/
cp python_backend/builddir/triton_python_backend_stub ./models/python_3_7
conda deactivate

# Create a model with python 3.6 version
create_conda_env "3.6" "python-3-6"
conda install numpy=1.18.1 -y
conda install tensorflow=2.1.0 -y
conda-pack -o python3.6.tar.gz
path_to_conda_pack=`pwd`/python3.6.tar.gz
create_python_backend_stub
mkdir -p models/python_3_6/1/
cp ../../python_models/python_version/config.pbtxt ./models/python_3_6
(cd models/python_3_6 && \
          sed -i "s/^name:.*/name: \"python_3_6\"/" config.pbtxt && \
          echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}" >> config.pbtxt)
rm -rf ./miniconda
cp ../../python_models/python_version/model.py ./models/python_3_6/1/
cp python_backend/builddir/triton_python_backend_stub ./models/python_3_6

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

set +e
grep "Python version is 3.6, NumPy version is 1.18.1, and Tensorflow version is 2.1.0" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Python version is 3.6, NumPy version is 1.18.1, and Tensorflow version is 2.1.0 was not found in Triton logs. \n***"
    RET=1
fi

grep "Python version is 3.7, NumPy version is 1.20.1, and Tensorflow version is 2.1.0" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Python version is 3.7, NumPy version is 1.20.1, and Tensorflow version is 2.1.0 was not found in Triton logs. \n***"
    RET=1
fi

grep "no version information available (required by /bin/bash)." $SERVER_LOG
if [ $? -eq 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** \"no version information available (required by /bin/bash).\" was found in the server logs. \n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Env Manager Test PASSED.\n***"
else
  cat $SERVER_LOG
  echo -e "\n***\n*** Env Manager Test FAILED.\n***"
fi

exit $RET

