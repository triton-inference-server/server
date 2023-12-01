#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
RET=0
set -e
if [ ${PYTHON_ENV_VERSION} = "10" ]; then
    echo No need to set up anything for default python3.${PYTHON_ENV_VERSION}
    exit $RET
fi

source common.sh
source ../common/util.sh

SERVER=/opt/tritonserver/bin/tritonserver
BASE_SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --disable-auto-complete-config"
PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
SERVER_ARGS=$BASE_SERVER_ARGS
SERVER_LOG="./inference_server.log"
export PYTHON_ENV_VERSION=${PYTHON_ENV_VERSION:="10"}
RET=0
EXPECTED_VERSION_STRINGS=""

rm -fr ./models
rm -rf *.tar.gz
install_build_deps
install_conda

# Test other python versions
conda update -n base -c defaults conda -y
# Create a model with python 3.8 version
# Successful execution of the Python model indicates that the environment has
# been setup correctly.
if [ ${PYTHON_ENV_VERSION} = "8" ]; then
    create_conda_env "3.8" "python-3-8"
    conda install -c conda-forge libstdcxx-ng=12 -y
    conda install numpy=1.23.4 -y
    conda install tensorflow=2.10.0 -y
    EXPECTED_VERSION_STRING="Python version is 3.8, NumPy version is 1.23.4, and Tensorflow version is 2.10.0"
    create_python_backend_stub
    conda-pack -o python3.8.tar.gz
    path_to_conda_pack="$PWD/python-3-8"
    mkdir -p $path_to_conda_pack
    tar -xzf python3.8.tar.gz -C $path_to_conda_pack
    mkdir -p models/python_3_8/1/
    cp ../python_models/python_version/config.pbtxt ./models/python_3_8
    (cd models/python_3_8 && \
            sed -i "s/^name:.*/name: \"python_3_8\"/" config.pbtxt && \
            echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}">> config.pbtxt)
    cp ../python_models/python_version/model.py ./models/python_3_8/1/
    cp python_backend/builddir/triton_python_backend_stub ./models/python_3_8
fi

# Create a model with python 3.9 version
# Successful execution of the Python model indicates that the environment has
# been setup correctly.
if [ ${PYTHON_ENV_VERSION} = "9" ]; then
    create_conda_env "3.9" "python-3-9"
    conda install -c conda-forge libstdcxx-ng=12 -y
    conda install numpy=1.23.4 -y
    conda install tensorflow=2.10.0 -y
    EXPECTED_VERSION_STRING="Python version is 3.9, NumPy version is 1.23.4, and Tensorflow version is 2.10.0"
    create_python_backend_stub
    conda-pack -o python3.9.tar.gz
    path_to_conda_pack="$PWD/python-3-9"
    mkdir -p $path_to_conda_pack
    tar -xzf python3.9.tar.gz -C $path_to_conda_pack
    mkdir -p models/python_3_9/1/
    cp ../python_models/python_version/config.pbtxt ./models/python_3_9
    (cd models/python_3_9 && \
            sed -i "s/^name:.*/name: \"python_3_9\"/" config.pbtxt && \
            echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}">> config.pbtxt)
    cp ../python_models/python_version/model.py ./models/python_3_9/1/
    cp python_backend/builddir/triton_python_backend_stub ./models/python_3_9
fi

# Create a model with python 3.11 version
# Successful execution of the Python model indicates that the environment has
# been setup correctly.
if [ ${PYTHON_ENV_VERSION} = "11" ]; then
    create_conda_env "3.11" "python-3-11"
    # tensorflow needs to be installed before numpy so pip does not mess up conda
    # environment
    pip install tensorflow==2.12.0
    conda install -c conda-forge libstdcxx-ng=12 -y
    conda install numpy=1.23.5 -y
    EXPECTED_VERSION_STRING="Python version is 3.11, NumPy version is 1.23.5, and Tensorflow version is 2.12.0"
    create_python_backend_stub
    conda-pack -o python3.11.tar.gz
    path_to_conda_pack="$PWD/python-3-11"
    mkdir -p $path_to_conda_pack
    tar -xzf python3.11.tar.gz -C $path_to_conda_pack
    mkdir -p models/python_3_11/1/
    cp ../python_models/python_version/config.pbtxt ./models/python_3_11
    (cd models/python_3_11 && \
            sed -i "s/^name:.*/name: \"python_3_11\"/" config.pbtxt && \
            echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}">> config.pbtxt)
    cp ../python_models/python_version/model.py ./models/python_3_11/1/
    cp python_backend/builddir/triton_python_backend_stub ./models/python_3_11
fi
conda deactivate
rm -rf ./miniconda

# test that
set +e
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

grep "$EXPECTED_VERSION_STRING" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** $EXPECTED_VERSION_STRING was not found in Triton logs. \n***"
    RET=1
fi
set -e

echo "python environment 3.${PYTHON_ENV_VERSION}"
# copy the stub out to /opt/tritonserver/backends/python/triton_python_backend_stub
cp python_backend/builddir/triton_python_backend_stub /opt/tritonserver/backends/python/triton_python_backend_stub
# Set up environment and stub for each test
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update && apt-get -y install \
                            "python3.${PYTHON_ENV_VERSION}-dev" \
                            "python3.${PYTHON_ENV_VERSION}-distutils" \
                            libboost-dev
rm -f /usr/bin/python3 && \
ln -s "/usr/bin/python3.${PYTHON_ENV_VERSION}" /usr/bin/python3
pip3 install --upgrade install requests numpy virtualenv protobuf
find /opt/tritonserver/qa/pkgs/ -maxdepth 1 -type f -name \
    "tritonclient-*linux*.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade

# Build triton-shm-monitor for the test
cd python_backend && rm -rf install build && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=$PWD/install \
        -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
        -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
        -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} .. && \
    make -j16 triton-shm-monitor install
cp $PWD/install/backends/python/triton_shm_monitor.cpython-* /opt/tritonserver/qa/common/.
set +e
exit $RET
