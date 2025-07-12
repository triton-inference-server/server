#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

USAGE="
usage: setup.sh [options]

Sets up python execution environment for AWS Neuron SDK for execution on Inferentia chips.
-h|--help                  Shows usage
-b|--python-backend-path   Python backend path, default is: /home/ubuntu/python_backend
-v|--python-version        Python version, default is 3.7
-i|--inferentia-path       Inferentia path, default is: /home/ubuntu
-p|--use-pytorch           Install pytorch-neuron if specified
-t|--use-tensorflow        Install tensorflow-neuron is specified
-inf2|--inf2-setup         Install pytorch or tensorflow neuronx packages for inf2, inf2 is default
-inf1|--inf1-setup         Install pytorch of tensorflow neuron packages for inf1
--tensorflow-version       Version of Tensorflow used. Default is 2. Ignored if installing pytorch-neuron
"

# Get all options:
OPTS=$(getopt -o hb:v:i:tp --long help,python-backend-path:,python-version:,inferentia-path:,use-tensorflow,use-pytorch,tensorflow-version: -- "$@")


export INFERENTIA_PATH=${TRITON_PATH:="/home/ubuntu"}
export PYTHON_BACKEND_PATH="/home/ubuntu/python_backend"
export PYTHON_VERSION=3.7
export USE_PYTORCH=0
export USE_TENSORFLOW=0
export TENSORFLOW_VERSION=2
export INSTALL_INF1=1
export INSTALL_INF2=0

for OPTS; do
    case "$OPTS" in
        -h|--help)
        printf "%s\\n" "$USAGE"
        return 0
        ;;
        -b|--python-backend-path)
        PYTHON_BACKEND_PATH=$2
        echo "Python backend path set to ${PYTHON_BACKEND_PATH}"
        shift 2
        ;;
        -v|--python-version)
        PYTHON_VERSION=$2
        shift 2
        echo "Python version set to ${PYTHON_VERSION}"
        ;;
        -i|--inferentia-path)
        INFERENTIA_PATH=$2
        echo "Inferentia path set to ${INFERENTIA_PATH}"
        shift 2
        ;;
        -t|--use-tensorflow)
        USE_TENSORFLOW=1
        echo "Installing tensorflow neuronx packages"
        shift 1
        ;;
        -p|--use-pytorch)
        USE_PYTORCH=1
        echo "Installing pytorch neuronx packages"
        shift 1
        ;;
        --tensorflow-version)
        TENSORFLOW_VERSION=$2
        echo "Tensorflow version: ${TENSORFLOW_VERSION}"
        shift 2
        ;;
        -inf1|--inf1-setup)
        INSTALL_INF1=1
        INSTALL_INF2=0
        echo "Installing framework and tools for inf1."
        shift 1
        ;;
        -inf2|--inf2-setup)
        INSTALL_INF2=1
        INSTALL_INF1=0
        echo "Installing framework and tools for inf2"
        shift 1
        ;;
        -trn1|--trn1-setup)
        INSTALL_INF2=1 # same frameworks are used for inf2 and trn1
        INSTALL_INF1=0
        echo "Installing framework and tools for trn1/inf2"
        shift 1
        ;;
    esac
done


if [ ${USE_TENSORFLOW} -ne 1 ] && [ ${USE_PYTORCH} -ne 1 ]; then
    echo "Error: need to specify either -p (use pytorch) or -t (use tensorflow)."
    printf "%s\\n" "${USAGE}"
    return 1
fi

if [ ${USE_TENSORFLOW} -eq 1 ] && [ ${USE_PYTORCH} -eq 1 ]; then
    echo "Error: can specify only one of -p (use pytorch) or -t (use tensorflow)."
    printf "%s\\n" "${USAGE}"
    return 1
fi

if [ ${USE_TENSORFLOW} -eq 1 ]; then
    if [ ${TENSORFLOW_VERSION} -ne 1 ] && [ ${TENSORFLOW_VERSION} -ne 2 ]; then
        echo "Error: need to specify --tensorflow-version to be 1 or 2. TENSORFLOW_VERSION currently is: ${TENSORFLOW_VERSION}"
        printf "%s\\n" "${USAGE}"
        return 1
    fi
fi

# Install python_backend_stub installing dependencies
apt-get update && \
    apt-get install -y --no-install-recommends \
              zlib1g-dev \
              wget \
              libarchive-dev   \
              rapidjson-dev


# Set Pip repository  to point to the Neuron repository
# since we need to use pip to update:
#  https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade pip

if [ ${INSTALL_INF2} -eq 1 ];then
    # Install Neuron Runtime
    # Then install new neuron libraries
    . /etc/os-release
    tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB |  apt-key add -
    apt-get update
    apt-get install -y aws-neuronx-collectives=2.* aws-neuronx-runtime-lib=2.*
fi


if [ ${USE_TENSORFLOW} -eq 1 ]; then
    # Update Neuron TensorFlow
    if [ ${INSTALL_INF1} -eq 1 ] && [ ${TENSORFLOW_VERSION} -eq 1 ]; then
        pip install --upgrade tensorflow-neuron==1.15.5.* neuron-cc
    elif [ ${INSTALL_INF1} -eq 1 ]; then
        pip install --upgrade tensorflow-neuron[cc]
    elif [ ${INSTALL_INF2} -eq 1 ] && [ ${TENSORFLOW_VERSION} -eq 1 ]; then
        pip install --upgrade neuronx-cc==2.* tensorflow-neuronx==1.* tensorboard-plugin-neuronx
    elif [ ${INSTALL_INF2} -eq 1 ]; then
        pip install --upgrade neuronx-cc==2.* tensorflow-neuronx==2.* tensorboard-plugin-neuronx
    fi
fi

if [ ${USE_PYTORCH} -eq 1 ];then
    # conda install torch-neuron torchvision -y
    # Upgrade torch-neuron and install transformers
    if [ ${INSTALL_INF1} -eq 1 ]; then
        pip install --upgrade torch-neuron neuron-cc[tensorflow] "protobuf" torchvision "transformers==4.6.0"
    elif [ ${INSTALL_INF2} -eq 1 ]; then
        pip install --upgrade neuronx-cc==2.* torch-neuronx torchvision transformers-neuronx
    fi
fi

# Upgrade the rules and sockets
cp /mylib/udev/rules.d/* /lib/udev/rules.d/
