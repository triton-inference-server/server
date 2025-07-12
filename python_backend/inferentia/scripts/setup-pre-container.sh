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

#! /bin/sh

USAGE="
usage: setup.sh [options]

Sets up runtime and tools for execution on Inferentia chips.
-h|--help                  Shows usage
-inf1|--inf1-setup         Installs runtime and tools for inf1/neuron, inf1 is default
-inf2|--inf2-setup         Installs runtime and tools for inf2/neuronx
-trn1|--trn1-setup         Installs runtime, tools for inf2, and installs EFA for trn1
"

# Get all options:
OPTS=$(getopt -o hb:v:i:tp --long help,python-backend-path:,python-version:,inferentia-path:,use-tensorflow,use-pytorch,tensorflow-version: -- "$@")


export INSTALL_INF2=0
export INSTALL_INF1=1
export INSTALL_TRN1=0

export CWD=`pwd`

cd /home/ubuntu

for OPTS; do
    case "$OPTS" in
        -h|--help)
        printf "%s\\n" "$USAGE"
        return 0
        ;;
        -inf1|--inf1-setup)
        INSTALL_INF1=1
        echo "Script will install runtime and tools for inf1/neuron"
        shift 1
        ;;
        -inf2|--inf2-setup)
        INSTALL_INF2=1
        shift 1
        echo "Script will install runtime and tools for inf2/neruonx"
        ;;
        -trn1|--trn1-setup)
        INSTALL_TRN1=1
        echo "Script will install runtime and tools for trn1"
        shift 1
        ;;
    esac
done

if [ ${INSTALL_INF1} -ne 1 ] && [ ${INSTALL_INF2} -ne 1 ]; then
    echo "Error: need to specify either -inf1, -inf2 or -trn1."
    printf "source %s\\n" "$USAGE"
    return 1
fi

if [ ${INSTALL_INF1} -eq 1 ] && [ ${INSTALL_INF2} -eq 1]
then
    echo "Error: cannot install both inf1 and inf2 dependencies. Please select either -inf1 or -inf2."
    return 1
fi

if [ ${INSTALL_INF1} -eq 1 ] && [ ${INSTALL_TRN1} -eq 1 ]
then
    echo "Error: cannot install both inf1 and trn1 dependencies. Selecting -trn1 will install inf2 dependencies and EFA."
fi

# First stop and remove old neuron 1.X runtime
sudo systemctl stop neuron-rtd || true
sudo apt remove aws-neuron-runtime -y || true

# Then install new neuron libraries
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
sudo wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB |  apt-key add -


sudo apt-get install -y \
    linux-headers-$(uname -r) \
    git \
    aws-neuronx-dkms=2.* \
    aws-neuronx-tools=2.* \
    aws-neuronx-collectives=2.* -y \
    aws-neuronx-runtime-lib=2.* -y

echo "Installation complete for inf2 runtime and tools."

if [ ${INSTALL_TRN1} -eq 1 ]
then
    # Install EFA Driver (only required for multi-instance training)
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
    wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
    cat aws-efa-installer.key | gpg --fingerprint
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
    tar -xvf aws-efa-installer-latest.tar.gz
    cd aws-efa-installer && sudo bash efa_installer.sh --yes
    cd
    sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer
fi

 # Add PATH
export PATH=/opt/aws/neuron/bin:$PATH
cd ${CWD}