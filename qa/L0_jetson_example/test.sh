#!/bin/bash
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_v2.1/zip -O pruned_v2.1.zip
unzip pruned_v2.1.zip -d concurrency_and_dynamic_batching/tao/models/peoplenet && rm pruned_v2.1.zip

# Use TAO convertor for JP4.6
wget --content-disposition https://developer.nvidia.com/jp46-20210820t231431z-001zip -O jp4.6-20210820T231431Z-001.zip
unzip jp4.6-20210820T231431Z-001.zip && rm jp4.6-20210820T231431Z-001.zip

cp tao-converter-jp46-trt8.0.1.6/tao-converter concurrency_and_dynamic_batching/tao/tao-converter && rm -rf jp4.6
chmod 777 concurrency_and_dynamic_batching/tao/tao-converter

(cd concurrency_and_dynamic_batching/tao && bash convert_peoplenet.sh)

# Build the example and make sure permissions
cd concurrency_and_dynamic_batching && make

CLIENT_LOG="./client.log"

# Running the example/s
./people_detection -m gpu -v -r trtis_model_repo_sample_1 -t 6 -s false -p ${HOME}/tritonserver >> ${CLIENT_LOG}.1 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG.1
    RET=1
fi

./people_detection -m gpu -v -r trtis_model_repo_sample_2 -t 6 -s false -p ${HOME}/tritonserver >> ${CLIENT_LOG}.2 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG.2
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
