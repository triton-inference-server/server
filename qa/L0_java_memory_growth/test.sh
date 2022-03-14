#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Create local model repository
MODEL_REPO=`pwd`/models
rm -r models
cp -r `pwd`/../L0_simple_ensemble/models .
mkdir ${MODEL_REPO}/ensemble_add_sub_int32_int32_int32/1

# Set up test files based on installation instructions
# https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/README.md
rm -r javacpp-presets
git clone https://github.com/bytedeco/javacpp-presets.git
cd javacpp-presets
mvn clean install --projects .,tritonserver
mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform.host
cd ..

SAMPLES_REPO=`pwd`/javacpp-presets/tritonserver/samples
cp Simple.java $SAMPLES_REPO
rm -f *.log
RET=0

# Run test
# If program runs successfully, there was no out of memory error with the specified constraints
mvn compile exec:java -f $SAMPLES_REPO/pom.xml -DargLine="-Xms128m -Xmx768m -Dorg.bytedeco.javacpp.maxPhysicalBytes=2000m -Dorg.bytedeco.javacpp.maxRetries=100" -Dexec.mainClass=Simple -Djavacpp.platform=linux-x86_64 -Dexec.args="-r ${MODEL_REPO} -i 1000000" >>client.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
