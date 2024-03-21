#!/bin/bash
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOG="`pwd`/log.txt"
CONFIG="`pwd`/mkdocs.yml"
RET=0
# Download necessary packages
python3 -m pip install mkdocs
python3 -m pip install mkdocs-htmlproofer-plugin

# Get the necessary repos
mkdir repos && cd repos
TRITON_REPO_ORGANIZATION=${TRITON_REPO_ORGANIZATION:="http://github.com/triton-inference-server"}
TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG:="main"}
echo ${TRITON_BACKEND_REPO_TAG}
git clone --single-branch --depth=1 -b ${TRITON_BACKEND_REPO_TAG} ${TRITON_REPO_ORGANIZATION}/backend.git
cd ..

exec mkdocs serve -f $CONFIG > $LOG &
PID=$!
# Time for the compilation to finish. This needs to be increased if other repos
# are added to the test
sleep 20

until [[ (-z `pgrep mkdocs`) ]]; do
    kill -2 $PID
    sleep 2
done

if [[ ! -z `grep "invalid url" $LOG` ]]; then
    cat $LOG
    RET=1
fi


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
# exit $RET
