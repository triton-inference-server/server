#!/bin/bash
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LOG="`pwd`/doc_links.log"
CONFIG="`pwd`/mkdocs.yml"
RET=0

# Download necessary packages
python3 -m pip install mkdocs
python3 -m pip install mkdocs-htmlproofer-plugin==0.10.3

#Download perf_analyzer docs
TRITON_REPO_ORGANIZATION=${TRITON_REPO_ORGANIZATION:="http://github.com/triton-inference-server"}
TRITON_PERF_ANALYZER_REPO_TAG="${TRITON_PERF_ANALYZER_REPO_TAG:=main}"
git clone -b ${TRITON_PERF_ANALYZER_REPO_TAG} ${TRITON_REPO_ORGANIZATION}/perf_analyzer.git
cp `pwd`/perf_analyzer/README.md .
cp -rf `pwd`/perf_analyzer/docs .

# Need to remove all links that start with -- or -. Mkdocs converts all -- to - for anchor links.
# This breaks all links to cli commands throughout the docs. This will iterate over all
# files in the docs directory and remove -- and - at the start of options, which allows the
# tool to check links for correctness.
for file in `pwd`/docs/*.md
do
  echo $file
  sed -i 's/`-*/`/g' $file
  sed -i 's/#-*/#/g' $file
done

exec mkdocs serve -f $CONFIG > $LOG &
PID=$!
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
exit $RET
