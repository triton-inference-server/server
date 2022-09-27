# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG:="main"}
echo ${TRITON_BACKEND_REPO_TAG}
git clone --single-branch --depth=1 -b ${TRITON_BACKEND_REPO_TAG} https://github.com/triton-inference-server/backend.git
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
