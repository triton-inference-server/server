#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export TRITON_PATH="/home/ubuntu"
export DEFAULT_REPO_TAG=${DEFAULT_REPO_TAG:="main"}
export TRITON_COMMON_REPO_TAG=${DEFAULT_REPO_TAG}
export TRITON_CORE_REPO_TAG=${DEFAULT_REPO_TAG}
export TRITON_BACKEND_REPO_TAG=${DEFAULT_REPO_TAG}
export TRITON_THIRD_PARTY_REPO_TAG=${DEFAULT_REPO_TAG}
export IDENTITY_BACKEND_REPO_TAG=${DEFAULT_REPO_TAG}
export PYTHON_BACKEND_REPO_TAG=${DEFAULT_REPO_TAG}
export CHECKSUM_REPOAGENT_REPO_TAG=${DEFAULT_REPO_TAG}
export TRITON_SERVER_REPO_TAG=${TRITON_SERVER_REPO_TAG:=${DEFAULT_REPO_TAG}}
export TRITON_CLIENT_REPO_TAG=${TRITON_CLIENT_REPO_TAG:=${DEFAULT_REPO_TAG}}
export BASE_IMAGE=tritonserver
export SDK_IMAGE=tritonserver_sdk
export BUILD_IMAGE=tritonserver_build
export QA_IMAGE=tritonserver_qa
export TEST_JSON_REPO=/opt/tritonserver/qa/common/inferentia_perf_analyzer_input_data_json
export TEST_REPO=/opt/tritonserver/qa/L0_inferentia_perf_analyzer
export TEST_SCRIPT="test.sh"
CONTAINER_NAME="qa_container"
CONTAINER_VERSION=""
UPSTREAM_CONTAINER_VERSION=""


USAGE="
usage: setup_test_enviroment_and_test.sh [options]. These setting will override exported variables

Setup environment for testing on Inferentia chips and run perf analyzer tests.
-h|--help                        Shows usage
-d|--default-repo-tag            DEFAULT_REPO_TAG for building the test container. Default is main
-s|--server-repo-tag             TRITON_SERVER_REPO_TAG for building test container. Default same DEFAULT_REPO_TAG
-c|--client-repo-tag             TRITON_CLIENT_REPO_TAG for building test container. Default same DEFAULT_REPO_TAG
-v|--container-version           Container version used in build.py. Default is container version used in build.py
-u|--upstream-container-version  Upstream container version for test container. Default is container version used in build.py
-p|--triton-path                 The path where python backend is located and where server repo will be cloned to. Default is /home/ubuntu
"

# Get all options:
OPTS=$(getopt -o hd:s:c:v:u:p: --long help,default-repo-tag:,server-repo-tag:,client-repo-tag:,container-version:,upstream-container-version:,triton-path -- "$@")

for OPTS; do
    case "$OPTS" in
        -h|--help)
        printf "%s\\n" "$USAGE"
        return 0
        ;;
        -d|--default-repo-tag)
        export DEFAULT_REPO_TAG=$2
        echo "Default repo tag set to: ${DEFAULT_REPO_TAG}"
        shift 2
        ;;
        -s|--server-repo-tag)
        export TRITON_SERVER_REPO_TAG=$2
        shift 2
        echo "Server repo tag set to: ${TRITON_SERVER_REPO_TAG}"
        ;;
        -c|--client-repo-tag)
        export TRITON_CLIENT_REPO_TAG=$2
        echo "Client repo tag set to: ${TRITON_CLIENT_REPO_TAG}"
        shift 2
        ;;
        -v|--container-version)
        export CONTAINER_VERSION=$2
        echo "Container version set to: ${CONTAINER_VERSION}"
        shift 2
        ;;
        -u|--upstream-container-version)
        export UPSTREAM_CONTAINER_VERSION=$2
        echo "Upstream container version set to: ${UPSTREAM_CONTAINER_VERSION}"
        shift 2
        ;;
        -p|--triton-path)
        export TRITON_PATH=$2
        echo "Triton path set to: ${TRITON_PATH}"
        shift 2
        ;;
    esac
done

cd ${TRITON_PATH}
echo "Using server repo tag: $TRITON_SERVER_REPO_TAG"
# Clone necessary branches
rm -rf ${TRITON_PATH}/server
git clone --single-branch --depth=1 -b ${TRITON_SERVER_REPO_TAG} \
          https://github.com/triton-inference-server/server.git
cd ${TRITON_PATH}/server
git clone --single-branch --depth=1 -b ${TRITON_CLIENT_REPO_TAG} \
          https://github.com/triton-inference-server/client.git clientrepo

# First set up inferentia and run in detached mode
cd ${TRITON_PATH}/python_backend
chmod 777 ${TRITON_PATH}/python_backend/inferentia/scripts/setup-pre-container.sh
sudo ${TRITON_PATH}/python_backend/inferentia/scripts/setup-pre-container.sh

# If container version is not known, look up container version and upstream container version from build.py
cd ${TRITON_PATH}/server
if [ "${CONTAINER_VERSION}" = "" ]; then
    QUERY_STRING="import build; container_version,_= build.container_versions('$(cat TRITON_VERSION)', None, None); print(container_version)"
    CONTAINER_VERSION=$(python3 -c "${QUERY_STRING}")
    echo "found container version: ${CONTAINER_VERSION} from build.py"
fi
if [ "${UPSTREAM_CONTAINER_VERSION}" = "" ]; then
    QUERY_STRING="import build; _,upstream_container_version = build.container_versions('$(cat TRITON_VERSION)', None, None); print(upstream_container_version)"
    UPSTREAM_CONTAINER_VERSION=$(python3 -c "${QUERY_STRING}")
    echo "found upstream container version: ${UPSTREAM_CONTAINER_VERSION} from build.py"
fi

# Build container with only python backend
cd ${TRITON_PATH}/server
pip3 install docker
./build.py --container-version=${CONTAINER_VERSION} \
           --upstream-container-version=${UPSTREAM_CONTAINER_VERSION} \
           --enable-logging --enable-stats --enable-tracing \
           --enable-metrics --enable-gpu-metrics --enable-gpu \
           --filesystem=gcs --filesystem=azure_storage --filesystem=s3 \
           --endpoint=http --endpoint=grpc \
           --repo-tag=common:${TRITON_COMMON_REPO_TAG} \
           --repo-tag=core:${TRITON_CORE_REPO_TAG} \
           --repo-tag=backend:${TRITON_BACKEND_REPO_TAG} \
           --repo-tag=thirdparty:${TRITON_THIRD_PARTY_REPO_TAG} \
           --backend=identity:${IDENTITY_BACKEND_REPO_TAG} \
           --backend=python:${PYTHON_BACKEND_REPO_TAG} \
           --repoagent=checksum:${CHECKSUM_REPOAGENT_REPO_TAG}
docker tag tritonserver_buildbase "${BUILD_IMAGE}"
docker tag tritonserver "${BASE_IMAGE}"

# Build docker container for SDK
docker build -t ${SDK_IMAGE} \
             -f ${TRITON_PATH}/server/Dockerfile.sdk \
             --build-arg "BASE_IMAGE=nvcr.io/nvidia/tritonserver:${UPSTREAM_CONTAINER_VERSION}-py3-min" \
             --build-arg "TRITON_CLIENT_REPO_SUBDIR=clientrepo" \
             --build-arg "TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG}" \
             --build-arg "TRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG}" \
             --build-arg "TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG}" \
             --build-arg "TRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG}" \
             --build-arg "NVIDIA_TRITON_SERVER_SDK_VERSION=${CONTAINER_VERSION}" .

# Build QA container
docker build -t ${QA_IMAGE} \
                   -f ${TRITON_PATH}/python_backend/inferentia/qa/Dockerfile.QA \
                   --build-arg "TRITON_PATH=${TRITON_PATH}" \
                   --build-arg "BASE_IMAGE=${BASE_IMAGE}"   \
                   --build-arg "BUILD_IMAGE=${BUILD_IMAGE}" \
                   --build-arg "SDK_IMAGE=${SDK_IMAGE}"     .

# Run pytorch instance test
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
docker create --name ${CONTAINER_NAME}             \
            --device /dev/neuron0                  \
            --device /dev/neuron1                  \
            --shm-size=1g --ulimit memlock=-1      \
            -p 8000:8000 -p 8001:8001 -p 8002:8002 \
            --ulimit stack=67108864                \
            -e TEST_REPO=${TEST_REPO}              \
            -e TEST_JSON_REPO=${TEST_JSON_REPO}    \
            -e TRITON_PATH=${TRITON_PATH}          \
            -e USE_PYTORCH="1"                     \
            --net host -ti ${QA_IMAGE}             \
            /bin/bash -c "bash -ex ${TEST_REPO}/${TEST_SCRIPT}" && \
            docker cp /lib/udev ${CONTAINER_NAME}:/mylib/udev && \
            docker cp /home/ubuntu/python_backend ${CONTAINER_NAME}:${TRITON_PATH}/python_backend && \
            docker start -a ${CONTAINER_NAME} || RV=$?;

# Run tensorflow instance tests
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
docker create --name ${CONTAINER_NAME}             \
            --device /dev/neuron0                  \
            --device /dev/neuron1                  \
            --shm-size=1g --ulimit memlock=-1      \
            -p 8000:8000 -p 8001:8001 -p 8002:8002 \
            --ulimit stack=67108864                \
            -e TEST_REPO=${TEST_REPO}              \
            -e TEST_JSON_REPO=${TEST_JSON_REPO}    \
            -e TRITON_PATH=${TRITON_PATH}          \
            -e USE_TENSORFLOW="1"                  \
            --net host -ti ${QA_IMAGE}             \
            /bin/bash -c "bash -ex ${TEST_REPO}/${TEST_SCRIPT}" && \
            docker cp /lib/udev ${CONTAINER_NAME}:/mylib/udev && \
            docker cp /home/ubuntu/python_backend ${CONTAINER_NAME}:${TRITON_PATH}/python_backend && \
            docker start -a ${CONTAINER_NAME} || RV=$?;
