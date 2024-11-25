#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
set -e

# Define constants
CONTAINER_NAME="tritonserver"
HOST_DIR=$(pwd)
HOST_OUTPUT_DIR="${HOST_DIR}/output"
HOST_DEPENDENCIES_DIR="${HOST_OUTPUT_DIR}/dependencies"
HOST_WHEELS_DIR="${HOST_OUTPUT_DIR}/wheels"
CONTAINER_DEPENDENCIES_DIR="/opt/tritonserver/dependencies"
CONTAINER_BACKENDS_DIR="/opt/tritonserver/backends/python"
CONTAINER_WHEELS_DIR="/opt/tritonserver/python"
CONTAINER_SCRIPTS_DIR="/tmp/scripts"
PATCHED_SCRIPTS_DIR="${HOST_DIR}/scripts"

# Ensure required directories exist
mkdir -p ${HOST_DEPENDENCIES_DIR}
mkdir -p ${HOST_WHEELS_DIR}
mkdir -p ${HOST_OUTPUT_DIR}

# Copy scripts to the container
echo "Copying helper scripts into the container..."
docker exec -ti ${CONTAINER_NAME} bash -c "mkdir -p ${CONTAINER_SCRIPTS_DIR}"
docker cp ${PATCHED_SCRIPTS_DIR}/extract_dependencies.sh ${CONTAINER_NAME}:${CONTAINER_SCRIPTS_DIR}/
docker cp ${PATCHED_SCRIPTS_DIR}/auditwheel_patched.py ${CONTAINER_NAME}:${CONTAINER_SCRIPTS_DIR}/

# Run the dependency extraction script in the container
echo "Running dependency extraction script inside the container..."
docker exec ${CONTAINER_NAME} bash -ex ${CONTAINER_SCRIPTS_DIR}/extract_dependencies.sh

# Copy extracted dependencies to the host
echo "Copying dependencies from container to host..."
docker cp ${CONTAINER_NAME}:${CONTAINER_DEPENDENCIES_DIR}/. ${HOST_DEPENDENCIES_DIR}

# Copy Python backend from container to host
echo "Copying Python backend from container to host..."
docker cp ${CONTAINER_NAME}:${CONTAINER_BACKENDS_DIR}/. ${HOST_DEPENDENCIES_DIR}

# Copy Python wheels from container to host
echo "Copying Python wheels from container to host..."
docker cp ${CONTAINER_NAME}:${CONTAINER_WHEELS_DIR}/. ${HOST_WHEELS_DIR}

# Process wheels on the host using process_wheels.sh
echo "Processing wheels to fix dependencies and update RPATH..."
bash -ex ${PATCHED_SCRIPTS_DIR}/process_wheels.sh ${HOST_DEPENDENCIES_DIR} ${HOST_WHEELS_DIR} ${HOST_OUTPUT_DIR}

echo "All operations completed. Fixed wheels are located in ${HOST_OUTPUT_DIR}."
