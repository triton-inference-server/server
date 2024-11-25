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

# Define paths
TRITON_LIB_PATH="/opt/tritonserver/lib/libtritonserver.so"
PYTHON_BACKENDS_PATH="/opt/tritonserver/backends/python"
DEPENDENCIES_OUTPUT_DIR="/opt/tritonserver/dependencies"

# Create output directory
mkdir -p ${DEPENDENCIES_OUTPUT_DIR}

# Extract dependencies for libtritonserver.so
echo "Extracting dependencies for libtritonserver.so..."
ldd ${TRITON_LIB_PATH} | awk '{print $3}' | grep -v '^$' | xargs -I{} cp -u {} ${DEPENDENCIES_OUTPUT_DIR}

echo "Extracting dependencies for libtritonserver.so..."
ldd ${TRITON_LIB_PATH} | awk '{print $3}' | grep -v '^$' | xargs -I{} cp -u {} ${DEPENDENCIES_OUTPUT_DIR}


# Extract dependencies for Python backend artifacts
echo "Extracting dependencies for Python backend..."
echo "Skipping for now"
for artifact in ${PYTHON_BACKENDS_PATH}/libtriton_python.so \
                ${PYTHON_BACKENDS_PATH}/triton_python_backend_stub \
                ${PYTHON_BACKENDS_PATH}/triton_python_backend_utils.py; do
    if [[ -f $artifact ]]; then
        ldd ${TRITON_LIB_PATH} | awk '{print $3}' | grep -v '^$' | grep -v '^not$' | xargs -I{} cp -u {} ${DEPENDENCIES_OUTPUT_DIR}
    else
        echo "Warning: $artifact not found, skipping."
    fi
done

echo "Dependencies have been extracted to ${DEPENDENCIES_OUTPUT_DIR}."
