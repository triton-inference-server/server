#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# Note: Before running this script, start Triton server in explicit model control mode:
# tritonserver --model-repository=/path/to/model/repository --model-control-mode=explicit

# Default server URL
SERVER_URL=${1:-"localhost:8001"}
PROTO_PATH="/mnt/builddir/triton-server/_deps/repo-common-src/protobuf"
PROTO_FILE="${PROTO_PATH}/grpccallback_service.proto"
HEALTH_PROTO="${PROTO_PATH}/health.proto"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to print test results
print_result() {
    local test_name=$1
    local result=$2
    if [ $result -eq 0 ]; then
        echo -e "${test_name}: ${GREEN}PASSED${NC}"
    else
        echo -e "${test_name}: ${RED}FAILED${NC}"
    fi
}

echo -e "\n${BOLD}Testing gRPC Callback RPCs against ${SERVER_URL}${NC}\n"

# Test Health Check
echo -e "\n${BOLD}Testing Health Check:${NC}"
grpcurl -proto ${HEALTH_PROTO} \
    --import-path ${PROTO_PATH} \
    -plaintext ${SERVER_URL} \
    grpc.health.v1.Health/Check
print_result "Health Check" $?

# Test Repository Index
echo -e "\n${BOLD}Testing Repository Index:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/RepositoryIndex
print_result "Repository Index" $?

# Test Model Load
echo -e "\n${BOLD}Testing Model Load:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext -d '{"model_name": "simple"}' \
    ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/RepositoryModelLoad
print_result "Model Load" $?

# Wait for model to load
sleep 2

# Test Model Unload
echo -e "\n${BOLD}Testing Model Unload:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext -d '{"model_name": "simple"}' \
    ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/RepositoryModelUnload
print_result "Model Unload" $?

# Test Server Live
echo -e "\n${BOLD}Testing Server Live:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/ServerLive
print_result "Server Live" $?

# Test Server Ready
echo -e "\n${BOLD}Testing Server Ready:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/ServerReady
print_result "Server Ready" $?

# Load model again before testing Model Ready
echo -e "\n${BOLD}Loading model for Model Ready test:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext -d '{"model_name": "simple"}' \
    ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/RepositoryModelLoad
print_result "Model Load" $?

# Wait for model to load
sleep 2

# Test Model Ready
echo -e "\n${BOLD}Testing Model Ready:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext -d '{"name": "simple"}' \
    ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/ModelReady
print_result "Model Ready" $?

# Test Server Metadata
echo -e "\n${BOLD}Testing Server Metadata:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/ServerMetadata
print_result "Server Metadata" $?

# Test Model Metadata
echo -e "\n${BOLD}Testing Model Metadata:${NC}"
grpcurl -proto ${PROTO_FILE} \
    --import-path ${PROTO_PATH} \
    -plaintext -d '{"name": "simple"}' \
    ${SERVER_URL} \
    inference.GRPCInferenceServiceCallback/ModelMetadata
print_result "Model Metadata" $?

echo -e "\n${BOLD}Test Summary:${NC}"
echo "----------------------------------------"