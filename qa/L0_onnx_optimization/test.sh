#!/bin/bash
# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

DATADIR=/data/inferenceserver/${REPO_VERSION}

CLIENT_LOG="./client.log"
ONNXTRT_OPTIMIZATION_TEST=onnxtrt_optimization_test.py

SERVER=/opt/tritonserver/bin/tritonserver
CACHE_PATH=`pwd`/trt_cache
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --exit-on-error=false"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

RET=0

for MODEL in \
        onnx_float32_float32_float32; do
    rm -f ./*.log
    rm -fr models && mkdir -p models
    cp -r $DATADIR/qa_model_repository/${MODEL} \
       models/${MODEL}_test && \
    rm -fr models/${MODEL}_test/2 && \
    rm -fr models/${MODEL}_test/3 && \
    # Set instance count > 1 to test parallel instance loading across all EPs
    INSTANCE_COUNT=5
    (cd models/${MODEL}_test && \
            sed -i 's/_float32_float32_float32/&_test/' config.pbtxt && \
            echo -e "\ninstance_group { count: ${INSTANCE_COUNT} }" >> config.pbtxt) && \
    # Enable session.use_device_allocator_for_initializers
    cp -r models/${MODEL}_test models/${MODEL}_session_config && \
    (cd models/${MODEL}_session_config && \
            sed -i 's/_float32_test/_float32_session_config/' config.pbtxt && \
            echo "parameters: { key: \"session.use_device_allocator_for_initializers\" value: { string_value: \"1\" }}" >> config.pbtxt) && \
    # CUDA EP optimization params
    cp -r models/${MODEL}_test models/${MODEL}_cuda_config && \
    (cd models/${MODEL}_cuda_config && \
            sed -i 's/_float32_test/_float32_cuda_config/' \
                config.pbtxt && \
            echo "parameters: { key: \"cudnn_conv_algo_search\" value: { string_value: \"1\" }} \
            parameters: { key: \"arena_extend_strategy\" value: { string_value: \"1\" }}
            parameters: { key: \"gpu_mem_limit\" value: { string_value: \"18446744073709551614\" }} " \ >> config.pbtxt) && \
    # CUDA EP optimization params specified in gpu_execution_accelerator field
    cp -r models/${MODEL}_test models/${MODEL}_cuda_param_field && \
    (cd models/${MODEL}_cuda_param_field && \
            sed -i 's/_float32_test/_float32_cuda_param_field/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"cuda\" \
            parameters { key: \"cudnn_conv_use_max_workspace\" value: \"0\" } \
            parameters { key: \"use_ep_level_unified_stream\" value: \"1\" } }]}}" \
            >> config.pbtxt) && \
    # CPU EP optimization params
    cp -r models/${MODEL}_test models/${MODEL}_cpu_config && \
    (cd models/${MODEL}_cpu_config && \
            sed -i 's/_float32_test/_float32_cpu_config/' \
                config.pbtxt && \
            echo "parameters: { key: \"intra_op_thread_count\" value: { string_value: \"1\" }} \
            parameters: { key: \"enable_mem_arena\" value: { string_value: \"1\" }}
            parameters: { key: \"enable_mem_pattern\" value: { string_value: \"1\" }}
            parameters: { key: \"memory.enable_memory_arena_shrinkage\" value: { string_value: \"cpu:0\" }} " \ >> config.pbtxt) && \
    # GPU execution accelerators with default setting
    cp -r models/${MODEL}_test models/${MODEL}_trt && \
    (cd models/${MODEL}_trt && \
            sed -i 's/_float32_test/_float32_trt/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\"} ] } }" >> config.pbtxt) && \
    # TRT execution accelerators with correct parameters
    cp -r models/${MODEL}_test models/${MODEL}_trt_param && \
    (cd models/${MODEL}_trt_param && \
            sed -i 's/_float32_test/_float32_trt_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"trt_max_partition_iterations\" value: \"1000\" } \
            parameters { key: \"trt_dump_subgraphs\" value: \"1\" } \
            parameters { key: \"trt_timing_cache_enable\" value: \"1\" } \
            parameters { key: \"trt_build_heuristics_enable\" value: \"1\" } \
            parameters { key: \"trt_cuda_graph_enable\" value: \"1\" } \
            parameters { key: \"max_workspace_size_bytes\" value: \"1073741824\" } }]}}" \
            >> config.pbtxt) && \
    # TRT execution accelerators with cache enabled
    cp -r models/${MODEL}_test models/${MODEL}_trt_cache_on && \
    (cd models/${MODEL}_trt_cache_on && \
            sed -i 's/_float32_test/_float32_trt_cache_on/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"trt_engine_cache_enable\" value: \"1\" } \
            parameters { key: \"trt_max_partition_iterations\" value: \"1000\" } \
            parameters { key: \"trt_dump_subgraphs\" value: \"1\" } \
            parameters { key: \"trt_timing_cache_enable\" value: \"1\" } \
            parameters { key: \"trt_build_heuristics_enable\" value: \"1\" } \
            parameters { key: \"trt_cuda_graph_enable\" value: \"1\" } \
            parameters { key: \"trt_engine_cache_path\" value: \"${CACHE_PATH}\" } }]}}" \
            >> config.pbtxt) && \
    # TRT execution accelerators with unknown parameters
    cp -r models/${MODEL}_test models/${MODEL}_trt_unknown_param && \
    (cd models/${MODEL}_trt_unknown_param && \
            sed -i 's/_float32_test/_float32_trt_unknown_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"segment_size\" value: \"1\" } }]}}" \
            >> config.pbtxt) && \
    # TRT execution accelerators with invalid parameters
    cp -r models/${MODEL}_test models/${MODEL}_trt_invalid_param && \
    (cd models/${MODEL}_trt_invalid_param && \
            sed -i 's/_float32_test/_float32_trt_invalid_param/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"tensorrt\" \
            parameters { key: \"precision_mode\" value: \"FP16\" } \
            parameters { key: \"max_workspace_size_bytes\" value: \"abc\" } }]}}" \
            >> config.pbtxt) && \
    # Unknown GPU execution accelerator
    cp -r models/${MODEL}_test models/${MODEL}_unknown_gpu && \
    (cd models/${MODEL}_unknown_gpu && \
            sed -i 's/_float32_test/_float32_unknown_gpu/' \
                config.pbtxt && \
            echo "optimization { execution_accelerators { gpu_execution_accelerator : [ { name : \"unknown_gpu\" } ] } }" >> config.pbtxt) && \

    run_server_tolive
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    grep "Configuring 'session.use_device_allocator_for_initializers' to '1'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected Configuring 'session.use_device_allocator_for_initializers' to '1'\n***"
        RET=1
    fi

    grep "TensorRT Execution Accelerator is set for '${MODEL}_trt'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set for '${MODEL}_trt'\n***"
        RET=1
    fi

    grep "TensorRT Execution Accelerator is set for '${MODEL}_trt_param'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set for '${MODEL}_trt_param'\n***"
        RET=1
    fi

    grep "TensorRT Execution Accelerator is set for '${MODEL}_trt_cache_on'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected TensorRT Execution Accelerator is set for '${MODEL}_trt_cache_on'\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_trt_unknown_param' version 1: Invalid argument: unknown parameter 'segment_size' is provided for TensorRT Execution Accelerator" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected unknown parameter 'segment_size' returns error\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_trt_invalid_param' version 1: Invalid argument: failed to convert 'abc' to unsigned long long integral number" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected invalid parameter 'abc' returns error\n***"
        RET=1
    fi

    grep "failed to load '${MODEL}_unknown_gpu' version 1: Invalid argument: unknown Execution Accelerator 'unknown_gpu' is requested" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected 'unknown_gpu' Execution Accelerator returns error\n***"
        RET=1
    fi

    grep "memory limit: 18446744073709551614 arena_extend_strategy: 1" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected configurations not set for '${MODEL}_cuda_config'\n***"
        RET=1
    fi

    grep "CUDA Execution Accelerator is set for '${MODEL}_cpu_config'" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected CUDA Execution Accelerator is set for '${MODEL}_cpu_config'\n***"
        RET=1
    fi

    matched_line=$(grep "CUDA Execution Accelerator is set for 'onnx_float32_float32_float32_cuda_param_field'" $SERVER_LOG)
    if [[ "$matched_line" != *"use_ep_level_unified_stream=1"* ]] || [[ "$matched_line" != *"cudnn_conv_use_max_workspace=0"* ]]; then
        echo -e "\n***\n*** Failed. Expected CUDA Execution Accelerator options correctly set for '${MODEL}_cuda_param_field'\n***"
        RET=1
    fi

    # arena configs
    grep "Configuring enable_mem_arena to 1" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected Configuring enable_mem_arena to 1\n***"
        RET=1
    fi

    grep "Configuring enable_mem_pattern to 1" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected Configuring enable_mem_pattern to 1\n***"
        RET=1
    fi

    grep "Configuring memory.enable_memory_arena_shrinkage to cpu:0" $SERVER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed. Expected Configuring memory.enable_memory_arena_shrinkage to cpu:0\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
