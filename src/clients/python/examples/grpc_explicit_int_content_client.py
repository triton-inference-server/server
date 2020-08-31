#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import numpy as np

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "simple"
    model_version = ""
    batch_size = 1

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Generate the request
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version

    # Input data
    input0_data = [i for i in range(16)]
    input1_data = [1 for i in range(16)]

    # Populate the inputs in inference request
    input0 = service_pb2.ModelInferRequest().InferInputTensor()
    input0.name = "INPUT0"
    input0.datatype = "INT32"
    input0.shape.extend([1, 16])
    input0.contents.int_contents[:] = input0_data

    input1 = service_pb2.ModelInferRequest().InferInputTensor()
    input1.name = "INPUT1"
    input1.datatype = "INT32"
    input1.shape.extend([1, 16])
    input1.contents.int_contents[:] = input1_data
    request.inputs.extend([input0, input1])

    # Populate the outputs in the inference request
    output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output0.name = "OUTPUT0"

    output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output1.name = "OUTPUT1"
    request.outputs.extend([output0, output1])

    response = grpc_stub.ModelInfer(request)

    output_results = []
    index = 0
    for output in response.outputs:
        shape = []
        for value in output.shape:
            shape.append(value)
        output_results.append(
            np.frombuffer(response.raw_output_contents[index], dtype=np.int32))
        output_results[-1] = np.resize(output_results[-1], shape)
        index += 1

    if len(output_results) != 2:
        print("expected two output results")
        sys.exit(1)

    for i in range(16):
        print(
            str(input0_data[i]) + " + " + str(input1_data[i]) + " = " +
            str(output_results[0][0][i]))
        print(
            str(input0_data[i]) + " - " + str(input1_data[i]) + " = " +
            str(output_results[1][0][i]))
        if (input0_data[i] + input1_data[i]) != output_results[0][0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[i] - input1_data[i]) != output_results[1][0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    # Populating additional content field should generate an error
    request.raw_input_contents.extend([np.array(input0_data[0:8]).tobytes()])
    request.inputs[0].contents.int_contents[:] = input0_data[8:]

    try:
        response = grpc_stub.ModelInfer(request)
    except Exception as e:
        if "contents field must not be specified when using " \
            "raw_input_contents for 'INPUT0' for model 'simple'" \
                in e.__str__():
            print('PASS: explicit int')
