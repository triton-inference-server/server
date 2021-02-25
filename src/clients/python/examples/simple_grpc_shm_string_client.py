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
import sys
from builtins import range

import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.shared_memory as shm

FLAGS = None

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

    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # To make sure no shared memory regions are registered with the
    # server.
    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    # We use a simple model that takes 2 input tensors of 16 strings
    # each and returns 2 output tensors of 16 strings each. The input
    # strings must represent integers. One output tensor is the
    # element-wise sum of the inputs and one output is the element-wise
    # difference.
    model_name = "simple_string"
    model_version = ""

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    in0 = np.arange(start=0, stop=16, dtype=np.int32)
    in0n = np.array([str(x).encode('utf-8') for x in in0.flatten()],
                    dtype=object)
    input0_data = in0n.reshape(in0.shape)
    in1 = np.ones(shape=16, dtype=np.int32)
    in1n = np.array([str(x).encode('utf-8') for x in in1.flatten()],
                    dtype=object)
    input1_data = in1n.reshape(in1.shape)

    expected_sum = np.array(
        [str(x).encode('utf-8') for x in np.add(in0, in1).flatten()],
        dtype=object)
    expected_diff = np.array(
        [str(x).encode('utf-8') for x in np.subtract(in0, in1).flatten()],
        dtype=object)
    expected_sum_serialized = utils.serialize_byte_tensor(expected_sum)
    expected_diff_serialized = utils.serialize_byte_tensor(expected_diff)

    input0_data_serialized = utils.serialize_byte_tensor(input0_data)
    input1_data_serialized = utils.serialize_byte_tensor(input1_data)
    input0_byte_size = utils.serialized_byte_size(input0_data_serialized)
    input1_byte_size = utils.serialized_byte_size(input1_data_serialized)
    output0_byte_size = utils.serialized_byte_size(expected_sum_serialized)
    output1_byte_size = utils.serialized_byte_size(expected_diff_serialized)
    output_byte_size = max(input0_byte_size, input1_byte_size) + 1

    # Create Output0 and Output1 in Shared Memory and store shared memory handles
    shm_op0_handle = shm.create_shared_memory_region("output0_data",
                                                     "/output0_simple",
                                                     output0_byte_size)
    shm_op1_handle = shm.create_shared_memory_region("output1_data",
                                                     "/output1_simple",
                                                     output1_byte_size)

    # Register Output0 and Output1 shared memory with Triton Server
    triton_client.register_system_shared_memory("output0_data",
                                                "/output0_simple",
                                                output0_byte_size)
    triton_client.register_system_shared_memory("output1_data",
                                                "/output1_simple",
                                                output1_byte_size)

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle = shm.create_shared_memory_region("input0_data",
                                                     "/input0_simple",
                                                     input0_byte_size)
    shm_ip1_handle = shm.create_shared_memory_region("input1_data",
                                                     "/input1_simple",
                                                     input1_byte_size)

    # Put input data values into shared memory
    shm.set_shared_memory_region(shm_ip0_handle, [input0_data_serialized])
    shm.set_shared_memory_region(shm_ip1_handle, [input1_data_serialized])

    # Register Input0 and Input1 shared memory with Triton Server
    triton_client.register_system_shared_memory("input0_data", "/input0_simple",
                                                input0_byte_size)
    triton_client.register_system_shared_memory("input1_data", "/input1_simple",
                                                input1_byte_size)

    # Set the parameters to use data from shared memory
    inputs = []
    inputs.append(grpcclient.InferInput('INPUT0', [1, 16], "BYTES"))
    inputs[-1].set_shared_memory("input0_data", input0_byte_size)

    inputs.append(grpcclient.InferInput('INPUT1', [1, 16], "BYTES"))
    inputs[-1].set_shared_memory("input1_data", input1_byte_size)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT0'))
    outputs[-1].set_shared_memory("output0_data", output0_byte_size)

    outputs.append(grpcclient.InferRequestedOutput('OUTPUT1'))
    outputs[-1].set_shared_memory("output1_data", output1_byte_size)

    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    # Read results from the shared memory.
    output0 = results.get_output("OUTPUT0")
    print(utils.triton_to_np_dtype(output0.datatype))
    if output0 is not None:
        output0_data = shm.get_contents_as_numpy(
            shm_op0_handle, utils.triton_to_np_dtype(output0.datatype),
            output0.shape)
    else:
        print("OUTPUT0 is missing in the response.")
        sys.exit(1)

    output1 = results.get_output("OUTPUT1")
    if output1 is not None:
        output1_data = shm.get_contents_as_numpy(
            shm_op1_handle, utils.triton_to_np_dtype(output1.datatype),
            output1.shape)
    else:
        print("OUTPUT1 is missing in the response.")
        sys.exit(1)

    for i in range(16):
        r0 = output0_data[0][i]
        r1 = output1_data[0][i]
        print(
            str(input0_data[i]) + " + " + str(input1_data[i]) + " = " + str(r0))
        print(
            str(input0_data[i]) + " - " + str(input1_data[i]) + " = " + str(r1))

        if expected_sum[i] != r0:
            print("shm infer error: incorrect sum")
            sys.exit(1)
        if expected_diff[i] != r1:
            print("shm infer error: incorrect difference")
            sys.exit(1)

    print(triton_client.get_system_shared_memory_status())
    triton_client.unregister_system_shared_memory()
    assert len(shm.mapped_shared_memory_regions()) == 4
    shm.destroy_shared_memory_region(shm_ip0_handle)
    shm.destroy_shared_memory_region(shm_ip1_handle)
    shm.destroy_shared_memory_region(shm_op0_handle)
    shm.destroy_shared_memory_region(shm_op1_handle)
    assert len(shm.mapped_shared_memory_regions()) == 0

    print('PASS: system shared memory')
