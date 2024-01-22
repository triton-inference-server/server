#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import concurrent.futures
import json
import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *


def infer_model_without_parameter_file():
    model_name = "addsub"
    shape = [4]

    with httpclient.InferenceServerClient("localhost:8000") as client:
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)
        inputs = [
            httpclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            httpclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
            httpclient.InferRequestedOutput("OUTPUT1"),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        output0_data = response.as_numpy("OUTPUT0")
        output1_data = response.as_numpy("OUTPUT1")

        print(
            "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
                input0_data, input1_data, output0_data
            )
        )
        print(
            "INPUT0 ({}) - INPUT1 ({}) = OUTPUT0 ({})".format(
                input0_data, input1_data, output1_data
            )
        )

        if not np.allclose(input0_data + input1_data, output0_data):
            print(model_name + " error: incorrect sum")
            return False

        if not np.allclose(input0_data - input1_data, output1_data):
            print(model_name + " error: incorrect difference")
            return False

        print("PASS: " + model_name)
        return True


def infer_model_with_parameter_file(batch_size, data_offset=0):
    model_name = "neuralnet"
    test_data_file = "neuralnet_test_data.json"
    np_dtype = np.single

    # prepare input data
    with open(test_data_file) as f:
        test_data = json.load(f)
    input_data = np.array(test_data["input_data"], dtype=np_dtype)
    input_data = input_data[data_offset : (data_offset + batch_size)]
    labels = test_data["labels"][data_offset : (data_offset + batch_size)]

    # inference
    with httpclient.InferenceServerClient("localhost:8000") as client:
        inputs = [
            httpclient.InferInput(
                "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)

        response = client.infer(model_name, inputs, request_id=str(1))
        output_data = response.as_numpy("OUTPUT")
        output_data_max = np.max(output_data, axis=1)

        print("Inference result: " + str(output_data))
        print("Inference result (max): " + str(output_data_max))
        print("Expected result: " + str(labels))

        if not np.all(np.isclose(np.max(output_data, axis=1), labels, atol=8)):
            print(model_name + " error: incorrect result")
            return False

    print("PASS: " + model_name)
    return True


def parallel_infer_a_full_dynamic_batch(max_batch_size):
    batch_size = 1
    success = True
    with concurrent.futures.ThreadPoolExecutor() as pool:
        threads = []
        for i in range(max_batch_size // batch_size):
            t = pool.submit(infer_model_with_parameter_file, batch_size, i)
            threads.append(t)
        for t in threads:
            success &= t.result()
    return success


if __name__ == "__main__":
    success = infer_model_without_parameter_file()
    success &= infer_model_with_parameter_file(batch_size=4)
    success &= parallel_infer_a_full_dynamic_batch(max_batch_size=8)
    if not success:
        sys.exit(1)
    sys.exit(0)
