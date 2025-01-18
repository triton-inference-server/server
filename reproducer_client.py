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
import sys

import numpy as np
import tritonclient.grpc as grpcclient

def _range_repr_dtype(dtype):
    if dtype == np.float64:
        return np.int32
    elif dtype == np.float32:
        return np.int16
    elif dtype == np.float16:
        return np.int8
    elif dtype == np.object_:  # TYPE_STRING
        return np.int32
    return dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "plan_int8_int8_int8"
    input_dtype = np.int8
    output0_dtype = np.int8
    output1_dtype = np.int8
    tensor_shape = [1, 16, 1, 1]
    # Returns only top 3 classes with the largest value.
    # For more info see here: https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md
    class_count = 3

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("INPUT0", tensor_shape, "INT8"))
    inputs.append(grpcclient.InferInput("INPUT1", tensor_shape, "INT8"))


    # outputs are sum and difference of inputs so set max input
    # values so that they will not overflow the output. This
    # allows us to do an exact match. For float types use 8, 16,
    # 32 int range for fp 16, 32, 64 respectively. When getting
    # class outputs the result value/probability is returned as a
    # float so must use fp32 range in that case.

    # requesting the class output hence the output should be set to false.
    output0_raw = False
    output1_raw = False

    rinput_dtype = _range_repr_dtype(input_dtype)
    routput0_dtype = _range_repr_dtype(output0_dtype if output0_raw else np.float32)
    routput1_dtype = _range_repr_dtype(output1_dtype if output1_raw else np.float32)
    val_min = (
        max(
            np.iinfo(rinput_dtype).min,
            np.iinfo(routput0_dtype).min,
            np.iinfo(routput1_dtype).min,
        )
        / 2
    )
    val_max = (
        min(
            np.iinfo(rinput_dtype).max,
            np.iinfo(routput0_dtype).max,
            np.iinfo(routput1_dtype).max,
        )
        / 2
    )

    input0_array = np.random.randint(
        low=val_min, high=val_max, size=tensor_shape, dtype=rinput_dtype
    )
    input1_array = np.random.randint(
        low=val_min, high=val_max, size=tensor_shape, dtype=rinput_dtype
    )
    if input_dtype != np.object_:
        input0_array = input0_array.astype(input_dtype)
        input1_array = input1_array.astype(input_dtype)

    expected_output0_array = input0_array - input1_array
    expected_output1_array = input0_array + input1_array

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_array)
    inputs[1].set_data_from_numpy(input1_array)

    outputs.append(grpcclient.InferRequestedOutput("OUTPUT0", class_count = class_count))
    outputs.append(grpcclient.InferRequestedOutput("OUTPUT1", class_count = class_count))

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )

    # Get the output arrays from the results
    expected0_flatten = expected_output0_array.flatten()
    expected1_flatten = expected_output1_array.flatten()

    expected0_sort_idx = [
        np.flip(np.argsort(x.flatten()), 0)
            for x in expected_output0_array.reshape(tensor_shape)
        ]
    expected1_sort_idx = [
        np.flip(np.argsort(x.flatten()), 0)
            for x in expected_output1_array.reshape(tensor_shape)
        ]
    
    for result_name in ["OUTPUT0", "OUTPUT1"]:
        class_list = results.as_numpy(result_name)[0]
        assert len(class_list) == class_count

        for idx, class_label in enumerate(class_list):
            # can't compare indices since could have different
            # indices with the same value/prob, so check that
            # the value of each index equals the expected value.
            # Only compare labels when the indices are equal.
            if type(class_label) == str:
                ctuple = class_label.split(":")
            else:
                ctuple = "".join(chr(x) for x in class_label).split(":")
            cval = float(ctuple[0])
            cidx = int(ctuple[1])
            if result_name == 'OUTPUT0':
                assert cval == expected0_flatten[cidx], f"Got mismatch {cval}, expected {expected0_flatten[cidx]}"
                assert cval == expected0_flatten[expected0_sort_idx[0][idx]], f"Got mismatch {cval}, expected {expected0_flatten[expected0_sort_idx[0][idx]]}"
                if cidx == expected0_sort_idx[0][idx]:
                    assert ctuple[2].strip("\r") == "label{}".format(expected0_sort_idx[0][idx])
            elif result_name == 'OUTPUT1':
                assert cval == expected1_flatten[cidx]
                assert cval == expected1_flatten[expected1_sort_idx[0][idx]]

    print("Execution Succesfull!")