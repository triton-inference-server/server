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

import tritonhttpclient
from tritonclientutils.utils import InferenceServerException

def test_infer(model_name, input0_data, input1_data, headers=None):
    inputs = []
    outputs = []
    inputs.append(tritonhttpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(tritonhttpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
    outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT1',
                                                   binary_data=False))
    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=outputs,
                                  query_params=query_params,
                                  headers=headers)

    return results


def test_infer_no_outputs(model_name, input0_data, input1_data, headers=None):
    inputs = []
    inputs.append(tritonhttpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(tritonhttpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=None,
                                  query_params=query_params,
                                  headers=headers)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-c',
                        '--use_custom_model',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use custom model')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    FLAGS = parser.parse_args()
    try:
        triton_client = tritonhttpclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "simple_custom" if FLAGS.use_custom_model else "simple" 

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

    if FLAGS.http_headers is not None:
        headers_dict = {l.split(':')[0]: l.split(':')[1]
                        for l in FLAGS.http_headers}
    else:
        headers_dict = None

    # Infer with requested Outputs
    results = test_infer(model_name, input0_data, input1_data, headers_dict)
    print(results.get_response())

    statistics = triton_client.get_inference_statistics(model_name=model_name, headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Validate the results by comparing with precomputed values.
    output0_data = results.as_numpy('OUTPUT0')
    output1_data = results.as_numpy('OUTPUT1')
    for i in range(16):
        print(str(input0_data[0][i]) + " + " + str(input1_data[0][i]) + " = " +
              str(output0_data[0][i]))
        print(str(input0_data[0][i]) + " - " + str(input1_data[0][i]) + " = " +
              str(output1_data[0][i]))
        if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    # Infer without requested Outputs
    results = test_infer_no_outputs(model_name, input0_data, input1_data, headers=headers_dict)
    print(results.get_response())

    # Validate the results by comparing with precomputed values.
    output0_data = results.as_numpy('OUTPUT0')
    output1_data = results.as_numpy('OUTPUT1')
    for i in range(16):
        print(str(input0_data[0][i]) + " + " + str(input1_data[0][i]) + " = " +
              str(output0_data[0][i]))
        print(str(input0_data[0][i]) + " - " + str(input1_data[0][i]) + " = " +
              str(output1_data[0][i]))
        if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    # Infer with incorrect model name
    try:
        response = test_infer("wrong_model_name", input0_data,
                            input1_data).get_response()
        print("expected error message for wrong model name")
        sys.exit(1)
    except InferenceServerException as ex:
        print(ex)
        if not (ex.message().startswith("Request for unknown model")):
            print("improper error message for wrong model name")
            sys.exit(1)
