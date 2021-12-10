#!/usr/bin/python

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
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

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
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Name of model.')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    # Run the custom_modulo model, which depends on a custom mod operation
    model_name = FLAGS.model
    shape = (3, 5)
    dtype = np.float32

    # Create the inference context for the model.
    client = client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)

    # Create the data for one input tensor.
    input_data = []
    input_data.append(np.ones((3, 5), dtype=np.float32))
    input_data.append(np.ones((3, 5), dtype=np.float32))

    inputs = []
    for i in range(len(input_data)):
        inputs.append(
            client_util.InferInput("input_{}".format(i + 1), shape,
                                   np_to_triton_dtype(dtype)))
        inputs[i].set_data_from_numpy(input_data[i])

    results = client.infer(model_name, inputs)

    # We expect 1 result of size 10 with alternating 1 and 0.
    output_data = results.as_numpy('output')
    if output_data is None:
        print("error: expected 'output'")
        sys.exit(1)

    for i in range(3):
        for j in range(5):
            print(
                str(input_data[0][i][j]) + " + " + str(input_data[1][i][j]) +
                " = " + str(output_data[i][j]))
            if ((input_data[0][i][j] + input_data[1][i][j]) !=
                    output_data[i][j]):
                print("error: incorrect value")
                sys.exit(1)
