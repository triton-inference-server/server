#!/usr/bin/python

# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Reuse a single client for all sync tests
    with client_util.InferenceServerClient(FLAGS.url,
                                           verbose=FLAGS.verbose) as client:
        for model_name, np_dtype, shape in (
                # yapf: disable
            ("identity_int32", np.int32, [0]),
            ("identity_int32", np.int32, [7])):
            # yapf: enable
            if np_dtype != object:
                input_data = (16384 * np.random.randn(*shape)).astype(np_dtype)
            else:
                in0 = (16384 * np.ones(shape, dtype='int'))
                in0n = np.array([str(x) for x in in0.reshape(in0.size)],
                                dtype=object)
                input_data = in0n.reshape(in0.shape)
            inputs = [
                client_util.InferInput("INPUT0", input_data.shape,
                                       np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)

            results = client.infer(model_name, inputs)
            print(results)

            # Make sure outputs are expected value
            output_data = results.as_numpy("OUTPUT0")
            if output_data is None:
                print("error: expected 'OUTPUT0'")
                sys.exit(1)

            if np_dtype == object:
                output_data = np.char.decode(output_data)

            if not np.array_equal(output_data, input_data):
                print("error: expected output {} to match input {}".format(
                    output_data, input_data))
                sys.exit(1)
