# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
sys.path.append("../common")

import argparse
from builtins import range
import threading
import numpy as np
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient

FLAGS = None
EXIT_CODE_ = 0


def infer(protocol, url, verbose):
    global EXIT_CODE_

    client_util = httpclient if protocol == "http" else grpcclient
    request_count = 3
    try:
        client = client_util.InferenceServerClient(url, verbose=verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)

    model_name = "simple"

    # Infer
    inputs = []
    outputs = []
    inputs.append(client_util.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(client_util.InferInput('INPUT1', [1, 16], "INT32"))

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(client_util.InferRequestedOutput('OUTPUT0'))
    outputs.append(client_util.InferRequestedOutput('OUTPUT1'))

    results = []
    for i in range(request_count):
        results.append(client.infer(model_name, inputs, outputs=outputs))

    for result in results:
        output0_data = result.as_numpy('OUTPUT0')
        output1_data = result.as_numpy('OUTPUT1')
        for i in range(16):
            print(str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
                  " = " + str(output0_data[0][i]))
            print(str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
                  " = " + str(output1_data[0][i]))
            if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
                print("sync infer error: incorrect sum")
                EXIT_CODE_ = 1
            if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
                print("sync infer error: incorrect difference")
                EXIT_CODE_ = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-t',
                        '--concurrency',
                        type=int,
                        required=False,
                        default=8,
                        help='Request concurrency. Default is 8.')
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
    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    threads = []

    for idx in range(FLAGS.concurrency):
        thread_name = "thread_{}".format(idx)

        threads.append(
            threading.Thread(target=infer,
                             args=(FLAGS.protocol, FLAGS.url, FLAGS.verbose)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    sys.exit(EXIT_CODE_)
