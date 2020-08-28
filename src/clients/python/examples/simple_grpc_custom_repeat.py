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

from functools import partial
import argparse
import numpy as np
import sys
import queue

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

FLAGS = None


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.'
    )

    FLAGS = parser.parse_args()

    # We use the custom "repeat_int32" model which takes 3 inputs and
    # 1 output. For a single request the model will generate 'repeat_count'
    # responses. See is src/backends/backend/examples/repeat.cc.
    model_name = "repeat_int32"
    model_version = ""
    repeat_count = 10
    data_offset = 100
    delay_time = 1000
    wait_time = 1000

    # Generate the data
    input_data = np.arange(start=data_offset,
                           stop=data_offset + repeat_count,
                           dtype=np.int32)
    delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
    wait_data = np.array([wait_time], dtype=np.uint32)

    # Initialize the data.
    inputs = []
    inputs.append(grpcclient.InferInput('IN', [repeat_count], "INT32"))
    inputs[-1].set_data_from_numpy(input_data)
    inputs.append(grpcclient.InferInput('DELAY', [repeat_count], "UINT32"))
    inputs[-1].set_data_from_numpy(delay_data)
    inputs.append(grpcclient.InferInput('WAIT', [1], "UINT32"))
    inputs[-1].set_data_from_numpy(wait_data)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('OUT'))

    result_list = []

    user_data = UserData()

    # It is advisable to use client object within with..as clause
    # when sending streaming requests. This ensures the client
    # is closed when the block inside with exits.
    with grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
        try:
            # Establish stream
            triton_client.start_stream(callback=partial(callback, user_data))
            # Send a single inference request
            triton_client.async_stream_infer(model_name=model_name,
                                             inputs=inputs,
                                             outputs=outputs)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        # Retrieve results...
        recv_count = 0
        while recv_count < repeat_count:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                print(data_item)
                sys.exit(1)
            else:
                result_list.append(data_item.as_numpy('OUT'))
            recv_count = recv_count + 1

    expected_data = data_offset
    for i in range(len(result_list)):
        if (len(result_list[i]) != 1):
            print(
                "unexpected number of elements in the output, expected 1, got {}"
                .format(len(result_list[i])))
            sys.exit(1)
        print("{} : {}".format(result_list[i][0], expected_data))
        if (result_list[i][0] != expected_data):
            print("mismatch in the results")
            sys.exit(1)
        expected_data += 1
    print("PASS: Decoupled API")
