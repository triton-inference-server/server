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

import queue
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


# This client sends a single request to the model with the
# following tensor data. In compliance with the behavior
# of repeat_int32 model, it will expect the 4 responses
# with output: [4], [2], [0] and [1] respectively.
model_name = "repeat_int32"
in_value = [4, 2, 0, 1]
delay_value = [1, 2, 3, 4]
wait_value = 5

inputs = []
inputs.append(grpcclient.InferInput("IN", [len(in_value)], "INT32"))
inputs.append(grpcclient.InferInput("DELAY", [len(delay_value)], "UINT32"))
inputs.append(grpcclient.InferInput("WAIT", [1], "UINT32"))

outputs = []
outputs.append(grpcclient.InferRequestedOutput("OUT"))
outputs.append(grpcclient.InferRequestedOutput("IDX"))

user_data = UserData()

with grpcclient.InferenceServerClient(
    url="localhost:8001", verbose=True
) as triton_client:
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))

    in_data = np.array(in_value, dtype=np.int32)
    inputs[0].set_data_from_numpy(in_data)
    delay_data = np.array(delay_value, dtype=np.uint32)
    inputs[1].set_data_from_numpy(delay_data)
    wait_data = np.array([wait_value], dtype=np.uint32)
    inputs[2].set_data_from_numpy(wait_data)

    request_id = "0"
    triton_client.async_stream_infer(
        model_name=model_name,
        inputs=inputs,
        request_id=request_id,
        outputs=outputs,
    )

    # Retrieve results...
    recv_count = 0
    expected_count = len(in_value)
    result_dict = {}
    while recv_count < expected_count:
        data_item = user_data._completed_requests.get()
        if type(data_item) == InferenceServerException:
            raise data_item
        else:
            this_id = data_item.get_response().id
            if this_id not in result_dict.keys():
                result_dict[this_id] = []
            result_dict[this_id].append((recv_count, data_item))

        recv_count += 1

    # Validate results...
    if len(result_dict[request_id]) != len(in_value):
        print(
            "expected {} many responses for request id {}, got {}".format(
                len(in_value), request_id, len(result_dict[request_id])
            )
        )
        sys.exit(1)

    result_list = result_dict[request_id]
    for i in range(len(result_list)):
        expected_data = np.array([in_value[i]], dtype=np.int32)
        this_data = result_list[i][1].as_numpy("OUT")
        if not np.array_equal(expected_data, this_data):
            print(
                "incorrect data: expected {}, got {}".format(expected_data, this_data)
            )
            sys.exit(1)

    print("PASS: repeat_int32")
    sys.exit(0)
