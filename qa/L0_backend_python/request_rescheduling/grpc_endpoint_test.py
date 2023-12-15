#!/usr/bin/env python
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append("../../common")

# GRPC streaming helpers..
import queue
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class GrpcEndpointTest(tu.TestResultCollector):
    def test_grpc_decoupled(self, sequence_id=0, sequence_start=False):
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            # Reload the model to reset the flag
            triton_client.unload_model("iterative_sequence")
            triton_client.load_model("iterative_sequence")

            triton_client.start_stream(callback=partial(callback, user_data))
            inputs = []
            inputs.append(grpcclient.InferInput("IN", [1], "INT32"))
            inputs[0].set_data_from_numpy(np.array([3], dtype=np.int32))

            triton_client.async_stream_infer(
                model_name="iterative_sequence",
                inputs=inputs,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
            )
            res_count = 3
            while res_count > 0:
                data_item = user_data._completed_requests.get()
                res_count -= 1
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    self.assertEqual(res_count, data_item.as_numpy("OUT")[0])
            self.assertEqual(0, res_count)

    def test_grpc_non_decoupled(self, sequence_id=0, sequence_start=False):
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            # Reload the model to reset the flag
            triton_client.unload_model("request_rescheduling_addsub")
            triton_client.load_model("request_rescheduling_addsub")

            inputs = []
            inputs.append(grpcclient.InferInput("INPUT0", [16], "FP32"))
            inputs.append(grpcclient.InferInput("INPUT1", [16], "FP32"))
            input0_val = np.random.randn(*[16]).astype(np.float32)
            input1_val = np.random.randn(*[16]).astype(np.float32)
            inputs[0].set_data_from_numpy(input0_val)
            inputs[1].set_data_from_numpy(input1_val)

            results = triton_client.infer(
                model_name="request_rescheduling_addsub",
                inputs=inputs,
            )

            output0_data = results.as_numpy("OUTPUT0")
            output1_data = results.as_numpy("OUTPUT1")

            self.assertTrue(np.array_equal(output0_data, input0_val + input1_val))
            self.assertTrue(np.array_equal(output1_data, input0_val - input1_val))


if __name__ == "__main__":
    unittest.main()
