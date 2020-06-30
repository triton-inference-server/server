#!/bin/bash
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

import sys
sys.path.append("../common")

from functools import partial
import numpy as np
import queue
import unittest
import os

import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class DecoupledTest(unittest.TestCase):
    def setUp(self):
        self.model_name_ = "repeat_int32"

        self.inputs_ = []
        self.inputs_.append(grpcclient.InferInput('IN', [1, 1], "INT32"))
        self.inputs_.append(grpcclient.InferInput('DELAY', [1, 1], "UINT32"))
        self.inputs_.append(grpcclient.InferInput('WAIT', [1, 1], "UINT32"))

        self.outputs_ = []
        self.outputs_.append(grpcclient.InferRequestedOutput('OUT'))

    def _decoupled_infer(self,
                         request_count,
                         repeat_count=1,
                         data_offset=100,
                         delay_time=1000,
                         wait_time=500):
        # Initialize data for IN
        input_data = np.arange(start=data_offset,
                               stop=data_offset + repeat_count,
                               dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        self.inputs_[0].set_shape([1, repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        delay_data = (np.ones([1, repeat_count], dtype=np.uint32)) * delay_time
        self.inputs_[1].set_shape([1, repeat_count])
        self.inputs_[1].set_data_from_numpy(delay_data)

        # Initialize data for WAIT
        wait_data = np.array([[wait_time]], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        user_data = UserData()
        result_dict = {}

        with grpcclient.InferenceServerClient(url="localhost:8001",
                                              verbose=True) as triton_client:
            # Establish stream
            triton_client.start_stream(callback=partial(callback, user_data))
            # Send specified many requests in parallel
            for i in range(request_count):
                triton_client.async_stream_infer(model_name=self.model_name_,
                                                 inputs=self.inputs_,
                                                 request_id=str(i),
                                                 outputs=self.outputs_)

            # Retrieve results...
            recv_count = 0
            while recv_count < (repeat_count * request_count):
                data_item = user_data._completed_requests.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    if this_id not in result_dict.keys():
                        result_dict[this_id] = []
                    result_dict[this_id].append(data_item.as_numpy('OUT'))
                recv_count += 1

        # Validate the results..
        for i in range(request_count):
            this_id = str(i)
            if repeat_count != 0 and this_id not in result_dict.keys():
                self.assertTrue(
                    False, "response for request id {} not received".format(this_id))
            elif repeat_count == 0 and this_id in result_dict.keys():
                self.assertTrue(
                    False,
                    "received unexpected response for request id {}".format(this_id))
            if repeat_count != 0:
                self.assertEqual(len(result_dict[this_id]), repeat_count)

                expected_data = data_offset
                result_list = result_dict[this_id]
                for j in range(len(result_list)):
                    self.assertEqual(len(result_list[j]), 1)
                    self.assertEqual(result_list[j][0], expected_data)
                    expected_data += 1

    def test_one_to_none(self):
        # Tests cases where each request generates no response.
        # Note the name of the test one_to_none implies the
        # mapping between requests and responses.

        # Single request case
        self._decoupled_infer(request_count=1, repeat_count=0)
        # Multiple request case
        self._decoupled_infer(request_count=10, repeat_count=0)

    def test_one_to_one(self):
        # Tests cases where each request generates single response.
        # Note the name of the test one_to_one implies the
        # mapping between requests and responses.

        # Single request case
        # Release request before the response is delivered
        self._decoupled_infer(request_count=1, wait_time=500)
        # Release request after the response is delivered
        self._decoupled_infer(request_count=1, wait_time=2000)

        # Multiple request case
        # Release request before the response is delivered
        self._decoupled_infer(request_count=10, wait_time=500)
        # Release request after the response is delivered
        self._decoupled_infer(request_count=10, wait_time=2000)

    def test_one_to_many(self):
        # Tests cases where each request generates multiple response.
        # Note the name of the test one_to_many implies the
        # mapping between requests and responses.

        self.assertFalse("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        # Single request case
        # Release request before the first response is delivered
        self._decoupled_infer(request_count=1, repeat_count=5, wait_time=500)
        # Release request when the responses are getting delivered
        self._decoupled_infer(request_count=1, repeat_count=5, wait_time=2000)
        # Release request after all the responses are delivered
        self._decoupled_infer(request_count=1,
                              repeat_count=5,
                              wait_time=10000)

        # Multiple request case
        # Release request before the first response is delivered
        self._decoupled_infer(request_count=10, repeat_count=5, wait_time=500)
        # Release request when the responses are getting delivered
        self._decoupled_infer(request_count=10,
                              repeat_count=5,
                              wait_time=2000)
        # Release request after all the responses are delivered
        self._decoupled_infer(request_count=10,
                              repeat_count=5,
                              wait_time=10000)

    def test_one_to_multi_many(self):
        # Tests cases where each request generates multiple response but the
        # responses are delayed so as to stress the control path handling the
        # queued responses.

        self.assertTrue("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        # Single request case
        # Release request before the first response is delivered
        self._decoupled_infer(request_count=1, repeat_count=5, wait_time=500)
        # Release request when the responses are getting delivered
        self._decoupled_infer(request_count=1, repeat_count=5, wait_time=8000)
        # Release request after all the responses are delivered
        self._decoupled_infer(request_count=1,
                              repeat_count=5,
                              wait_time=20000)

        # Multiple request case
        # Release request before the first response is delivered
        self._decoupled_infer(request_count=10, repeat_count=5, wait_time=500)
        # Release request when the responses are getting delivered
        self._decoupled_infer(request_count=10,
                              repeat_count=5,
                              wait_time=8000)
        # Release request after all the responses are delivered
        self._decoupled_infer(request_count=10,
                              repeat_count=5,
                              wait_time=20000)

    def _no_streaming_helper(self, protocol):
        data_offset = 100
        repeat_count = 1
        delay_time = 1000
        wait_time = 2000

        input_data = np.arange(start=data_offset,
                               stop=data_offset + repeat_count,
                               dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        delay_data = (np.ones([1, repeat_count], dtype=np.uint32)) * delay_time
        wait_data = np.array([[wait_time]], dtype=np.uint32)

        if protocol is "grpc":
            # Use the inputs and outputs from the setUp
            this_inputs = self.inputs_
            this_outputs = self.outputs_
        else:
            this_inputs = []
            this_inputs.append(
                httpclient.InferInput('IN', [1, repeat_count], "INT32"))
            this_inputs.append(httpclient.InferInput('DELAY', [1, 1], "UINT32"))
            this_inputs.append(httpclient.InferInput('WAIT', [1, 1], "UINT32"))
            this_outputs = []
            this_outputs.append(httpclient.InferRequestedOutput('OUT'))

        # Initialize data for IN
        this_inputs[0].set_shape([1, repeat_count])
        this_inputs[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        this_inputs[1].set_shape([1, repeat_count])
        this_inputs[1].set_data_from_numpy(delay_data)

        # Initialize data for WAIT
        this_inputs[2].set_data_from_numpy(wait_data)

        if protocol is "grpc":
            triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True)
        else:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True)
        try:
            triton_client.infer(model_name=self.model_name_,
                                      inputs=this_inputs,
                                      outputs=this_outputs)
            self.assertTrue(
                False, "async_infer is expected to fail for decoupled models")
        except InferenceServerException as ex:
            self.assertTrue(
                "doesn't support models with decoupled transaction policy"
                in ex.message())

    def test_no_streaming(self):
        # Tests cases with no streaming inference. Server should give
        # appropriate error in such cases.

        self._no_streaming_helper("grpc")
        self._no_streaming_helper("http")


if __name__ == '__main__':
    unittest.main()
