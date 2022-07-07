#!/bin/bash
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
import time
import test_util as tu

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


class DecoupledTest(tu.TestResultCollector):

    def setUp(self):
        self.trials_ = [("repeat_int32", None), ("simple_repeat", None),
                        ("sequence_repeat", None),
                        ("fan_repeat", self._fan_validate),
                        ("repeat_square", self._nested_validate),
                        ("nested_square", self._nested_validate)]
        self.model_name_ = "repeat_int32"

        self.inputs_ = []
        self.inputs_.append(grpcclient.InferInput('IN', [1], "INT32"))
        self.inputs_.append(grpcclient.InferInput('DELAY', [1], "UINT32"))
        self.inputs_.append(grpcclient.InferInput('WAIT', [1], "UINT32"))

        self.outputs_ = []
        self.outputs_.append(grpcclient.InferRequestedOutput('OUT'))
        self.outputs_.append(grpcclient.InferRequestedOutput('IDX'))
        # Some trials only expect a subset of outputs
        self.requested_outputs_ = self.outputs_

    def _stream_infer(self, request_count, request_delay, expected_count,
                      delay_data, delay_factor, user_data, result_dict):
        with grpcclient.InferenceServerClient(url="localhost:8001",
                                              verbose=True) as triton_client:
            # Establish stream
            triton_client.start_stream(callback=partial(callback, user_data))
            # Send specified many requests in parallel
            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                triton_client.async_stream_infer(
                    model_name=self.model_name_,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_)
                # Update delay input in accordance with the scaling factor
                delay_data = delay_data * delay_factor
                delay_data = delay_data.astype(np.uint32)

            # Retrieve results...
            recv_count = 0
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

    def _fan_validate(self, result_list, data_offset, repeat_count):
        # fan_repeat returns "2 * data_offset" as result
        self.assertEqual(len(result_list), repeat_count)
        expected_data = 2 * data_offset
        for j in range(len(result_list)):
            this_data = result_list[j][1].as_numpy('OUT')
            self.assertEqual(len(this_data), 1)
            self.assertEqual(this_data[0], expected_data)
            expected_data += 2

    def _nested_validate(self, result_list, data_offset, repeat_count):
        # if repeat model returns repeat result n, repeat_square-like model
        # will return the same result n times
        expected_len = sum(
            x for x in range(data_offset, data_offset + repeat_count))
        self.assertEqual(len(result_list), expected_len)
        expected_data = data_offset
        expected_count = expected_data
        for j in range(len(result_list)):
            this_data = result_list[j][1].as_numpy('OUT')
            self.assertEqual(len(this_data), 1)
            self.assertEqual(this_data[0], expected_data)
            expected_count -= 1
            if expected_count == 0:
                expected_data += 1
                expected_count = expected_data

    def _decoupled_infer(self,
                         request_count,
                         request_delay=0,
                         repeat_count=1,
                         data_offset=100,
                         delay_time=1000,
                         delay_factor=1,
                         wait_time=500,
                         order_sequence=None,
                         validate_fn=None):
        # Initialize data for IN
        input_data = np.arange(start=data_offset,
                               stop=data_offset + repeat_count,
                               dtype=np.int32)
        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        self.inputs_[1].set_shape([repeat_count])

        # Initialize data for WAIT
        wait_data = np.array([wait_time], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        # use validate_fn to differentiate requested outputs
        self.requested_outputs_ = self.outputs_ if validate_fn is None else self.outputs_[
            0:1]

        user_data = UserData()
        result_dict = {}

        try:
            if "square" not in self.model_name_:
                expected_count = (repeat_count * request_count)
            else:
                expected_count = sum(
                    x for x in range(data_offset, data_offset +
                                     repeat_count)) * request_count
            self._stream_infer(request_count, request_delay, expected_count,
                               delay_data, delay_factor, user_data, result_dict)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Validate the results..
        for i in range(request_count):
            this_id = str(i)
            if repeat_count != 0 and this_id not in result_dict.keys():
                self.assertTrue(
                    False,
                    "response for request id {} not received".format(this_id))
            elif repeat_count == 0 and this_id in result_dict.keys():
                self.assertTrue(
                    False,
                    "received unexpected response for request id {}".format(
                        this_id))
            if repeat_count != 0:
                if validate_fn is None:
                    self.assertEqual(len(result_dict[this_id]), repeat_count)
                    expected_data = data_offset
                    result_list = result_dict[this_id]
                    for j in range(len(result_list)):
                        if order_sequence is not None:
                            self.assertEqual(result_list[j][0],
                                             order_sequence[i][j])
                        this_data = result_list[j][1].as_numpy('OUT')
                        self.assertEqual(len(this_data), 1)
                        self.assertEqual(this_data[0], expected_data)
                        this_idx = result_list[j][1].as_numpy('IDX')
                        self.assertEqual(len(this_idx), 1)
                        self.assertEqual(this_idx[0], j)
                        expected_data += 1
                else:
                    validate_fn(result_dict[this_id], data_offset, repeat_count)

    def test_one_to_none(self):
        # Test cases where each request generates no response.
        # Note the name of the test one_to_none implies the
        # mapping between requests and responses.

        for trial in self.trials_:
            self.model_name_ = trial[0]
            # Single request case
            self._decoupled_infer(request_count=1,
                                  repeat_count=0,
                                  validate_fn=trial[1])
            # Multiple request case
            self._decoupled_infer(request_count=5,
                                  repeat_count=0,
                                  validate_fn=trial[1])

    def test_one_to_one(self):
        # Test cases where each request generates single response.
        # Note the name of the test one_to_one implies the
        # mapping between requests and responses.

        for trial in self.trials_:
            self.model_name_ = trial[0]
            # Single request case
            # Release request before the response is delivered
            self._decoupled_infer(request_count=1,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request after the response is delivered
            self._decoupled_infer(request_count=1,
                                  wait_time=2000,
                                  validate_fn=trial[1])

            # Multiple request case
            # Release request before the response is delivered
            self._decoupled_infer(request_count=5,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request after the response is delivered
            self._decoupled_infer(request_count=5,
                                  wait_time=2000,
                                  validate_fn=trial[1])

    def test_one_to_many(self):
        # Test cases where each request generates multiple response.
        # Note the name of the test one_to_many implies the
        # mapping between requests and responses.

        self.assertFalse("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]
            # Single request case
            # Release request before the first response is delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request when the responses are getting delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=2000,
                                  validate_fn=trial[1])
            # Release request after all the responses are delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=10000,
                                  validate_fn=trial[1])

            # Multiple request case
            # Release request before the first response is delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request when the responses are getting delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=2000,
                                  validate_fn=trial[1])
            # Release request after all the responses are delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=10000,
                                  validate_fn=trial[1])

    def test_one_to_multi_many(self):
        # Test cases where each request generates multiple response but the
        # responses are delayed so as to stress the control path handling the
        # queued responses.

        self.assertTrue("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]
            # Single request case
            # Release request before the first response is delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request when the responses are getting delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=8000,
                                  validate_fn=trial[1])
            # Release request after all the responses are delivered
            self._decoupled_infer(request_count=1,
                                  repeat_count=5,
                                  wait_time=20000,
                                  validate_fn=trial[1])

            # Multiple request case
            # Release request before the first response is delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=500,
                                  validate_fn=trial[1])
            # Release request when the responses are getting delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=3000,
                                  validate_fn=trial[1])
            # Release request after all the responses are delivered
            self._decoupled_infer(request_count=5,
                                  repeat_count=5,
                                  wait_time=10000,
                                  validate_fn=trial[1])

    def test_response_order(self):
        # Test the expected response order for different cases

        self.assertFalse("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]

            # Case 1: Interleaved responses
            self._decoupled_infer(request_count=2,
                                  request_delay=500,
                                  repeat_count=4,
                                  order_sequence=[[0, 2, 4, 6], [1, 3, 5, 7]],
                                  validate_fn=trial[1])

            # Case 2: All responses of second request delivered before any
            # response from the first
            self._decoupled_infer(request_count=2,
                                  request_delay=500,
                                  repeat_count=4,
                                  delay_time=2000,
                                  delay_factor=0.1,
                                  order_sequence=[[4, 5, 6, 7], [0, 1, 2, 3]],
                                  validate_fn=trial[1])

            # Case 3: Similar to Case 2, but the second request is generated
            # after the first response from first request is received
            self._decoupled_infer(request_count=2,
                                  request_delay=2500,
                                  repeat_count=4,
                                  delay_time=2000,
                                  delay_factor=0.1,
                                  order_sequence=[[0, 5, 6, 7], [1, 2, 3, 4]],
                                  validate_fn=trial[1])

            # Case 4: All the responses of second requests are dleivered after
            # all the responses from first requests are received
            self._decoupled_infer(request_count=2,
                                  request_delay=100,
                                  repeat_count=4,
                                  delay_time=500,
                                  delay_factor=10,
                                  order_sequence=[[0, 1, 2, 3], [4, 5, 6, 7]],
                                  validate_fn=trial[1])

            # Case 5: Similar to Case 4, but the second request is generated
            # after the first response from the first request is received
            self._decoupled_infer(request_count=2,
                                  request_delay=750,
                                  repeat_count=4,
                                  delay_time=500,
                                  delay_factor=10,
                                  order_sequence=[[0, 1, 2, 3], [4, 5, 6, 7]],
                                  validate_fn=trial[1])

    def _no_streaming_helper(self, protocol):
        data_offset = 100
        repeat_count = 1
        delay_time = 1000
        wait_time = 2000

        input_data = np.arange(start=data_offset,
                               stop=data_offset + repeat_count,
                               dtype=np.int32)
        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        wait_data = np.array([wait_time], dtype=np.uint32)

        if protocol == "grpc":
            # Use the inputs and outputs from the setUp
            this_inputs = self.inputs_
            this_outputs = self.outputs_
        else:
            this_inputs = []
            this_inputs.append(
                httpclient.InferInput('IN', [repeat_count], "INT32"))
            this_inputs.append(httpclient.InferInput('DELAY', [1], "UINT32"))
            this_inputs.append(httpclient.InferInput('WAIT', [1], "UINT32"))
            this_outputs = []
            this_outputs.append(httpclient.InferRequestedOutput('OUT'))

        # Initialize data for IN
        this_inputs[0].set_shape([repeat_count])
        this_inputs[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        this_inputs[1].set_shape([repeat_count])
        this_inputs[1].set_data_from_numpy(delay_data)

        # Initialize data for WAIT
        this_inputs[2].set_data_from_numpy(wait_data)

        if protocol == "grpc":
            triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True)
        else:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True)

        with self.assertRaises(InferenceServerException) as cm:
            triton_client.infer(model_name=self.model_name_,
                                inputs=this_inputs,
                                outputs=this_outputs)

        self.assertIn(
            "doesn't support models with decoupled transaction policy",
            str(cm.exception))

    def test_no_streaming(self):
        # Test cases with no streaming inference. Server should give
        # appropriate error in such cases.

        self._no_streaming_helper("grpc")
        self._no_streaming_helper("http")

    def test_wrong_shape(self):
        # Sends mismatching shapes for IN and DELAY. Server should return
        # appropriate error message. The shape of IN is [repeat_count],
        # where as shape of DELAY is [repeat_count + 1].

        data_offset = 100
        repeat_count = 1
        delay_time = 1000
        wait_time = 2000

        input_data = np.arange(start=data_offset,
                               stop=data_offset + repeat_count,
                               dtype=np.int32)
        delay_data = (np.ones([repeat_count + 1], dtype=np.uint32)) * delay_time
        wait_data = np.array([wait_time], dtype=np.uint32)

        # Initialize data for IN
        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        self.inputs_[1].set_shape([repeat_count + 1])
        self.inputs_[1].set_data_from_numpy(delay_data)

        # Initialize data for WAIT
        self.inputs_[2].set_data_from_numpy(wait_data)

        user_data = UserData()
        result_dict = {}

        with self.assertRaises(InferenceServerException) as cm:
            self._stream_infer(1, 0, repeat_count, delay_data, 1, user_data,
                               result_dict)

        self.assertIn("expected IN and DELAY shape to match, got [1] and [2]",
                      str(cm.exception))


if __name__ == '__main__':
    unittest.main()
