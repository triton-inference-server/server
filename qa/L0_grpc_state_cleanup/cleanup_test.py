#!/usr/bin/env python3

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

sys.path.append("../common")

import os
import queue
import signal
import time
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


class CleanUpTest(tu.TestResultCollector):
    SERVER_PID = None

    def setUp(self):
        self.decoupled_model_name_ = "repeat_int32"
        self.sequence_model_name_ = "simple_sequence"
        self.identity_model_name_ = "simple_identity"

        self.inputs_ = []
        self.inputs_.append(grpcclient.InferInput("IN", [1], "INT32"))
        self.inputs_.append(grpcclient.InferInput("DELAY", [1], "UINT32"))
        self.inputs_.append(grpcclient.InferInput("WAIT", [1], "UINT32"))

        self.outputs_ = []
        self.outputs_.append(grpcclient.InferRequestedOutput("OUT"))
        self.outputs_.append(grpcclient.InferRequestedOutput("IDX"))
        self.requested_outputs_ = self.outputs_

    def _stream_infer_with_params(
        self,
        request_count,
        request_delay,
        _,
        delay_data,
        delay_factor,
        user_data,
        result_dict,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:
            # Establish stream
            triton_client.start_stream(
                callback=partial(callback, user_data), stream_timeout=stream_timeout
            )
            # Send specified many requests in parallel
            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                if kill_server:
                    if i == kill_server:
                        os.kill(int(self.SERVER_PID), signal.SIGINT)
                triton_client.async_stream_infer(
                    model_name=self.decoupled_model_name_,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_,
                    # Opt-in to receiving flags-only responses from model/backend
                    # to help detect final responses for decoupled models.
                    enable_empty_final_response=True,
                )
                # Update delay input in accordance with the scaling factor
                delay_data = delay_data * delay_factor
                delay_data = delay_data.astype(np.uint32)

            # Retrieve results...
            recv_count = 0
            completed_requests = 0
            while completed_requests < request_count:
                if cancel_response_idx:
                    if cancel_response_idx == recv_count:
                        triton_client.stop_stream(cancel_requests=True)
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    response = data_item.get_response()
                    # Request IDs should generally be provided with each request
                    # to associate decoupled responses with their requests.
                    if not response.id:
                        raise ValueError(
                            "No response id found. Was a request_id provided?"
                        )

                    # Detect final response. Parameters are oneof and we expect bool_param
                    if response.parameters.get("triton_final_response").bool_param:
                        completed_requests += 1

                    # Only process non-empty response, ignore if empty (no outputs)
                    if response.outputs:
                        if response.id not in result_dict:
                            result_dict[response.id] = []
                        result_dict[response.id].append((recv_count, data_item))
                        recv_count += 1

    def _stream_infer(
        self,
        request_count,
        request_delay,
        expected_count,
        delay_data,
        delay_factor,
        user_data,
        result_dict,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:
            # Establish stream
            triton_client.start_stream(
                callback=partial(callback, user_data), stream_timeout=stream_timeout
            )
            # Send specified many requests in parallel
            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                if kill_server:
                    if i == kill_server:
                        os.kill(int(self.SERVER_PID), signal.SIGINT)
                triton_client.async_stream_infer(
                    model_name=self.decoupled_model_name_,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_,
                )
                # Update delay input in accordance with the scaling factor
                delay_data = delay_data * delay_factor
                delay_data = delay_data.astype(np.uint32)

            # Retrieve results...
            recv_count = 0
            while recv_count < expected_count:
                if cancel_response_idx:
                    if cancel_response_idx == recv_count:
                        triton_client.stop_stream(cancel_requests=True)
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    if this_id not in result_dict:
                        result_dict[this_id] = []
                    result_dict[this_id].append((recv_count, data_item))

                recv_count += 1

    def _decoupled_infer(
        self,
        request_count,
        request_delay=0,
        repeat_count=1,
        data_offset=100,
        delay_time=1000,
        delay_factor=1,
        wait_time=500,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
        should_error=True,
        infer_helper_map=[True, True],
    ):
        # Initialize data for IN
        input_data = np.arange(
            start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
        )
        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        # Initialize data for DELAY
        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        self.inputs_[1].set_shape([repeat_count])

        # Initialize data for WAIT
        wait_data = np.array([wait_time], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        # use validate_fn to differentiate requested outputs
        self.requested_outputs_ = self.outputs_

        infer_helpers = []
        if infer_helper_map[0]:
            infer_helpers.append(self._stream_infer)
        if infer_helper_map[1]:
            infer_helpers.append(self._stream_infer_with_params)

        for infer_helper in infer_helpers:
            user_data = UserData()
            result_dict = {}

            try:
                expected_count = repeat_count * request_count
                infer_helper(
                    request_count,
                    request_delay,
                    expected_count,
                    delay_data,
                    delay_factor,
                    user_data,
                    result_dict,
                    cancel_response_idx,
                    stream_timeout,
                    kill_server,
                )
            except Exception as ex:
                if cancel_response_idx or stream_timeout or should_error:
                    raise ex
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Validate the results..
            for i in range(request_count):
                this_id = str(i)
                if repeat_count != 0 and this_id not in result_dict.keys():
                    self.assertTrue(
                        False, "response for request id {} not received".format(this_id)
                    )
                elif repeat_count == 0 and this_id in result_dict.keys():
                    self.assertTrue(
                        False,
                        "received unexpected response for request id {}".format(
                            this_id
                        ),
                    )
                if repeat_count != 0:
                    self.assertEqual(len(result_dict[this_id]), repeat_count)
                    expected_data = data_offset
                    result_list = result_dict[this_id]
                    for j in range(len(result_list)):
                        this_data = result_list[j][1].as_numpy("OUT")
                        self.assertEqual(len(this_data), 1)
                        self.assertEqual(this_data[0], expected_data)
                        this_idx = result_list[j][1].as_numpy("IDX")
                        self.assertEqual(len(this_idx), 1)
                        self.assertEqual(this_idx[0], j)
                        expected_data += 1

    #    def test_infer(self):
    #
    #    def test_infer_cancellation(self):
    #
    #    def test_infer_timeout(self):
    #
    #    def test_infer_error_status(self):
    #
    #    def test_infer_shutdownserver(self):
    #
    #
    #    def test_streaming_infer(self):
    #
    #    def test_streaming_cancellation(self):
    #
    #    def test_streaming_timeout(self):
    #
    #    def test_streaming_error_status(self):
    #
    #    def test_streaming_infer_shutdownserver(self):

    ###
    ### Decoupled Steaming Tests
    ###
    def test_decoupled_infer(self):
        # Sanity test to check whether all the state objects
        # are correclty released. Sends 10 requests in a single
        # gRPC bidirectional stream and expects each of these
        # requests to generate 10 responses.
        self._decoupled_infer(request_count=10, repeat_count=10)

    def test_decoupled_cancellation(self):
        # This test case is used to check whether all the states are
        # correctly released when the stream is closed when fifth
        # response is received.
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10, repeat_count=10, cancel_response_idx=5
            )
        self.assertIn("Locally cancelled by application!", str(cm.exception))

    def test_decoupled_timeout(self):
        # This test case is used to check whether all the states are
        # released when some of the requests timeouts.
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10, repeat_count=10, request_delay=1, stream_timeout=2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_decoupled_error_status(self):
        # This test case is used to check whether all the state objects are
        # released when RPC runs into error.
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(request_count=10, repeat_count=10, should_error=True)
        self.assertIn(
            "This protocol is restricted, expecting header 'triton-grpc-protocol-infer-key'",
            str(cm.exception),
        )

    def test_decoupled_infer_shutdownserver(self):
        # This test case is used to check whether all the state objects are
        # released when the server is interrupted to shutdown in middle of
        # inference run.
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10,
                repeat_count=10,
                request_delay=1,
                kill_server=5,
                should_error=True,
                infer_helper_map=[True, False],
            )
        self.assertIn("Request for unknown model", str(cm.exception))

    def test_decoupled_infer_with_params_shutdownserver(self):
        # This test case is used to check whether all the state objects are
        # released when the server is interrupted to shutdown in middle of
        # inference run with final parameters being returned.
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10,
                repeat_count=10,
                request_delay=1,
                kill_server=5,
                should_error=True,
                infer_helper_map=[False, True],
            )
        self.assertIn("Request for unknown model", str(cm.exception))


if __name__ == "__main__":
    CleanUpTest.SERVER_PID = os.environ.get("SERVER_PID", CleanUpTest.SERVER_PID)
    unittest.main()
