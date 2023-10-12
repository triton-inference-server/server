#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
import time
import unittest
from functools import partial

import numpy as np
import shm_util
import test_util as tu
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


class DecoupledTest(tu.TestResultCollector):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def test_decoupled_execute_error(self):
        # The decoupled_execute_error model returns an error for the first
        # request and successfully processes the second request. This is making
        # sure that an error in a single request does not completely fail the
        # batch.

        model_name = "decoupled_execute_error"
        shape = [2, 2]
        number_of_requests = 2
        user_data = UserData()
        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
                triton_client.start_stream(callback=partial(callback, user_data))

                input_datas = []
                for i in range(number_of_requests):
                    input_data = np.random.randn(*shape).astype(np.float32)
                    input_datas.append(input_data)
                    inputs = [
                        grpcclient.InferInput(
                            "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    triton_client.async_stream_infer(
                        model_name=model_name, inputs=inputs
                    )

                for i in range(number_of_requests):
                    result = user_data._completed_requests.get()
                    if i == 0:
                        self.assertIs(type(result), InferenceServerException)
                        continue

                    print(result)
                    output_data = result.as_numpy("OUT")
                    self.assertIsNotNone(output_data, "error: expected 'OUT'")
                    self.assertTrue(
                        np.array_equal(output_data, input_datas[i]),
                        "error: expected output {} to match input {}".format(
                            output_data, input_datas[i]
                        ),
                    )

    def test_decoupled_bls(self):
        # Test combinations of BLS and decoupled API in Python backend.
        model_name = "decoupled_bls"
        shape = [1, 2]
        user_data = UserData()
        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
                triton_client.start_stream(callback=partial(callback, user_data))

                input_datas = []
                input_data = np.random.randn(*shape).astype(np.float32)
                input_datas.append(input_data)
                inputs = [
                    grpcclient.InferInput(
                        "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                triton_client.async_stream_infer(model_name=model_name, inputs=inputs)

                # Check the results of the decoupled model using BLS
                def check_result(result):
                    # Make sure the result is not an exception
                    self.assertIsNot(type(result), InferenceServerException)

                    output_data = result.as_numpy("OUT")
                    self.assertIsNotNone(output_data, "error: expected 'OUT'")
                    self.assertTrue(
                        np.array_equal(output_data, input_data),
                        "error: expected output {} to match input {}".format(
                            output_data, input_data
                        ),
                    )

                result = user_data._completed_requests.get()
                check_result(result)

    def test_decoupled_bls_stream(self):
        # Test combinations of BLS and decoupled API in Python backend.
        model_name = "decoupled_bls_stream"
        in_values = [4, 2, 0, 1]
        user_data = UserData()
        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
                triton_client.start_stream(callback=partial(callback, user_data))
                for i in range(len(in_values)):
                    input_data = np.array([in_values[i]], dtype=np.int32)
                    inputs = [
                        grpcclient.InferInput(
                            "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    triton_client.async_stream_infer(
                        model_name=model_name, inputs=inputs, request_id=str(i)
                    )

                # Retrieve results...
                recv_count = 0
                expected_count = sum(in_values)
                result_dict = {}
                while recv_count < expected_count:
                    data_item = user_data._completed_requests.get()
                    self.assertIsNot(type(data_item), InferenceServerException)

                    this_id = data_item.get_response().id
                    if this_id not in result_dict.keys():
                        result_dict[this_id] = []
                    result_dict[this_id].append((recv_count, data_item))

                    recv_count += 1
                # Validate results...
                for i in range(len(in_values)):
                    this_id = str(i)
                    is_received = False
                    if this_id in result_dict.keys():
                        is_received = True

                    if in_values[i] != 0:
                        self.assertTrue(
                            is_received,
                            "response for request id {} not received".format(this_id),
                        )
                        self.assertEqual(len(result_dict[this_id]), in_values[i])

                        result_list = result_dict[this_id]
                        expected_data = np.array([in_values[i]], dtype=np.int32)
                        for j in range(len(result_list)):
                            this_data = result_list[j][1].as_numpy("OUT")
                            self.assertTrue(
                                np.array_equal(expected_data, this_data),
                                "error: incorrect data: expected {}, got {}".format(
                                    expected_data, this_data
                                ),
                            )
                    else:
                        self.assertFalse(
                            is_received,
                            "received unexpected response for request id {}".format(
                                this_id
                            ),
                        )

    def test_decoupled_return_response_error(self):
        model_name = "decoupled_return_response_error"
        shape = [16]
        user_data = UserData()
        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                client.start_stream(callback=partial(callback, user_data))
                input_data_0 = np.random.random(shape).astype(np.float32)
                input_data_1 = np.random.random(shape).astype(np.float32)
                inputs = [
                    grpcclient.InferInput(
                        "INPUT0",
                        input_data_0.shape,
                        np_to_triton_dtype(input_data_0.dtype),
                    ),
                    grpcclient.InferInput(
                        "INPUT1",
                        input_data_1.shape,
                        np_to_triton_dtype(input_data_1.dtype),
                    ),
                ]
                inputs[0].set_data_from_numpy(input_data_0)
                inputs[1].set_data_from_numpy(input_data_1)
                client.async_stream_infer(model_name=model_name, inputs=inputs)
                data_item = user_data._completed_requests.get()
                if type(data_item) == InferenceServerException:
                    self.assertEqual(
                        data_item.message(),
                        "Python model 'decoupled_return_response_error_0_0' is using "
                        "the decoupled mode and the execute function must return "
                        "None.",
                        "Exception message didn't match.",
                    )

    def test_decoupled_send_after_close_error(self):
        model_name = "decoupled_send_after_close_error"
        shape = [16]
        user_data = UserData()
        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                client.start_stream(callback=partial(callback, user_data))
                input_data_0 = np.random.random(shape).astype(np.float32)
                input_data_1 = np.random.random(shape).astype(np.float32)
                inputs = [
                    grpcclient.InferInput(
                        "INPUT0",
                        input_data_0.shape,
                        np_to_triton_dtype(input_data_0.dtype),
                    ),
                    grpcclient.InferInput(
                        "INPUT1",
                        input_data_1.shape,
                        np_to_triton_dtype(input_data_1.dtype),
                    ),
                ]
                inputs[0].set_data_from_numpy(input_data_0)
                inputs[1].set_data_from_numpy(input_data_1)
                client.async_stream_infer(model_name=model_name, inputs=inputs)

                # Because the model has closed the response sender there is no
                # way to deliver the error message to the client. The error
                # will be logged on the server side.
                time.sleep(4)
                self.assertEqual(
                    user_data._completed_requests.qsize(),
                    0,
                    "The completed request size must be zero.",
                )

    def test_decoupled_execute_cancel(self):
        model_name = "execute_cancel"
        log_path = "decoupled_server.log"
        execute_delay = 4.0  # seconds
        shape = [1, 1]
        user_data = UserData()

        with self._shm_leak_detector.Probe(debug_str=model_name) as shm_probe:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                client.start_stream(callback=partial(callback, user_data))
                input_data = np.array([[execute_delay]], dtype=np.float32)
                inputs = [
                    grpcclient.InferInput(
                        "EXECUTE_DELAY", shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                client.async_stream_infer(model_name, inputs)
                time.sleep(2)  # model delay for decoupling request and response sender
                time.sleep(2)  # ensure the request is executing
                client.stop_stream(cancel_requests=True)
                time.sleep(2)  # ensure the cancellation is delivered

            self.assertFalse(user_data._completed_requests.empty())
            while not user_data._completed_requests.empty():
                data_item = user_data._completed_requests.get()
                self.assertIsInstance(data_item, InferenceServerException)
                self.assertEqual(data_item.status(), "StatusCode.CANCELLED")

            with open(log_path, mode="r", encoding="utf-8", errors="strict") as f:
                log_text = f.read()
            self.assertIn("[execute_cancel] Request not cancelled at 1.0 s", log_text)
            self.assertIn("[execute_cancel] Request cancelled at ", log_text)

    def test_decoupled_raise_exception(self):
        # The decoupled_raise_exception model raises an exception for the request.
        # This test case is making sure that repeated exceptions are properly handled.

        model_name = "decoupled_raise_exception"
        shape = [2, 2]
        number_of_requests = 10
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            input_datas = []
            for i in range(number_of_requests):
                input_data = np.random.randn(*shape).astype(np.float32)
                input_datas.append(input_data)
                inputs = [
                    grpcclient.InferInput(
                        "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                triton_client.async_stream_infer(model_name=model_name, inputs=inputs)

            for i in range(number_of_requests):
                result = user_data._completed_requests.get()
                self.assertIs(type(result), InferenceServerException)
                self.assertIn("Intentional Error", result.message())

            self.assertTrue(triton_client.is_model_ready(model_name))


if __name__ == "__main__":
    unittest.main()
