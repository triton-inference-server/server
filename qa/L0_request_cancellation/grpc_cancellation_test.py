#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import os
import queue
import re
import time
import unittest
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclientaio
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class GrpcCancellationTest(unittest.IsolatedAsyncioTestCase):
    _model_name = "custom_identity_int32"
    _model_delay = 10.0  # seconds
    _grpc_params = {"url": "localhost:8001", "verbose": True}

    def setUp(self):
        self._client = grpcclient.InferenceServerClient(**self._grpc_params)
        self._client_aio = grpcclientaio.InferenceServerClient(**self._grpc_params)
        self._user_data = UserData()
        self._callback = partial(callback, self._user_data)
        self._prepare_request()
        self._start_time = time.time()  # seconds
        self.test_duration_delta = 0.5

    def tearDown(self):
        self._end_time = time.time()  # seconds
        self._assert_max_duration()

    def _prepare_request(self):
        self._inputs = []
        self._inputs.append(grpcclient.InferInput("INPUT0", [1, 1], "INT32"))
        self._outputs = []
        self._outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
        self._inputs[0].set_data_from_numpy(np.array([[10]], dtype=np.int32))

    def _assert_max_duration(self):
        max_duration = self._model_delay * self.test_duration_delta  # seconds
        duration = self._end_time - self._start_time  # seconds
        self.assertLess(
            duration,
            max_duration,
            f"test runtime expected less than {max_duration}s response time, got {duration}s",
        )

    def _assert_callback_cancelled(self):
        self.assertFalse(self._user_data._completed_requests.empty())
        data_item = self._user_data._completed_requests.get()
        self.assertIsInstance(data_item, InferenceServerException)
        self.assertIn("Locally cancelled by application!", str(data_item))

    def test_grpc_async_infer(self):
        future = self._client.async_infer(
            model_name=self._model_name,
            inputs=self._inputs,
            callback=self._callback,
            outputs=self._outputs,
        )
        time.sleep(2)  # ensure the inference has started
        future.cancel()
        time.sleep(0.1)  # context switch
        self._assert_callback_cancelled()

    def test_grpc_stream_infer(self):
        self._client.start_stream(callback=self._callback)
        self._client.async_stream_infer(
            model_name=self._model_name, inputs=self._inputs, outputs=self._outputs
        )
        time.sleep(2)  # ensure the inference has started
        self._client.stop_stream(cancel_requests=True)
        self._assert_callback_cancelled()

    async def test_aio_grpc_async_infer(self):
        infer_task = asyncio.create_task(
            self._client_aio.infer(
                model_name=self._model_name, inputs=self._inputs, outputs=self._outputs
            )
        )
        await asyncio.sleep(2)  # ensure the inference has started
        infer_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await infer_task

    async def test_aio_grpc_stream_infer(self):
        async def requests_generator():
            yield {
                "model_name": self._model_name,
                "inputs": self._inputs,
                "outputs": self._outputs,
            }

        responses_iterator = self._client_aio.stream_infer(requests_generator())
        await asyncio.sleep(2)  # ensure the inference has started
        self.assertTrue(responses_iterator.cancel())
        with self.assertRaises(asyncio.CancelledError):
            async for result, error in responses_iterator:
                self._callback(result, error)

    def test_grpc_async_infer_cancellation_at_step_start(self):
        # This is a longer test
        self.test_duration_delta = 4.5
        server_log_name = "grpc_cancellation_test.test_grpc_async_infer_cancellation_at_step_start.server.log"
        with open(server_log_name, "r") as f:
            server_log = f.read()

        prev_new_req_handl_count = len(
            re.findall("New request handler for ModelInferHandler", server_log)
        )
        self.assertEqual(
            prev_new_req_handl_count,
            2,
            "Expected 2 request handler for ModelInferHandler log entries, but got {}".format(
                prev_new_req_handl_count
            ),
        )
        future = self._client.async_infer(
            model_name=self._model_name,
            inputs=self._inputs,
            callback=self._callback,
            outputs=self._outputs,
        )
        time.sleep(2)  # ensure the inference request reached server
        future.cancel()
        # ensures TRITONSERVER_DELAY_GRPC_PROCESS delay passed on the server
        time.sleep(self._model_delay * 2)

        with open(server_log_name, "r") as f:
            server_log = f.read()

        cur_new_req_handl_count = len(
            re.findall("New request handler for ModelInferHandler", server_log)
        )
        self.assertGreater(
            cur_new_req_handl_count,
            prev_new_req_handl_count,
            "gRPC Cancellation on step START Test Failed: New request handler for ModelInferHandler was not created",
        )

    def test_grpc_async_infer_response_complete_during_cancellation(self):
        # long test
        self.test_duration_delta = 2
        delay_notification_sec = (
            int(os.getenv("TRITONSERVER_DELAY_GRPC_NOTIFICATION")) / 1000
        )
        delay_queue_cancellation_sec = (
            int(os.getenv("TRITONSERVER_DELAY_GRPC_ENQUEUE")) / 1000
        )
        future = self._client.async_infer(
            model_name=self._model_name,
            inputs=self._inputs,
            callback=self._callback,
            outputs=self._outputs,
        )
        # ensure cancellation is received before InferResponseComplete and is processed after InferResponseComplete
        time.sleep(self._model_delay - 2)
        future.cancel()
        time.sleep(
            delay_notification_sec + delay_queue_cancellation_sec
        )  # ensure the cancellation is processed
        self._assert_callback_cancelled()

    def test_grpc_async_infer_cancellation_during_response_complete(self):
        # long test
        self.test_duration_delta = 2.5
        delay_notification_sec = (
            int(os.getenv("TRITONSERVER_DELAY_GRPC_NOTIFICATION")) / 1000
        )
        delay_response_completion_sec = (
            int(os.getenv("TRITONSERVER_DELAY_RESPONSE_COMPLETION")) / 1000
        )
        future = self._client.async_infer(
            model_name=self._model_name,
            inputs=self._inputs,
            callback=self._callback,
            outputs=self._outputs,
        )
        # ensure the cancellation is received between InferResponseComplete checking cancellation and Finish
        time.sleep(self._model_delay + 2)
        future.cancel()
        time.sleep(
            delay_notification_sec + delay_response_completion_sec
        )  # ensure the cancellation is processed
        self._assert_callback_cancelled()


if __name__ == "__main__":
    unittest.main()
