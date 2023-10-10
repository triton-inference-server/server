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

import asyncio
import queue
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


class GrpcCancellationTest(unittest.TestCase):
    _model_name = "custom_identity_int32"
    _model_delay = 10.0  # seconds
    _grpc_params = {"url": "localhost:8001", "verbose": True}

    def setUp(self):
        self._client = grpcclient.InferenceServerClient(**self._grpc_params)
        self._client_aio = grpcclientaio.InferenceServerClient(**self._grpc_params)
        self._user_data = UserData()
        self._callback = partial(callback, self._user_data)
        self._prepare_request()
        self._record_start_time()

    def tearDown(self):
        self._record_end_time()
        self._assert_max_duration()
        self._assert_cancelled_by_client()

    def _record_start_time(self):
        self._start_time = time.time()  # seconds

    def _record_end_time(self):
        self._end_time = time.time()  # seconds

    def _prepare_request(self):
        self._inputs = []
        self._inputs.append(grpcclient.InferInput("INPUT0", [1, 1], "INT32"))
        self._outputs = []
        self._outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
        self._inputs[0].set_data_from_numpy(np.array([[10]], dtype=np.int32))

    def _assert_max_duration(self):
        max_duration = self._model_delay * 0.5  # seconds
        duration = self._end_time - self._start_time  # seconds
        self.assertLess(
            duration,
            max_duration,
            "test runtime expected less than "
            + str(max_duration)
            + "s response time, got "
            + str(duration)
            + "s",
        )

    def _assert_cancelled_by_client(self):
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

    def test_grpc_stream_infer(self):
        self._client.start_stream(callback=self._callback)
        self._client.async_stream_infer(
            model_name=self._model_name, inputs=self._inputs, outputs=self._outputs
        )
        time.sleep(2)  # ensure the inference has started
        self._client.stop_stream(cancel_requests=True)


# Disabling AsyncIO cancellation testing. Enable once
# DLIS-5476 is implemented.
#    def test_aio_grpc_async_infer(self):
#        # Sends a request using infer of grpc.aio to a
#        # model that takes 10s to execute. Issues
#        # a cancellation request after 2s. The client
#        # should return with appropriate exception within
#        # 5s.
#        async def cancel_request(call):
#            await asyncio.sleep(2)
#            self.assertTrue(call.cancel())
#
#        async def handle_response(generator):
#            with self.assertRaises(asyncio.exceptions.CancelledError) as cm:
#                _ = await anext(generator)
#
#        async def test_aio_infer(self):
#            triton_client = grpcclientaio.InferenceServerClient(
#                url=self._triton_grpc_url, verbose=True
#            )
#            self._prepare_request()
#            self._record_start_time_ms()
#
#            generator = triton_client.infer(
#                model_name=self.model_name_,
#                inputs=self.inputs_,
#                outputs=self.outputs_,
#                get_call_obj=True,
#            )
#            grpc_call = await anext(generator)
#
#            tasks = []
#            tasks.append(asyncio.create_task(handle_response(generator)))
#            tasks.append(asyncio.create_task(cancel_request(grpc_call)))
#
#            for task in tasks:
#                await task
#
#            self._record_end_time_ms()
#            self._assert_runtime_duration(5000)
#
#        asyncio.run(test_aio_infer(self))
#
#    def test_aio_grpc_stream_infer(self):
#        # Sends a request using stream_infer of grpc.aio
#        # library model that takes 10s to execute. Issues
#        # stream closure with cancel_requests=True. The client
#        # should return with appropriate exception within
#        # 5s.
#        async def test_aio_streaming_infer(self):
#            async with grpcclientaio.InferenceServerClient(
#                url=self._triton_grpc_url, verbose=True
#            ) as triton_client:
#
#                async def async_request_iterator():
#                    for i in range(1):
#                        await asyncio.sleep(1)
#                        yield {
#                            "model_name": self.model_name_,
#                            "inputs": self.inputs_,
#                            "outputs": self.outputs_,
#                        }
#
#                self._prepare_request()
#                self._record_start_time_ms()
#                response_iterator = triton_client.stream_infer(
#                    inputs_iterator=async_request_iterator(), get_call_obj=True
#                )
#                streaming_call = await anext(response_iterator)
#
#                async def cancel_streaming(streaming_call):
#                    await asyncio.sleep(2)
#                    streaming_call.cancel()
#
#                async def handle_response(response_iterator):
#                    with self.assertRaises(asyncio.exceptions.CancelledError) as cm:
#                        async for response in response_iterator:
#                            self.assertTrue(False, "Received an unexpected response!")
#
#                tasks = []
#                tasks.append(asyncio.create_task(handle_response(response_iterator)))
#                tasks.append(asyncio.create_task(cancel_streaming(streaming_call)))
#
#                for task in tasks:
#                    await task
#
#                self._record_end_time_ms()
#                self._assert_runtime_duration(5000)
#
#        asyncio.run(test_aio_streaming_infer(self))


if __name__ == "__main__":
    unittest.main()
