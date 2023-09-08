#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import queue
import time
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as aiogrpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class ClientCancellationTest(tu.TestResultCollector):
    def setUp(self):
        self.model_name_ = "custom_identity_int32"
        self.input0_data_ = np.array([[10]], dtype=np.int32)
        self._start_time_ms = 0
        self._end_time_ms = 0

    def _record_start_time_ms(self):
        self._start_time_ms = int(round(time.time() * 1000))

    def _record_end_time_ms(self):
        self._end_time_ms = int(round(time.time() * 1000))

    def _test_runtime_duration(self, upper_limit):
        self.assertTrue(
            (self._end_time_ms - self._start_time_ms) < upper_limit,
            "test runtime expected less than "
            + str(upper_limit)
            + "ms response time, got "
            + str(self._end_time_ms - self._start_time_ms)
            + " ms",
        )

    def _prepare_request(self):
        self.inputs_ = []
        self.inputs_.append(grpcclient.InferInput("INPUT0", [1, 1], "INT32"))
        self.outputs_ = []
        self.outputs_.append(grpcclient.InferRequestedOutput("OUTPUT0"))

        self.inputs_[0].set_data_from_numpy(self.input0_data_)

    def test_grpc_async_infer(self):
        # Sends a request using async_infer to a
        # model that takes 10s to execute. Issues
        # a cancellation request after 2s. The client
        # should return with appropriate exception within
        # 5s.
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        self._prepare_request()

        user_data = UserData()

        self._record_start_time_ms()

        # Expect inference to pass successfully for a large timeout
        # value
        future = triton_client.async_infer(
            model_name=self.model_name_,
            inputs=self.inputs_,
            callback=partial(callback, user_data),
            outputs=self.outputs_,
        )
        time.sleep(2)
        future.cancel()

        # Wait until the results is captured via callback
        data_item = user_data._completed_requests.get()
        self.assertEqual(type(data_item), grpcclient.CancelledError)

        self._record_end_time_ms()
        self._test_runtime_duration(5000)

    def test_grpc_stream_infer(self):
        # Sends a request using async_stream_infer to a
        # model that takes 10s to execute. Issues stream
        # closure with cancel_requests=True. The client
        # should return with appropriate exception within
        # 5s.
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )

        self._prepare_request()
        user_data = UserData()

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        triton_client.start_stream(callback=partial(callback, user_data))
        self._record_start_time_ms()
        for i in range(1):
            triton_client.async_stream_infer(
                model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
            )

        time.sleep(2)
        triton_client.stop_stream(cancel_requests=True)

        data_item = user_data._completed_requests.get()
        self.assertEqual(type(data_item), grpcclient.CancelledError)

        self._record_end_time_ms()
        self._test_runtime_duration(5000)

    def test_aio_grpc_async_infer(self):
        # Sends a request using infer of grpc.aio to a
        # model that takes 10s to execute. Issues
        # a cancellation request after 2s. The client
        # should return with appropriate exception within
        # 5s.
        async def cancel_request(call):
            await asyncio.sleep(2)
            self.assertTrue(call.cancel())

        async def handle_response(call):
            with self.assertRaises(asyncio.exceptions.CancelledError) as cm:
                await call

        async def test_aio_infer(self):
            triton_client = aiogrpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True
            )
            self._prepare_request()
            self._record_start_time_ms()
            # Expect inference to pass successfully for a large timeout
            # value
            call = await triton_client.infer(
                model_name=self.model_name_,
                inputs=self.inputs_,
                outputs=self.outputs_,
                get_call_obj=True,
            )
            asyncio.create_task(handle_response(call))
            asyncio.create_task(cancel_request(call))

            self._record_end_time_ms()
            self._test_runtime_duration(5000)

        asyncio.run(test_aio_infer(self))

    def test_aio_grpc_stream_infer(self):
        # Sends a request using stream_infer of grpc.aio
        # library model that takes 10s to execute. Issues
        # stream closure with cancel_requests=True. The client
        # should return with appropriate exception within
        # 5s.
        async def test_aio_streaming_infer(self):
            async with aiogrpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True
            ) as triton_client:

                async def async_request_iterator():
                    for i in range(1):
                        await asyncio.sleep(1)
                        yield {
                            "model_name": self.model_name_,
                            "inputs": self.inputs_,
                            "outputs": self.outputs_,
                        }

                self._prepare_request()
                self._record_start_time_ms()
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(), get_call_obj=True
                )
                streaming_call = await response_iterator.__anext__()

                async def cancel_streaming(streaming_call):
                    await asyncio.sleep(2)
                    streaming_call.cancel()

                async def handle_response(response_iterator):
                    with self.assertRaises(asyncio.exceptions.CancelledError) as cm:
                        async for response in response_iterator:
                            self.assertTrue(False, "Received an unexpected response!")

                asyncio.create_task(handle_response(response_iterator))
                asyncio.create_task(cancel_streaming(streaming_call))

                self._record_end_time_ms()
                self._test_runtime_duration(5000)

        asyncio.run(test_aio_streaming_infer(self))


if __name__ == "__main__":
    unittest.main()
