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

import concurrent.futures
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TestScheduler(unittest.TestCase):
    def setUp(self):
        # Initialize client
        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def _get_inputs(self, batch_size):
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        shape = [batch_size, 8]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _generate_callback_and_response_pair(self):
        response = {"responded": False, "result": None, "error": None}

        def callback(result, error):
            response["responded"] = True
            response["result"] = result
            response["error"] = error

        return callback, response

    def _assert_response_is_cancelled(self, response):
        self.assertTrue(response["responded"])
        self.assertEqual(response["result"], None)
        self.assertIsInstance(response["error"], InferenceServerException)
        self.assertEqual(response["error"].status(), "StatusCode.CANCELLED")

    # Test queued requests on dynamic batch scheduler can be cancelled
    def test_dynamic_batch_scheduler_request_cancellation(self):
        model_name = "dynamic_batch"
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Saturate the 2 batch slots on the model of 1 instance
            saturate_thread_1 = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            saturate_thread_2 = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            time.sleep(2)  # ensure the slots are filled
            # The next request should be queued
            callback, response = self._generate_callback_and_response_pair()
            queue_future = self._triton.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            time.sleep(2)  # ensure the request is queued
            self.assertFalse(response["responded"])
            # Cancel the queued request
            queue_future.cancel()
            time.sleep(2)  # ensure the cancellation is delivered
            self._assert_response_is_cancelled(response)
            # Join saturating thread
            saturate_thread_1.result()
            saturate_thread_2.result()

    # Test backlogged requests on sequence batch scheduler can be cancelled
    def test_sequence_batch_scheduler_backlog_request_cancellation(self):
        model_name = "sequence_direct"
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Saturate the single sequence slot
            saturate_thread = pool.submit(
                self._triton.infer,
                model_name,
                self._get_inputs(batch_size=1),
                sequence_id=1,
                sequence_start=True,
            )
            time.sleep(2)  # ensure the slot is filled
            # The next sequence with 2 requests should be on the backlog
            backlog_requests = []
            for i in range(2):
                callback, response = self._generate_callback_and_response_pair()
                backlog_future = self._triton.async_infer(
                    model_name,
                    self._get_inputs(batch_size=1),
                    callback,
                    sequence_id=2,
                    sequence_start=(True if i == 0 else False),
                )
                backlog_requests.append(
                    {"future": backlog_future, "response": response}
                )
            time.sleep(2)  # ensure the sequence is backlogged
            self.assertFalse(backlog_requests[0]["response"]["responded"])
            self.assertFalse(backlog_requests[1]["response"]["responded"])
            # Cancelling any backlogged request cancels the entire sequence
            backlog_requests[0]["future"].cancel()
            time.sleep(2)  # ensure the cancellation is delivered
            self._assert_response_is_cancelled(backlog_requests[0]["response"])
            self._assert_response_is_cancelled(backlog_requests[1]["response"])
            # Join saturating thread
            saturate_thread.result()

    # Test queued requests on direct sequence batch scheduler can be cancelled
    def test_direct_sequence_batch_scheduler_request_cancellation(self):
        model_name = "sequence_direct"
        self._test_sequence_batch_scheduler_queued_request_cancellation(model_name)

    # Test queued requests on oldest sequence batch scheduler can be cancelled
    def test_oldest_sequence_batch_scheduler_request_cancellation(self):
        model_name = "sequence_oldest"
        self._test_sequence_batch_scheduler_queued_request_cancellation(model_name)

    # Helper function
    def _test_sequence_batch_scheduler_queued_request_cancellation(self, model_name):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Start the sequence
            start_thread = pool.submit(
                self._triton.infer,
                model_name,
                self._get_inputs(batch_size=1),
                sequence_id=1,
                sequence_start=True,
            )
            time.sleep(2)  # ensure the sequence has started
            # The next 2 requests should be queued
            queue_requests = []
            for i in range(2):
                callback, response = self._generate_callback_and_response_pair()
                queue_future = self._triton.async_infer(
                    model_name, self._get_inputs(batch_size=1), callback, sequence_id=1
                )
                queue_requests.append({"future": queue_future, "response": response})
            time.sleep(2)  # ensure the requests are queued
            self.assertFalse(queue_requests[0]["response"]["responded"])
            self.assertFalse(queue_requests[1]["response"]["responded"])
            # Cancelling any queued request cancels the entire sequence
            queue_requests[0]["future"].cancel()
            time.sleep(2)  # ensure the cancellation is delivered
            self._assert_response_is_cancelled(queue_requests[0]["response"])
            self._assert_response_is_cancelled(queue_requests[1]["response"])
            # Join start thread
            start_thread.result()

    # Test ensemble scheduler will propagate cancellation request to child
    def test_ensemble_scheduler_request_cancellation(self):
        model_name = "ensemble_model"
        callback, response = self._generate_callback_and_response_pair()
        infer_future = self._triton.async_infer(
            model_name, self._get_inputs(batch_size=1), callback
        )
        time.sleep(2)  # ensure the inference has started
        self.assertFalse(response["responded"])
        infer_future.cancel()
        time.sleep(2)  # ensure the cancellation is delivered
        self._assert_response_is_cancelled(response)


if __name__ == "__main__":
    unittest.main()
