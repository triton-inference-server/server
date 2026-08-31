#!/usr/bin/env python3

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
import time
import unittest

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# 'plan_no_batching' has no batcher, so its requests go straight to the rate
# limiter. Both models need the single global "SHARED" resource, so while
# 'resource_holder' runs the TensorRT request stays queued there long enough to
# be cancelled deterministically.
TRT_MODEL = "plan_no_batching"
HOLDER_MODEL = "resource_holder"

# Plan model from qa_model_repository: OUTPUT0 = INPUT0 + INPUT1.
TRT_SHAPE = [1, 16]
HOLDER_SHAPE = [1, 8]


class TestTrtRequestCancellation(unittest.TestCase):
    def setUp(self):
        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def _trt_inputs(self, value):
        inputs = [
            grpcclient.InferInput("INPUT0", TRT_SHAPE, "FP32"),
            grpcclient.InferInput("INPUT1", TRT_SHAPE, "FP32"),
        ]
        for model_input in inputs:
            model_input.set_data_from_numpy(
                np.full(TRT_SHAPE, value, dtype=np.float32)
            )
        return inputs

    def _holder_inputs(self):
        inputs = [grpcclient.InferInput("INPUT0", HOLDER_SHAPE, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(HOLDER_SHAPE, dtype=np.float32))
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

    def _execution_count(self, model_name):
        stats = self._triton.get_inference_statistics(
            model_name=model_name, as_json=True
        )
        model_stats = stats.get("model_stats", [])
        if not model_stats:
            return 0
        return int(model_stats[0].get("execution_count", 0))

    def _get_metrics(self):
        r = requests.get("http://localhost:8002/metrics")
        r.raise_for_status()
        return r.text

    def _failure_count(self, model, reason):
        pattern = (
            rf'nv_inference_request_failure\{{model="{model}",'
            rf'reason="{reason}",version="1"\}} (\d+)'
        )
        match = re.search(pattern, self._get_metrics())
        return int(match.group(1)) if match else 0

    def _hold_resource(self, pool):
        """Occupy the shared resource for roughly six seconds."""
        holder = pool.submit(self._triton.infer, HOLDER_MODEL, self._holder_inputs())
        time.sleep(2)  # ensure the resource has been acquired
        return holder

    # A request cancelled while queued in the rate limiter must never reach the
    # TensorRT backend.
    def test_trt_queued_request_cancellation_skips_execution(self):
        executions_before = self._execution_count(TRT_MODEL)
        failures_before = self._failure_count(TRT_MODEL, "CANCELED")

        with concurrent.futures.ThreadPoolExecutor() as pool:
            holder = self._hold_resource(pool)

            callback, response = self._generate_callback_and_response_pair()
            queued = self._triton.async_infer(
                TRT_MODEL, self._trt_inputs(value=1.0), callback
            )
            time.sleep(2)  # ensure the request is queued on the rate limiter
            self.assertFalse(
                response["responded"],
                "the request was not held by the rate limiter",
            )

            queued.cancel()
            time.sleep(2)  # ensure the cancellation is delivered

            # Releasing the resource lets the queued payload be scheduled
            holder.result()
            time.sleep(3)  # ensure the cancelled request has been responded to

            self._assert_response_is_cancelled(response)

        self.assertEqual(
            self._execution_count(TRT_MODEL),
            executions_before,
            "the TensorRT model executed a request that had been cancelled",
        )
        self.assertEqual(
            self._failure_count(TRT_MODEL, "CANCELED"),
            failures_before + 1,
            "the cancelled request was not reported as CANCELED",
        )

    # A queued request that is not cancelled must still execute normally.
    def test_trt_queued_request_without_cancellation_still_executes(self):
        executions_before = self._execution_count(TRT_MODEL)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            holder = self._hold_resource(pool)

            callback, response = self._generate_callback_and_response_pair()
            # Hold the context so the request is not cancelled by going out of
            # scope
            queued = self._triton.async_infer(
                TRT_MODEL, self._trt_inputs(value=2.0), callback
            )
            self.assertIsNotNone(queued)
            time.sleep(2)  # ensure the request is queued on the rate limiter
            self.assertFalse(response["responded"])

            holder.result()
            time.sleep(3)  # ensure the request has been executed

            self.assertTrue(response["responded"])
            self.assertIsNone(response["error"])
            np.testing.assert_allclose(
                response["result"].as_numpy("OUTPUT0"),
                np.full(TRT_SHAPE, 4.0, dtype=np.float32),
            )

        self.assertEqual(self._execution_count(TRT_MODEL), executions_before + 1)

    # A plain request must be unaffected.
    def test_trt_request_is_unaffected(self):
        result = self._triton.infer(TRT_MODEL, self._trt_inputs(value=3.0))
        np.testing.assert_allclose(
            result.as_numpy("OUTPUT0"), np.full(TRT_SHAPE, 6.0, dtype=np.float32)
        )
        np.testing.assert_allclose(
            result.as_numpy("OUTPUT1"), np.zeros(TRT_SHAPE, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
