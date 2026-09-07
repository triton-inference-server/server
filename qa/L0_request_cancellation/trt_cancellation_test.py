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
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from cancellation_test_utils import CancellationTest

# A TensorRT request cancelled while waiting in the rate limiter must not run
# inference. Both models need the one global "SHARED" resource, so
# 'resource_holder' keeps the TensorRT requests queued.
TRT_MODEL = "plan_no_batching"
HOLDER_MODEL = "resource_holder"

# Plan model from qa_model_repository: OUTPUT0 = INPUT0 + INPUT1.
TRT_SHAPE = [1, 16]
HOLDER_SHAPE = [1, 8]


class TestTrtRequestCancellation(CancellationTest, unittest.TestCase):
    def setUp(self):
        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    # Build the two TensorRT inputs with a predictable summed output.
    def _trt_inputs(self, value):
        inputs = [
            grpcclient.InferInput("INPUT0", TRT_SHAPE, "FP32"),
            grpcclient.InferInput("INPUT1", TRT_SHAPE, "FP32"),
        ]
        for model_input in inputs:
            model_input.set_data_from_numpy(np.full(TRT_SHAPE, value, dtype=np.float32))
        return inputs

    # Build identity-model input used only to occupy the shared resource.
    def _holder_inputs(self):
        inputs = [grpcclient.InferInput("INPUT0", HOLDER_SHAPE, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(HOLDER_SHAPE, dtype=np.float32))
        return inputs

    # A cancelled TRT request must not run, while an adjacent live request does.
    def test_trt_rate_limited_cancellation_skips_only_cancelled_request(self):
        executions_before = self._execution_count(TRT_MODEL)
        failures_before = self._failure_count(TRT_MODEL, "CANCELED")

        with concurrent.futures.ThreadPoolExecutor() as pool:
            holders = self._start_holders(
                pool, HOLDER_MODEL, lambda _: self._holder_inputs()
            )

            (
                cancelled_callback,
                cancelled_response,
            ) = self._generate_callback_and_response_pair()
            cancelled_request = self._triton.async_infer(
                TRT_MODEL, self._trt_inputs(value=1.0), cancelled_callback
            )

            live_callback, live_response = self._generate_callback_and_response_pair()
            live_request = self._triton.async_infer(
                TRT_MODEL, self._trt_inputs(value=2.0), live_callback
            )

            self._wait_until_pending(TRT_MODEL, 2)
            self.assertFalse(
                cancelled_response["responded"],
                "the cancelled request was not held by the rate limiter",
            )
            self.assertFalse(
                live_response["responded"],
                "the live request was not held by the rate limiter",
            )
            self.assertIsNotNone(live_request)

            self._cancel_and_wait(cancelled_request)

            # Let the holders finish so the queued TRT requests can be scheduled.
            for holder in holders:
                holder.result(timeout=60)

            self._assert_response_is_cancelled(cancelled_response)
            self._wait_for_response(live_response)
            self.assertIsNone(live_response["error"])
            self.assertIsNotNone(live_response["result"])
            np.testing.assert_allclose(
                live_response["result"].as_numpy("OUTPUT0"),
                np.full(TRT_SHAPE, 4.0, dtype=np.float32),
            )

        self._assert_metrics(TRT_MODEL, "CANCELED", 1, failures_before)
        self._wait_for_execution_count(TRT_MODEL, executions_before + 1)


if __name__ == "__main__":
    unittest.main()
