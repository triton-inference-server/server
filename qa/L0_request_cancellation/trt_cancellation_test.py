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
import os
import re
import threading
import time
import unittest

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# A TensorRT request cancelled while waiting in the rate limiter must be
# discarded by core before backend dispatch. Both models need the one global
# "SHARED" resource, so 'resource_holder' keeps the TensorRT requests queued.
TRT_MODEL = "plan_no_batching"
HOLDER_MODEL = "resource_holder"
CANCELLATION_LOG_LINE = "Cancellation notification received for "

# Plan model from qa_model_repository: OUTPUT0 = INPUT0 + INPUT1.
TRT_SHAPE = [1, 16]
HOLDER_SHAPE = [1, 8]


class TestTrtRequestCancellation(unittest.TestCase):
    def setUp(self):
        self._triton = grpcclient.InferenceServerClient("localhost:8001")
        self._server_log = os.environ["SERVER_LOG"]

    def _trt_inputs(self, value):
        inputs = [
            grpcclient.InferInput("INPUT0", TRT_SHAPE, "FP32"),
            grpcclient.InferInput("INPUT1", TRT_SHAPE, "FP32"),
        ]
        for model_input in inputs:
            model_input.set_data_from_numpy(np.full(TRT_SHAPE, value, dtype=np.float32))
        return inputs

    def _holder_inputs(self):
        inputs = [grpcclient.InferInput("INPUT0", HOLDER_SHAPE, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(HOLDER_SHAPE, dtype=np.float32))
        return inputs

    def _generate_callback_and_response_pair(self):
        response = {
            "completed": threading.Event(),
            "responded": False,
            "result": None,
            "error": None,
        }

        def callback(result, error):
            response["result"] = result
            response["error"] = error
            response["responded"] = True
            response["completed"].set()

        return callback, response

    def _wait_for_response(self, response, timeout=30):
        self.assertTrue(
            response["completed"].wait(timeout),
            f"inference callback was not invoked within {timeout}s",
        )

    def _assert_response_is_cancelled(self, response, timeout=30):
        self._wait_for_response(response, timeout)
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
        r = requests.get("http://localhost:8002/metrics", timeout=5)
        r.raise_for_status()
        return r.text

    def _cancellation_notification_count(self):
        with open(self._server_log, encoding="utf-8") as server_log:
            return server_log.read().count(CANCELLATION_LOG_LINE)

    def _metric_value(self, metric_name, expected_labels):
        for line in self._get_metrics().splitlines():
            if not line.startswith(f"{metric_name}{{"):
                continue

            fields = line.split()
            if len(fields) < 2:
                continue

            metric_and_labels = fields[0]
            label_text = metric_and_labels[
                len(metric_name) + 1 : metric_and_labels.rfind("}")
            ]
            labels = dict(re.findall(r'(\w+)="([^"]*)"', label_text))
            if all(labels.get(key) == value for key, value in expected_labels.items()):
                return int(float(fields[1]))

        return None

    def _failure_count(self, model, reason):
        value = self._metric_value(
            "nv_inference_request_failure",
            {"model": model, "reason": reason, "version": "1"},
        )
        return 0 if value is None else value

    def _pending_count(self, model):
        value = self._metric_value(
            "nv_inference_pending_request_count",
            {"model": model, "version": "1"},
        )
        return 0 if value is None else value

    def _wait_until(self, predicate, description, timeout=30, interval=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval)
        self.fail(f"timed out after {timeout}s waiting for {description}")

    def _wait_until_pending(self, model, expected, timeout=30, stable_for=0.5):
        """Wait until a pending count remains stable for the requested period."""
        deadline = time.monotonic() + timeout
        stable_since = None
        last_count = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            last_count = self._pending_count(model)
            if last_count == expected:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= stable_for:
                    return
            else:
                stable_since = None
            time.sleep(0.1)

        self.fail(
            f"'{model}' pending count did not remain at {expected} for "
            f"{stable_for}s within {timeout}s; last count was {last_count}"
        )

    def _hold_resource(self, pool):
        """Hold the shared resource and wait until it is held.
        One request executes, the other stays queued behind it.
        """
        holders = [
            pool.submit(self._triton.infer, HOLDER_MODEL, self._holder_inputs())
            for _ in range(2)
        ]
        self._wait_until_pending(HOLDER_MODEL, 1)
        return holders

    # Core must discard a cancelled TensorRT request after it leaves the rate
    # limiter, while an adjacent live request must still execute normally.
    def test_trt_rate_limited_cancellation_skips_only_cancelled_request(self):
        executions_before = self._execution_count(TRT_MODEL)
        failures_before = self._failure_count(TRT_MODEL, "CANCELED")

        with concurrent.futures.ThreadPoolExecutor() as pool:
            holders = self._hold_resource(pool)

            (
                cancelled_callback,
                cancelled_response
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

            notifications_before = self._cancellation_notification_count()
            cancelled_request.cancel()
            self._wait_until(
                lambda: self._cancellation_notification_count() > notifications_before,
                "the server to receive the gRPC cancellation notification",
            )

            # The server-side notification is the synchronization point; the
            # holder completion is not used as a cancellation-delivery delay.
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

        expected_failures = failures_before + 1
        self._wait_until(
            lambda: self._failure_count(TRT_MODEL, "CANCELED") == expected_failures,
            f"{TRT_MODEL} CANCELED failure count to reach {expected_failures}",
        )
        expected_executions = executions_before + 1
        self._wait_until(
            lambda: self._execution_count(TRT_MODEL) >= expected_executions,
            f"{TRT_MODEL} execution count to reach {expected_executions}",
        )
        self.assertEqual(
            self._execution_count(TRT_MODEL),
            expected_executions,
            "the TensorRT model executed a request that had been cancelled",
        )


if __name__ == "__main__":
    unittest.main()
