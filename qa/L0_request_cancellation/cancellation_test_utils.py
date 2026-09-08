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

import os
import re
import threading
import time

import requests
from tritonclient.utils import InferenceServerException


class CancellationTest:
    """Shared synchronization and server-state helpers for cancellation tests."""

    # Build an event-backed callback and mutable response state for async infer.
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

    # Wait for an async inference callback with a bounded timeout.
    def _wait_for_response(self, response, timeout=30):
        self.assertTrue(
            response["completed"].wait(timeout),
            f"inference callback was not invoked within {timeout}s",
        )

    # Verify that an async request completed with the gRPC CANCELLED status.
    def _assert_response_is_cancelled(self, response, timeout=30):
        self._wait_for_response(response, timeout)
        self.assertTrue(response["responded"])
        self.assertIsNone(response["result"])
        self.assertIsInstance(response["error"], InferenceServerException)
        self.assertEqual(response["error"].status(), "StatusCode.CANCELLED")

    # Read one labeled Prometheus metric value independent of label ordering.
    def _metric_value(self, metric_name, expected_labels):
        response = requests.get("http://localhost:8002/metrics", timeout=5)
        response.raise_for_status()
        for line in response.text.splitlines():
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

    # Return the backend execution count reported for a model.
    def _execution_count(self, model_name):
        stats = self._triton.get_inference_statistics(
            model_name=model_name, as_json=True
        )
        model_stats = stats.get("model_stats", [])
        if not model_stats:
            return 0
        return int(model_stats[0].get("execution_count", 0))

    # Return the failure count for a model and failure reason.
    def _failure_count(self, model_name, reason):
        value = self._metric_value(
            "nv_inference_request_failure",
            {"model": model_name, "reason": reason, "version": "1"},
        )
        return 0 if value is None else value

    # Wait for a failure metric to increase by the expected amount.
    def _assert_metrics(
        self, model_name, reason, expected_count_increase, initial_count
    ):
        expected_count = initial_count + expected_count_increase
        self._wait_until(
            lambda: self._failure_count(model_name, reason) == expected_count,
            f"{model_name} {reason} failure count to reach {expected_count}",
        )

    # Poll a condition until it succeeds or reaches a bounded timeout.
    def _wait_until(self, predicate, description, timeout=30, interval=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval)
        self.fail(f"timed out after {timeout}s waiting for {description}")

    # Cancel a request and wait for cancellation log message.
    def _cancel_and_wait(self, request, request_id):
        def cancellation_count():
            with open(os.environ["SERVER_LOG"], encoding="utf-8") as server_log:
                return server_log.read().count(
                    f"[request id: {request_id}] Cancellation issued"
                )

        cancellations_before = cancellation_count()
        request.cancel()
        self._wait_until(
            lambda: cancellation_count() > cancellations_before,
            "the server to issue cancellation",
        )

    # Wait until a model's pending-request count is stable at the expected value.
    def _wait_until_pending(self, model_name, expected, timeout=30, stable_for=0.5):
        deadline = time.monotonic() + timeout
        stable_since = None
        last_count = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            last_count = self._metric_value(
                "nv_inference_pending_request_count",
                {"model": model_name, "version": "1"},
            )
            last_count = 0 if last_count is None else last_count
            if last_count == expected:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= stable_for:
                    return
            else:
                stable_since = None
            time.sleep(0.1)

        self.fail(
            f"'{model_name}' pending count did not remain at {expected} for "
            f"{stable_for}s within {timeout}s; last count was {last_count}"
        )

    # Start two holders and wait until one is queued behind the other.
    def _start_holders(self, pool, model_name, input_factory):
        holders = [
            pool.submit(self._triton.infer, model_name, input_factory(index))
            for index in range(2)
        ]
        self._wait_until_pending(model_name, 1)
        return holders

    # Wait for an exact execution count so extra backend execution is detected.
    def _wait_for_execution_count(self, model_name, expected):
        self._wait_until(
            lambda: self._execution_count(model_name) >= expected,
            f"{model_name} execution count to reach {expected}",
        )
        self.assertEqual(self._execution_count(model_name), expected)
