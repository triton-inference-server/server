#!/usr/bin/env python3

# Copyright 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import threading
import time
import unittest

import numpy as np
import requests
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

    def _assert_response_is_cancelled(self, response, timeout=30):
        self.assertTrue(
            response["completed"].wait(timeout),
            f"inference callback was not invoked within {timeout}s",
        )
        self.assertTrue(response["responded"])
        self.assertEqual(response["result"], None)
        self.assertIsInstance(response["error"], InferenceServerException)
        self.assertEqual(response["error"].status(), "StatusCode.CANCELLED")

    def _generate_streaming_callback_and_response_pair(self):
        response = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _assert_streaming_response_is_cancelled(self, response):
        self.assertGreater(len(response), 0)
        cancelled_count = 0
        for res in response:
            result, error = res["result"], res["error"]
            if error:
                self.assertEqual(result, None)
                self.assertIsInstance(error, InferenceServerException)
                if error.status() == "StatusCode.CANCELLED":
                    cancelled_count += 1
        self.assertEqual(cancelled_count, 1)

    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url, timeout=5)
        r.raise_for_status()
        return r.text

    def _metric_value(self, metric_name, expected_labels):
        for line in self._get_metrics().splitlines():
            if not line.startswith(f"{metric_name}"):
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

    def _wait_until(self, predicate, description, timeout=30, interval=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval)
        self.fail(f"timed out after {timeout}s waiting for {description}")

    def _metrics_before_test(self, model, reason):
        value = self._metric_value(
            "nv_inference_request_failure",
            {"model": model, "reason": reason, "version": "1"},
        )
        if value is None:
            raise Exception(f"Failure metrics for model='{model}' not found")
        return value

    def _assert_metrics(
        self, model_name, reason, expected_count_increase, initial_count
    ):
        expected_count = initial_count + expected_count_increase
        labels = {"model": model_name, "reason": reason, "version": "1"}
        self._wait_until(
            lambda: self._metric_value("nv_inference_request_failure", labels)
            == expected_count,
            f"{model_name} {reason} failure count to reach {expected_count}",
        )

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
            self._assert_response_is_cancelled(response)
            # Join saturating thread
            saturate_thread_1.result(timeout=60)
            saturate_thread_2.result(timeout=60)

    def _get_inputs_with_value(self, batch_size, value):
        shape = [batch_size, 8]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.full(shape, value, dtype=np.float32))
        return inputs

    def _cache_hit_count(self, model_name):
        stats = self._triton.get_inference_statistics(
            model_name=model_name, as_json=True
        )
        model_stats = stats.get("model_stats", [])
        if not model_stats:
            return 0
        infer_stats = model_stats[0]["inference_stats"]
        return int(infer_stats.get("cache_hit", {}).get("count", 0))

    def _execution_count(self, model_name):
        stats = self._triton.get_inference_statistics(
            model_name=model_name, as_json=True
        )
        model_stats = stats.get("model_stats", [])
        if not model_stats:
            return 0
        return int(model_stats[0].get("execution_count", 0))

    def _pending_count(self, model_name):
        value = self._metric_value(
            "nv_inference_pending_request_count",
            {"model": model_name, "version": "1"},
        )
        return 0 if value is None else value

    def _wait_until_pending(self, model_name, expected, timeout=30, stable_for=0.5):
        """Wait until a pending count remains stable for the requested period."""
        deadline = time.monotonic() + timeout
        stable_since = None
        last_count = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            last_count = self._pending_count(model_name)
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

    def _hold_model_instance(self, pool, model_name, input_factory):
        """Run one request and keep another queued behind the same instance."""
        holders = [
            pool.submit(self._triton.infer, model_name, input_factory(index))
            for index in range(2)
        ]
        self._wait_until_pending(model_name, 1)
        return holders

    def _wait_for_execution_count(self, model_name, expected):
        self._wait_until(
            lambda: self._execution_count(model_name) >= expected,
            f"{model_name} execution count to reach {expected}",
        )
        self.assertEqual(self._execution_count(model_name), expected)

    # Test queued requests can be cancelled on a model with no batcher, where
    # the request waits on the rate limiter for a model instance
    def test_no_batcher_queued_request_cancellation(self):
        model_name = "no_batching"
        initial_metrics_value = self._metrics_before_test(model_name, "CANCELED")
        executions_before = self._execution_count(model_name)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # A stable pending request proves that one holder is executing and
            # another is queued behind the single model instance.
            holders = self._hold_model_instance(
                pool, model_name, lambda _: self._get_inputs(batch_size=1)
            )

            callback, response = self._generate_callback_and_response_pair()
            queue_future = self._triton.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            self._wait_until_pending(model_name, 2)
            self.assertFalse(response["responded"])

            queue_future.cancel()

            # The second holder keeps the instance occupied while cancellation
            # propagates. No execution-delay boundary is used for synchronization.
            for holder in holders:
                holder.result(timeout=60)
            self._assert_response_is_cancelled(response)

        self._assert_metrics(
            model_name,
            "CANCELED",
            1,
            initial_metrics_value,
        )
        self._wait_for_execution_count(model_name, executions_before + 2)

    # Test a cancelled response is not written to the response cache, which
    # would serve it as a successful hit to a later matching request
    def test_no_batcher_cancelled_response_is_not_cached(self):
        model_name = "no_batching_cache"
        cancelled_value = 2.0
        failures_before = self._metrics_before_test(model_name, "CANCELED")
        executions_before = self._execution_count(model_name)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Unique holder inputs avoid cache hits while the instance is held.
            holders = self._hold_model_instance(
                pool,
                model_name,
                lambda index: self._get_inputs_with_value(
                    batch_size=1, value=10.0 + index
                ),
            )

            callback, response = self._generate_callback_and_response_pair()
            queue_future = self._triton.async_infer(
                model_name,
                self._get_inputs_with_value(batch_size=1, value=cancelled_value),
                callback,
            )
            self._wait_until_pending(model_name, 2)
            self.assertFalse(response["responded"])

            queue_future.cancel()
            for holder in holders:
                holder.result(timeout=60)
            self._assert_response_is_cancelled(response)

        self._assert_metrics(model_name, "CANCELED", 1, failures_before)
        self._wait_for_execution_count(model_name, executions_before + 2)
        hits_after_cancellation = self._cache_hit_count(model_name)

        # The first identical request must execute. If the cancelled response
        # was cached, this request would incorrectly be served as a cache hit.
        first_result = self._triton.infer(
            model_name, self._get_inputs_with_value(batch_size=1, value=cancelled_value)
        )
        first_output = first_result.as_numpy("OUTPUT0")
        self.assertIsNotNone(first_output)
        np.testing.assert_allclose(
            first_output, np.full([1, 8], cancelled_value, dtype=np.float32)
        )
        self._wait_for_execution_count(model_name, executions_before + 3)
        hits_after_first_request = self._cache_hit_count(model_name)
        self.assertEqual(hits_after_first_request, hits_after_cancellation)

        # The successful response must now be cached. A second identical request
        # must be a cache hit and must not execute the backend again.
        second_result = self._triton.infer(
            model_name, self._get_inputs_with_value(batch_size=1, value=cancelled_value)
        )
        second_output = second_result.as_numpy("OUTPUT0")
        self.assertIsNotNone(second_output)
        np.testing.assert_allclose(
            second_output, np.full([1, 8], cancelled_value, dtype=np.float32)
        )
        self._wait_until(
            lambda: self._cache_hit_count(model_name) == hits_after_cancellation + 1,
            f"{model_name} cache hit count to increase by one",
        )
        self.assertEqual(self._execution_count(model_name), executions_before + 3)

    # Test backlogged requests on sequence batch scheduler can be cancelled
    def test_sequence_batch_scheduler_backlog_request_cancellation(self):
        model_name = "sequence_direct"
        initial_metrics_value = self._metrics_before_test(model_name, "CANCELED")
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
            self._assert_response_is_cancelled(backlog_requests[0]["response"])
            self._assert_response_is_cancelled(backlog_requests[1]["response"])
            # Join saturating thread
            saturate_thread.result(timeout=60)
        expected_count_increase = 2
        self._assert_metrics(
            model_name,
            "CANCELED",
            expected_count_increase,
            initial_metrics_value,
        )

    # Test queued requests on direct sequence batch scheduler can be cancelled
    def test_direct_sequence_batch_scheduler_request_cancellation(self):
        model_name = "sequence_direct"
        initial_metrics_value = self._metrics_before_test(model_name, "CANCELED")
        self._test_sequence_batch_scheduler_queued_request_cancellation(model_name)
        expected_count_increase = 2
        self._assert_metrics(
            model_name,
            "CANCELED",
            expected_count_increase,
            initial_metrics_value,
        )

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
            self._assert_response_is_cancelled(queue_requests[0]["response"])
            self._assert_response_is_cancelled(queue_requests[1]["response"])
            # Join start thread
            start_thread.result(timeout=60)

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
        self._assert_response_is_cancelled(response)

    # Test cancellation on multiple gRPC streaming sequences
    def test_scheduler_streaming_request_cancellation(self):
        model_name = "sequence_oldest"
        # Start 2 sequences with many requests
        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        for sequence_id in [1, 2]:
            sequence_start = True
            for request_id in range(16):
                self._triton.async_stream_infer(
                    model_name,
                    self._get_inputs(batch_size=1),
                    sequence_id=sequence_id,
                    sequence_start=sequence_start,
                )
                sequence_start = False
        time.sleep(2)  # ensure the requests are delivered
        # Cancelling the stream cancels all requests on the stream
        self._triton.stop_stream(cancel_requests=True)
        time.sleep(2)  # ensure the cancellation is delivered
        time.sleep(2)  # ensure reaper thread has responded
        self._assert_streaming_response_is_cancelled(response)


if __name__ == "__main__":
    unittest.main()
