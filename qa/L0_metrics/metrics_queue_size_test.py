#!/usr/bin/python
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

import sys

sys.path.append("../common")

import math
import time
import unittest
from functools import partial

import numpy as np
import requests
import test_util as tu
import tritonclient.http
from tritonclient.utils import triton_to_np_dtype

QUEUE_METRIC_TEMPLATE = (
    'nv_inference_pending_request_count{{model="{model_name}",version="1"}}'
)
INFER_METRIC_TEMPLATE = 'nv_inference_count{{model="{model_name}",version="1"}}'
EXEC_METRIC_TEMPLATE = 'nv_inference_exec_count{{model="{model_name}",version="1"}}'


class MetricsPendingRequestCountTest(tu.TestResultCollector):
    def setUp(self):
        self.metrics = None
        self.metrics_url = "http://localhost:8002/metrics"
        self.server_url = "localhost:8000"

        # Used to verify model config is set to expected values
        self.max_batch_size = 4
        self.delay_ms = 2000
        self.delay_sec = self.delay_ms // 1000

        # Setup dummy inputs
        dtype = "FP32"
        shape = (1, 1)
        input_np = np.ones(shape, dtype=triton_to_np_dtype(dtype))
        self.inputs = [
            tritonclient.http.InferInput("INPUT0", shape, dtype).set_data_from_numpy(
                input_np
            )
        ]
        self.ensemble_inputs = [
            tritonclient.http.InferInput(
                "ENSEMBLE_INPUT0", shape, dtype
            ).set_data_from_numpy(input_np)
        ]

        # Verify values for filling request queues
        self.num_requests = 10
        self.concurrency = 10
        # Concurrency must be at least as high as number of async requests we intend
        # to send N requests to fill request queues before blocking on any results.
        self.assertGreaterEqual(self.concurrency, self.num_requests)
        self.client = tritonclient.http.InferenceServerClient(
            url=self.server_url, concurrency=self.concurrency
        )

        # Test specific configurations
        self.max_queue_size = 0

    def _validate_model_config(self, model_name, max_queue_size=0):
        config = self.client.get_model_config(model_name)
        print(config)
        params = config.get("parameters", {})
        delay_ms = int(params.get("execute_delay_ms", {}).get("string_value"))
        max_batch_size = config.get("max_batch_size")
        self.assertEqual(delay_ms, self.delay_ms)
        self.assertEqual(max_batch_size, self.max_batch_size)

        dynamic_batching = config.get("dynamic_batching", {})
        default_queue_policy = dynamic_batching.get("default_queue_policy", {})
        self.max_queue_size = default_queue_policy.get("max_queue_size", 0)

        self.assertEqual(self.max_queue_size, max_queue_size)

        return config

    def _get_metrics(self):
        r = requests.get(self.metrics_url)
        r.raise_for_status()
        return r.text

    def _get_metric_line(self, metric, metrics):
        for line in metrics.splitlines():
            if metric in line:
                return line
        return None

    def _get_metric_value(self, metric):
        metrics = self._get_metrics()
        self.assertIn(metric, metrics)
        line = self._get_metric_line(metric, metrics)
        print(line)
        if not line:
            return None
        value = line.split()[1]
        return float(value)

    def _assert_metric_equals(self, metric, expected_value):
        value = self._get_metric_value(metric)
        self.assertEqual(value, expected_value)

    def _assert_metric_greater_than(self, metric, gt_value):
        value = self._get_metric_value(metric)
        self.assertGreater(value, gt_value)

    def _send_async_requests(self, model_name, inputs, futures):
        for _ in range(self.num_requests):
            futures.append(self.client.async_infer(model_name, inputs))

    def _send_async_requests_sequence(self, num_seq_slots, model_name, inputs, futures):
        started_seqs = {}
        num_sent = 0
        while num_sent < self.num_requests:
            # Add requests to each sequence slot round-robin, seq_id must be > 0
            # We don't care about finishing any sequences, just need to queue up
            # requests for each sequence until num_requests is hit.
            seq_id = (num_sent % num_seq_slots) + 1
            # Toggle start flag to False after first request per sequence ID
            start = True if seq_id not in started_seqs else False
            started_seqs[seq_id] = True
            futures.append(
                self.client.async_infer(
                    model_name,
                    inputs,
                    request_id=str(num_sent),
                    sequence_id=seq_id,
                    sequence_start=start,
                )
            )
            num_sent += 1

    def _test_helper(
        self, model_name, batch_size, send_requests_func, max_queue_size=0
    ):
        self._validate_model_config(model_name, max_queue_size=max_queue_size)

        queue_size = QUEUE_METRIC_TEMPLATE.format(model_name=model_name)
        infer_count = INFER_METRIC_TEMPLATE.format(model_name=model_name)
        exec_count = EXEC_METRIC_TEMPLATE.format(model_name=model_name)
        # Metric should be zero before sending any requests
        self._assert_metric_equals(queue_size, 0)
        # Send N requests, letting scheduler delay queue fill up when applicable
        futures = []
        send_requests_func(model_name, self.inputs, futures)
        # Give Triton a second to load all requests into queues
        time.sleep(1)

        # Start from (num_requests-batch_size) because 1 batch should be executing,
        # and the rest of the requests should be queued.
        # If max_queue_size is specified then the queued requests would be capped
        # at max_queue_size.
        if max_queue_size != 0:
            self._assert_metric_equals(queue_size, max_queue_size)
            starting_queue_size = max_queue_size
        else:
            starting_queue_size = self.num_requests - batch_size

        for expected_queue_size in range(starting_queue_size, 0, -1 * batch_size):
            self._assert_metric_equals(queue_size, expected_queue_size)
            time.sleep(self.delay_sec)
        # Queue should be empty now
        self._assert_metric_equals(queue_size, 0)
        # Let final batch finish
        time.sleep(self.delay_sec)

        # All requests should've been executed without any batching
        expected_infer_count = starting_queue_size + batch_size
        self._assert_metric_equals(infer_count, expected_infer_count)
        expected_exec_count = math.ceil(expected_infer_count / batch_size)
        self._assert_metric_equals(exec_count, expected_exec_count)

        failed_count = 0
        for future in futures:
            try:
                future.get_result()
            except Exception as e:
                failed_count = failed_count + 1

        self.assertEqual(
            failed_count, self.num_requests - batch_size - starting_queue_size
        )

    def test_default_scheduler(self):
        model_name = "default"
        # Default scheduler won't do any batching
        batch_size = 1
        self._test_helper(model_name, batch_size, self._send_async_requests)

    def test_dynamic_batch_scheduler(self):
        model_name = "dynamic"
        # With sufficient queue delay set, we expect full batches to be executed
        batch_size = self.max_batch_size
        self._test_helper(model_name, batch_size, self._send_async_requests)

    def test_fail_max_queue_size(self):
        model_name = "max_queue_size"
        # This test checks whether metrics are properly accounts for requests
        # that fail to enqueue on the server. The test sets the max_queue_size
        # and any additional requests beyond the specified queue size should fail
        # instead of waiting for execution.
        batch_size = self.max_batch_size
        self._test_helper(
            model_name, batch_size, self._send_async_requests, max_queue_size=4
        )

    def test_sequence_batch_scheduler_direct(self):
        model_name = "sequence_direct"
        # With sufficient queue delay and minimum_slot_utilization set, we
        # expect full batches to be executed.
        batch_size = self.max_batch_size
        num_seq_slots = batch_size
        send_requests_func = partial(self._send_async_requests_sequence, num_seq_slots)
        self._test_helper(model_name, batch_size, send_requests_func)

    def test_sequence_batch_scheduler_oldest(self):
        model_name = "sequence_oldest"
        # With sufficient queue delay set, we expect full batches to be executed
        batch_size = self.max_batch_size
        num_seq_slots = batch_size
        send_requests_func = partial(self._send_async_requests_sequence, num_seq_slots)
        self._test_helper(model_name, batch_size, send_requests_func)

    def test_ensemble_scheduler(self):
        ensemble_model_name = "ensemble"
        composing_model_names = ["dynamic_composing", "default_composing"]
        ensemble_queue_size = QUEUE_METRIC_TEMPLATE.format(
            model_name=ensemble_model_name
        )
        composing_queue_sizes = [
            QUEUE_METRIC_TEMPLATE.format(model_name=name)
            for name in composing_model_names
        ]
        ensemble_infer_count = INFER_METRIC_TEMPLATE.format(
            model_name=ensemble_model_name
        )
        composing_infer_counts = [
            INFER_METRIC_TEMPLATE.format(model_name=name)
            for name in composing_model_names
        ]

        # Metric should be zero before sending any requests
        self._assert_metric_equals(ensemble_queue_size, 0)
        for queue_size in composing_queue_sizes:
            self._assert_metric_equals(queue_size, 0)
        # Send some ensemble requests
        futures = []
        self._send_async_requests(ensemble_model_name, self.ensemble_inputs, futures)
        # Give Triton time to pass some requests to composing models. This test
        # is less comprehensive on checking exact queue values, and just verifies
        # each composing queue gets filled and ensemble's queue is empty.
        time.sleep(1)

        # Top-level ensemble size should still be zero, as all pending requests should
        # be scheduled and reflected in composing models, and not considered "pending" at ensemble level.
        self._assert_metric_equals(ensemble_queue_size, 0)
        # Composing models should be non-zero
        for queue_size in composing_queue_sizes:
            self._assert_metric_greater_than(queue_size, 0)

        # Verify no inference exceptions were raised and let composing models
        # finish their requests
        for future in futures:
            future.get_result()

        # Check that all queues are empty after getting results
        self._assert_metric_equals(ensemble_queue_size, 0)
        for queue_size in composing_queue_sizes:
            self._assert_metric_equals(queue_size, 0)

        # Sanity check infer counts on ensemble and composing models
        self._assert_metric_equals(ensemble_infer_count, self.num_requests)
        for infer_count in composing_infer_counts:
            self._assert_metric_equals(infer_count, self.num_requests)


if __name__ == "__main__":
    unittest.main()
