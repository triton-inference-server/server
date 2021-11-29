# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import numpy as np
import os
import unittest
import threading
import time
import sequence_util as su
import tritongrpcclient as grpcclient
from tritonclientutils import *

_inference_count = 80
_inference_concurrency = 8
_response_wait_time_s = 10
_finish_wait_time_s = 10
_exit_signal = False


class AsyncGrpcRunner:

    def __init__(self, tester, server_url, model_name, delay_ms):
        self._tester = tester
        self._server_url = server_url
        self._model_name = model_name
        self._delay_ms = delay_ms

        self._input_data = []
        self._shape = [1, 1]
        self._dtype = np.float32
        self._results = {}
        self._processed_all = False
        self._errors = []
        self._inflight_requests = 0
        self._num_sent_request = 0
        self._processed_request_count = 0
        self._sync = threading.Condition()
        self._req_thread = threading.Thread(target=self.req_loop, daemon=True)

    def _on_result(self, result, error):
        with self._sync:
            if error:
                self._errors.append(error)
            else:
                this_id = int(result.get_response().id)
                self._results[this_id] = result
            self._inflight_requests -= 1
            self._sync.notify_all()

    def req_loop(self):
        client = grpcclient.InferenceServerClient(self._server_url)

        inputs = [
            grpcclient.InferInput("INPUT0", self._shape,
                                  np_to_triton_dtype(self._dtype))
        ]

        self._inflight_requests = 0
        start_stat = client.get_inference_statistics(
            model_name=self._model_name)
        global _exit_signal

        while not _exit_signal:
            input_numpy = np.random.random_sample(self._shape).astype(
                self._dtype)
            inputs[0].set_data_from_numpy(input_numpy)
            self._input_data.append(input_numpy)

            with self._sync:

                def _check_can_send():
                    return self._inflight_requests < _inference_concurrency

                can_send = self._sync.wait_for(_check_can_send,
                                               timeout=_response_wait_time_s)
                self._tester.assertTrue(
                    can_send,
                    "client didn't receive a response within {}s".format(
                        _response_wait_time_s))

                callback = functools.partial(AsyncGrpcRunner._on_result, self)
                client.async_infer(
                    model_name=self._model_name,
                    inputs=inputs,
                    request_id="{}".format(self._num_sent_request),
                    callback=callback,
                )
                self._inflight_requests += 1
                self._num_sent_request += 1
                if (self._num_sent_request == _inference_count):
                    _exit_signal = True
                time.sleep(self._delay_ms / 1000.0)

        # wait till receive all requested data
        with self._sync:

            def _all_processed():
                return self._inflight_requests == 0

            self._processed_all = self._sync.wait_for(_all_processed,
                                                      _finish_wait_time_s)
            self._tester.assertTrue(
                self._processed_all,
                "the processing didn't complete even after waiting for {}s".
                format(_finish_wait_time_s))

        end_stat = client.get_inference_statistics(model_name=self._model_name)
        self._processed_request_count = end_stat.model_stats[
            0].inference_stats.success.count - start_stat.model_stats[
                0].inference_stats.success.count

    def start(self):
        self._req_thread.start()

    def _validate_run(self):
        if len(self._errors) != 0:
            raise self._errors[0]
        self._tester.assertEqual(
            len(self._input_data), len(self._results.keys()),
            "the number of inputs and output should match")
        for i in range(len(self._input_data)):
            self._tester.assertFalse(
                (self._input_data[i] !=
                 self._results[i].as_numpy('OUTPUT0')).any(),
                "the output data should match with the input data")

    def join(self):
        self._req_thread.join()
        self._validate_run()


class RateLimiterTest(su.SequenceBatcherTestUtil):

    def stress_models(self, model_names, delay_ms=0):
        infer_counts = {}
        try:
            runners = []
            for model_name in model_names:
                runners.append(
                    AsyncGrpcRunner(self,
                                    "localhost:8001",
                                    model_name,
                                    delay_ms=delay_ms))
            for r in runners:
                r.start()
            for r in runners:
                r.join()
                infer_counts[r._model_name] = r._processed_request_count
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        return infer_counts

    def test_single_model(self):
        # Send all the inference requests to a single model.
        # Simple sanity check.

        model_names = ["custom_zero_1_float32"]
        infer_counts = self.stress_models(model_names)

        self.assertEqual(infer_counts[model_names[0]], _inference_count)

    def test_cross_model_prioritization_limited_resource(self):
        # Sends requests to two models, one operating at
        # priority of 1 and other at 2 respectively.
        # The availabe resource counts doesn't allow models
        # to execute simultaneously.

        model_names = ["custom_zero_1_float32", "custom_zero_1_float32_v2"]

        # TODO: Validate the priority and resource counts are set correctly

        infer_counts = self.stress_models(model_names)
        infer_ratio = infer_counts[model_names[0]] / float(
            infer_counts[model_names[1]])

        self.assertGreater(
            infer_ratio, 1.80,
            "Got infer ratio across models {}, expected closer to 2".format(
                infer_ratio))

    def test_cross_model_prioritization_plenty_resource(self):
        # Sends requests to two models, one operating at
        # priority of 1 and other at 2 respectively.
        # The availabe resource counts wll allow both models
        # to run simulataneously.

        model_names = ["custom_zero_1_float32", "custom_zero_1_float32_v2"]

        # TODO: Validate the priority and resource counts are set correctly

        infer_counts = self.stress_models(model_names)
        infer_diff = abs(infer_counts[model_names[0]] -
                         infer_counts[model_names[1]])

        self.assertGreater(
            10, infer_diff,
            "Got infer difference between models {}, expected closer to 0".
            format(infer_diff))

    def test_single_model_dynamic_batching(self):
        # Send all the inference requests with a delay to a model

        self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
        model_names = ["custom_zero_1_float32"]
        infer_counts = self.stress_models(model_names, delay_ms=100)

        self.assertEqual(infer_counts[model_names[0]], _inference_count)

        # Check whether all requests used batch size of 4 or not
        client = grpcclient.InferenceServerClient("localhost:8001")
        stats = client.get_inference_statistics(model_names[0], "1")
        self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")

        batch_stats = stats.model_stats[0].batch_stats
        self.assertEqual(
            len(batch_stats), 1,
            "expected single batch-size, got {}".format(len(batch_stats)))

        for batch_stat in batch_stats:
            self.assertEqual(
                batch_stat.batch_size, 4,
                "unexpected batch-size {}".format(batch_stat.batch_size))
            # Get count from one of the stats
            self.assertEqual(
                batch_stat.compute_infer.count, _inference_count / 4,
                "expected model-execution-count {} for batch size {}, got {}".
                format(_inference_count / 4, 4, batch_stat.compute_infer.count))

    def test_single_model_sequence_batching(self):
        # Send one sequence and check for correct accumulator
        # result. The result should be returned immediately.
        # This test checks whether all the requests are
        # directed to the same instance.

        try:
            model_name = "custom_sequence_int32"
            self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
            self.check_sequence(
                'custom',
                model_name,
                np.int32,
                5,
                (4000, None),
                # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                (("start", 1, None, None), (None, 2, None, None),
                 (None, 3, None, None), (None, 4, None, None),
                 (None, 5, None, None), (None, 6, None, None),
                 (None, 7, None, None), (None, 8, None, None),
                 ("end", 9, None, None)),
                45,
                'grpc')

            self.check_deferred_exception()
            self.check_status(model_name, {1: 9}, 9, 9)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
