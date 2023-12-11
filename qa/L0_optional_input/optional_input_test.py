#!/usr/bin/python

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import threading
import time
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


# Similar set up as dynamic batcher tests
class OptionalInputTest(tu.TestResultCollector):
    def setUp(self):
        global _deferred_exceptions
        _deferred_exceptions = []

        # The helper client for setup will be GRPC for simplicity.
        self.triton_client_ = grpcclient.InferenceServerClient("localhost:8001")
        self.model_name_ = "identity_2_float32"
        # This will not be changed even when ensemble is under test,
        # as the dynamic batching is performed within the composing model
        self.check_status_model = "identity_2_float32"
        self.tensor_shape_ = (1, 1)
        self.inputs_ = {
            "INPUT0": grpcclient.InferInput("INPUT0", [1, 1], "FP32"),
            "INPUT1": grpcclient.InferInput("INPUT1", [1, 1], "FP32"),
        }
        self.input_data_ = {
            "INPUT0": np.ones(shape=(1, 1), dtype=np.float32),
            "INPUT1": np.zeros(shape=(1, 1), dtype=np.float32),
        }
        self.inputs_["INPUT0"].set_data_from_numpy(self.input_data_["INPUT0"])
        self.inputs_["INPUT1"].set_data_from_numpy(self.input_data_["INPUT1"])
        self.outputs_ = {
            "INPUT0": grpcclient.InferRequestedOutput("OUTPUT0"),
            "INPUT1": grpcclient.InferRequestedOutput("OUTPUT1"),
        }

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        # Just raise one of the exceptions...
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_response(self, thresholds, provided_inputs=("INPUT0", "INPUT1")):
        try:
            start_ms = int(round(time.time() * 1000))

            inputs = []
            outputs = []
            for provided_input in provided_inputs:
                inputs.append(self.inputs_[provided_input])
                outputs.append(self.outputs_[provided_input])

            triton_client = grpcclient.InferenceServerClient("localhost:8001")
            results = triton_client.infer(
                model_name=self.model_name_, inputs=inputs, outputs=outputs
            )

            end_ms = int(round(time.time() * 1000))

            for provided_input in provided_inputs:
                output_name = self.outputs_[provided_input].name()
                expected = self.input_data_[provided_input]
                output_data = results.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected),
                    "{}, {}, expected: {}, got {}".format(
                        self.model_name_, output_name, expected, output_data
                    ),
                )

            gt_ms = thresholds[0]
            lt_ms = thresholds[1]
            if lt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) < lt_ms,
                    "expected less than "
                    + str(lt_ms)
                    + "ms response time, got "
                    + str(end_ms - start_ms)
                    + " ms",
                )
            if gt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) > gt_ms,
                    "expected greater than "
                    + str(gt_ms)
                    + "ms response time, got "
                    + str(end_ms - start_ms)
                    + " ms",
                )
        except Exception as ex:
            self.add_deferred_exception(ex)

    def check_status(self, model_name, batch_exec, request_cnt, infer_cnt):
        # There is a time window between when responses are returned and statistics are updated.
        # To prevent intermittent test failure during that window, wait up to 10 seconds for the
        # inference statistics to be ready.
        num_tries = 10
        for i in range(num_tries):
            stats = self.triton_client_.get_inference_statistics(model_name, "1")
            self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
            actual_exec_cnt = stats.model_stats[0].execution_count
            if stats.model_stats[0].execution_count > 0:
                break
            time.sleep(1)

        self.assertEqual(
            stats.model_stats[0].name,
            model_name,
            "expect model stats for model {}".format(model_name),
        )
        self.assertEqual(
            stats.model_stats[0].version,
            "1",
            "expect model stats for model {} version 1".format(model_name),
        )

        batch_stats = stats.model_stats[0].batch_stats
        self.assertEqual(
            len(batch_stats),
            len(batch_exec),
            "expected {} different batch-sizes, got {}".format(
                len(batch_exec), len(batch_stats)
            ),
        )

        for batch_stat in batch_stats:
            bs = batch_stat.batch_size
            bc = batch_stat.compute_infer.count
            self.assertTrue(bs in batch_exec, "unexpected batch-size {}".format(bs))
            # Get count from one of the stats
            self.assertEqual(
                bc,
                batch_exec[bs],
                "expected model-execution-count {} for batch size {}, got {}".format(
                    batch_exec[bs], bs, bc
                ),
            )

        actual_request_cnt = stats.model_stats[0].inference_stats.success.count
        self.assertEqual(
            actual_request_cnt,
            request_cnt,
            "expected model-request-count {}, got {}".format(
                request_cnt, actual_request_cnt
            ),
        )

        actual_exec_cnt = stats.model_stats[0].execution_count
        self.assertEqual(
            actual_request_cnt,
            request_cnt,
            "expected model-exec-count {}, got {}".format(request_cnt, actual_exec_cnt),
        )

        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt,
            infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt
            ),
        )

    def test_all_inputs(self):
        # Provide all inputs, send requests that don't form preferred batch
        # so all requests should be returned after the queue delay
        try:
            threads = []
            threads.append(
                threading.Thread(target=self.check_response, args=((4000, None),))
            )
            threads.append(
                threading.Thread(target=self.check_response, args=((4000, None),))
            )
            threads[0].start()
            threads[1].start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(self.check_status_model, {2: 1}, 2, 2)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_optional_same_input(self):
        # Provide only one of the inputs, send requests that don't form
        # preferred batch so all requests should be returned after
        # the queue delay
        try:
            threads = []
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((4000, None),),
                    kwargs={"provided_inputs": ("INPUT1",)},
                )
            )
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((4000, None),),
                    kwargs={"provided_inputs": ("INPUT1",)},
                )
            )
            threads[0].start()
            threads[1].start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(self.check_status_model, {2: 1}, 2, 2)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_optional_mix_inputs(self):
        # Each request provides one of the inputs interleavingly,
        # all requests except the last one should be returned in less
        # than the queue delay because batcher should send the batch immediately
        # when it sees the provided inputs are different
        try:
            threads = []
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((0, 4000),),
                    kwargs={"provided_inputs": ("INPUT0",)},
                )
            )
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((0, 4000),),
                    kwargs={"provided_inputs": ("INPUT1",)},
                )
            )

            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((0, 4000),),
                    kwargs={"provided_inputs": ("INPUT0",)},
                )
            )
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((4000, None),),
                    kwargs={"provided_inputs": ("INPUT1",)},
                )
            )
            for t in threads:
                t.start()
                time.sleep(0.5)

            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(self.check_status_model, {1: 4}, 4, 4)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_optional_mix_inputs_2(self):
        # Each request provides one of the inputs or all inputs interleavingly,
        # all requests except the last one should be returned in less
        # than the queue delay because batcher should send the batch immediately
        # when it sees the provided inputs are different
        try:
            threads = []
            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((0, 4000),),
                    kwargs={"provided_inputs": ("INPUT0",)},
                )
            )
            threads.append(
                threading.Thread(target=self.check_response, args=((0, 4000),))
            )

            threads.append(
                threading.Thread(
                    target=self.check_response,
                    args=((0, 4000),),
                    kwargs={"provided_inputs": ("INPUT0",)},
                )
            )
            threads.append(
                threading.Thread(target=self.check_response, args=((4000, None),))
            )
            for t in threads:
                t.start()
                time.sleep(0.5)

            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(self.check_status_model, {1: 4}, 4, 4)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_ensemble_all_inputs(self):
        # The ensemble is only a wrapper over 'identity_2_float32'
        self.model_name_ = "ensemble_identity_2_float32"
        self.test_all_inputs()
        # From the ensemble's perspective, the requests are processed as it is
        self.check_status(self.model_name_, {1: 2}, 2, 2)

    def test_ensemble_optional_same_input(self):
        # The ensemble is only a wrapper over 'identity_2_float32'
        self.model_name_ = "ensemble_identity_2_float32"
        self.test_optional_same_input()
        # From the ensemble's perspective, the requests are processed as it is
        self.check_status(self.model_name_, {1: 2}, 2, 2)

    def test_ensemble_optional_mix_inputs(self):
        # The ensemble is only a wrapper over 'identity_2_float32'
        self.model_name_ = "ensemble_identity_2_float32"
        self.test_optional_mix_inputs()
        # From the ensemble's perspective, the requests are processed as it is
        self.check_status(self.model_name_, {1: 4}, 4, 4)

    def test_ensemble_optional_mix_inputs_2(self):
        # The ensemble is only a wrapper over 'identity_2_float32'
        self.model_name_ = "ensemble_identity_2_float32"
        self.test_optional_mix_inputs_2()
        # From the ensemble's perspective, the requests are processed as it is
        self.check_status(self.model_name_, {1: 4}, 4, 4)

    def test_ensemble_optional_pipeline(self):
        # The ensemble is a special case of pipelining models with optional
        # inputs, where the ensemble step only connects a subset of inputs
        # for the second model (which is valid because the disconnected inputs
        # are marked optional). See 'config.pbtxt' for detail.
        self.model_name_ = "pipeline_identity_2_float32"

        # Provide all inputs, send requests that don't form preferred batch
        # so all requests should be returned after the queue delay
        try:
            provided_inputs = ("INPUT0", "INPUT1")
            inputs = []
            for provided_input in provided_inputs:
                inputs.append(self.inputs_[provided_input])

            triton_client = grpcclient.InferenceServerClient("localhost:8001")
            results = triton_client.infer(model_name=self.model_name_, inputs=inputs)

            # OUTPU0 is always zero, OUTPUT1 = INPUT0
            output_data = results.as_numpy("OUTPUT0")
            expected = np.zeros(shape=(1, 1), dtype=np.float32)
            self.assertTrue(
                np.array_equal(output_data, expected),
                "{}, {}, expected: {}, got {}".format(
                    self.model_name_, "OUTPUT0", expected, output_data
                ),
            )

            expected = self.input_data_["INPUT0"]
            output_data = results.as_numpy("OUTPUT1")
            self.assertTrue(
                np.array_equal(output_data, expected),
                "{}, {}, expected: {}, got {}".format(
                    self.model_name_, "OUTPUT1", expected, output_data
                ),
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_ensemble_optional_connecting_tensor(self):
        # The ensemble is a special case of pipelining models with optional
        # inputs, where the request will only produce a subset of inputs
        # for the second model while the ensemble graph connects all inputs of
        # the second model (which is valid because the not-provided inputs
        # are marked optional). See 'config.pbtxt' for detail.
        self.model_name_ = "optional_connecting_tensor"

        # Provide all inputs, send requests that don't form preferred batch
        # so all requests should be returned after the queue delay
        try:
            provided_inputs = ("INPUT0",)
            inputs = []
            outputs = []
            for provided_input in provided_inputs:
                inputs.append(self.inputs_[provided_input])
                outputs.append(self.outputs_[provided_input])

            triton_client = grpcclient.InferenceServerClient("localhost:8001")
            results = triton_client.infer(
                model_name=self.model_name_, inputs=inputs, outputs=outputs
            )

            expected = self.input_data_["INPUT0"]
            output_data = results.as_numpy("OUTPUT0")
            self.assertTrue(
                np.array_equal(output_data, expected),
                "{}, {}, expected: {}, got {}".format(
                    self.model_name_, "OUTPUT0", expected, output_data
                ),
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == "__main__":
    unittest.main()
