#!/usr/bin/env python3

# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import threading
import time
import unittest

import numpy as np
import sequence_util as su
import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

_test_system_shared_memory = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))
_test_cuda_shared_memory = bool(int(os.environ.get("TEST_CUDA_SHARED_MEMORY", 0)))

_no_batching = int(os.environ["NO_BATCHING"]) == 1
_model_instances = int(os.environ["MODEL_INSTANCES"])

if _no_batching:
    _trials = ("savedmodel_nobatch", "graphdef_nobatch", "plan_nobatch", "onnx_nobatch")
else:
    _trials = ("savedmodel", "graphdef", "plan", "onnx")

_protocols = ("http", "grpc")
_max_sequence_idle_ms = 5000


class SequenceCorrIDBatcherTest(su.SequenceBatcherTestUtil):
    def get_datatype(self, trial):
        return np.int32

    def get_expected_result(self, expected_result, corrid, value, trial, flag_str=None):
        # Adjust the expected_result for models that
        # could not implement the full accumulator. See
        # qa/common/gen_qa_dyna_sequence_models.py for more
        # information.
        if (
            (("nobatch" not in trial) and ("custom" not in trial))
            or ("graphdef" in trial)
            or ("plan" in trial)
            or ("onnx" in trial)
        ) or ("libtorch" in trial):
            expected_result = value
            if flag_str is not None:
                if "start" in flag_str:
                    expected_result += 1
                if "end" in flag_str:
                    expected_result += corrid
        return expected_result

    def data_type_to_string(self, dtype):
        if dtype == "TYPE_STRING":
            return "BYTES"
        else:
            return dtype.replace("TYPE_", "")

    def test_skip_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four sequences in parallel where
        # two sequences have shorter length so that padding must be
        # applied correctly for the longer sequences.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 3), dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions(
                (11, 12, 13, 14), dtype, 1
            )
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 113), dtype, 2
            )
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113, 1114), dtype, 3
            )
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12)
                self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                )

                corrids = [1001, 1002, 1003, 1004]
                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[0],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (("start", 1, None), ("end", 3, None)),
                            self.get_expected_result(
                                4 + corrids[0], corrids[0], 3, trial, "end"
                            ),
                            precreated_shm0_handles,
                        ),
                        kwargs={"sequence_name": "{}".format(self._testMethodName)},
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[1],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start", 11, None),
                                (None, 12, None),
                                (None, 13, None),
                                ("end", 14, None),
                            ),
                            self.get_expected_result(
                                50 + corrids[1], corrids[1], 14, trial, "end"
                            ),
                            precreated_shm1_handles,
                        ),
                        kwargs={"sequence_name": "{}".format(self._testMethodName)},
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[2],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (("start", 111, None), ("end", 113, None)),
                            self.get_expected_result(
                                224 + corrids[2], corrids[2], 113, trial, "end"
                            ),
                            precreated_shm2_handles,
                        ),
                        kwargs={"sequence_name": "{}".format(self._testMethodName)},
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[3],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start", 1111, None),
                                (None, 1112, None),
                                (None, 1113, None),
                                ("end", 1114, None),
                            ),
                            self.get_expected_result(
                                4450 + corrids[3], corrids[3], 1114, trial, "end"
                            ),
                            precreated_shm3_handles,
                        ),
                        kwargs={"sequence_name": "{}".format(self._testMethodName)},
                    )
                )

                threads[1].start()
                threads[3].start()
                time.sleep(1)
                threads[0].start()
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                if _model_instances == 1:
                    self.check_status(model_name, {4: 4}, 12, 12)
                elif _model_instances == 2:
                    self.check_status(model_name, {2: 8}, 12, 12)
                elif _model_instances == 4:
                    self.check_status(model_name, {1: 12}, 12, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)

    def test_corrid_data_type(self):
        model_name = "add_sub"
        expected_corrid_dtype = os.environ["TRITONSERVER_CORRID_DATA_TYPE"]

        for corrid, corrid_dtype in [("corrid", "TYPE_STRING"), (123, "TYPE_UINT64")]:
            # Check if the corrid data type matches the expected corrid data type specified in the model config
            dtypes_match = True
            if (corrid_dtype == "TYPE_STRING") and (
                expected_corrid_dtype != "TYPE_STRING"
            ):
                dtypes_match = False
            elif (corrid_dtype == "TYPE_UINT64") and (
                expected_corrid_dtype
                not in ["TYPE_UINT32", "TYPE_INT32", "TYPE_UINT64", "TYPE_INT64"]
            ):
                dtypes_match = False

            with httpclient.InferenceServerClient("localhost:8000") as client:
                input0_data = np.random.rand(16).astype(np.float32)
                input1_data = np.random.rand(16).astype(np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0",
                        input0_data.shape,
                        np_to_triton_dtype(input0_data.dtype),
                    ),
                    httpclient.InferInput(
                        "INPUT1",
                        input1_data.shape,
                        np_to_triton_dtype(input1_data.dtype),
                    ),
                ]

                inputs[0].set_data_from_numpy(input0_data)
                inputs[1].set_data_from_numpy(input1_data)

                if not dtypes_match:
                    with self.assertRaises(InferenceServerException) as e:
                        client.infer(
                            model_name,
                            inputs,
                            sequence_id=corrid,
                            sequence_start=True,
                            sequence_end=False,
                        )
                    err_str = str(e.exception)
                    self.assertIn(
                        f"sequence batching control 'CORRID' data-type is '{self.data_type_to_string(corrid_dtype)}', but model '{model_name}' expects '{self.data_type_to_string(expected_corrid_dtype)}'",
                        err_str,
                    )
                else:
                    response = client.infer(
                        model_name,
                        inputs,
                        sequence_id=corrid,
                        sequence_start=True,
                        sequence_end=False,
                    )
                    response.get_response()
                    output0_data = response.as_numpy("OUTPUT0")
                    output1_data = response.as_numpy("OUTPUT1")

                    self.assertTrue(
                        np.allclose(input0_data + input1_data, output0_data),
                        "add_sub example error: incorrect sum",
                    )

                    self.assertTrue(
                        np.allclose(input0_data - input1_data, output1_data),
                        "add_sub example error: incorrect difference",
                    )


if __name__ == "__main__":
    unittest.main()
