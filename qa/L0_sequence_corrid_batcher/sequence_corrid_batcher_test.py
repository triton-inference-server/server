# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import time
import threading
import unittest
import numpy as np
import test_util as tu
import sequence_util as su

_test_system_shared_memory = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
_test_cuda_shared_memory = bool(
    int(os.environ.get('TEST_CUDA_SHARED_MEMORY', 0)))

_no_batching = (int(os.environ['NO_BATCHING']) == 1)
_model_instances = int(os.environ['MODEL_INSTANCES'])

if _no_batching:
    _trials = ("savedmodel_nobatch", "graphdef_nobatch", "plan_nobatch",
               "onnx_nobatch")
else:
    _trials = ("savedmodel", "graphdef", "plan", "onnx")

_protocols = ("http", "grpc")
_max_sequence_idle_ms = 5000


class SequenceCorrIDBatcherTest(su.SequenceBatcherTestUtil):

    def get_datatype(self, trial):
        return np.int32

    def get_expected_result(self,
                            expected_result,
                            corrid,
                            value,
                            trial,
                            flag_str=None):
        # Adjust the expected_result for models that
        # couldn't implement the full accumulator. See
        # qa/common/gen_qa_dyna_sequence_models.py for more
        # information.
        if ((("nobatch" not in trial) and ("custom" not in trial)) or \
            ("graphdef" in trial) or ("plan" in trial) or \
            ("onnx" in trial)) or ("libtorch" in trial):
            expected_result = value
            if flag_str is not None:
                if "start" in flag_str:
                    expected_result += 1
                if "end" in flag_str:
                    expected_result += corrid
        return expected_result

    def test_skip_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four sequences in parallel where
        # two sequences have shorter length so that padding must be
        # applied correctly for the longer sequences.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 3),
                                                                      dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions(
                (11, 12, 13, 14), dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113, 1114), dtype, 3)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12)
                self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                              os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

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
                            self.get_expected_result(4 + corrids[0], corrids[0],
                                                     3, trial, "end"),
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
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
                            (("start", 11, None), (None, 12, None),
                             (None, 13, None), ("end", 14, None)),
                            self.get_expected_result(50 + corrids[1],
                                                     corrids[1], 14, trial,
                                                     "end"),
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
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
                            self.get_expected_result(224 + corrids[2],
                                                     corrids[2], 113, trial,
                                                     "end"),
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
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
                            (("start", 1111, None), (None, 1112, None),
                             (None, 1113, None), ("end", 1114, None)),
                            self.get_expected_result(4450 + corrids[3],
                                                     corrids[3], 1114, trial,
                                                     "end"),
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

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


if __name__ == '__main__':
    unittest.main()
