# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

from builtins import str
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

NO_BATCHING = (int(os.environ.get('NO_BATCHING', 0)) == 1)
BACKENDS = os.environ.get(
    'BACKENDS', "graphdef savedmodel libtorch onnx plan custom custom_string")
IMPLICIT_STATE = (int(os.environ['IMPLICIT_STATE']) == 1)

_trials = BACKENDS.split(' ')
for backend in BACKENDS.split(" "):
    if NO_BATCHING:
        if (backend != 'custom') and (backend != 'custom_string'):
            _trials += (backend + "_nobatch",)

_ragged_batch_supported_trials = []
if 'custom' in BACKENDS.split(' '):
    _ragged_batch_supported_trials.append('custom')

_protocols = ("http", "grpc")
_max_sequence_idle_ms = 5000


class DynaSequenceBatcherTest(su.SequenceBatcherTestUtil):

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
            ("graphdef" in trial) or ("plan" in trial) or ("onnx" in trial) or \
            ("libtorch" in trial)):
            expected_result = value
            if flag_str is not None:
                if "start" in flag_str:
                    expected_result += 1
                if "end" in flag_str:
                    if isinstance(corrid, str):
                        expected_result += int(corrid)
                    else:
                        expected_result += corrid
        return expected_result

    def get_expected_result_implicit(self,
                                     expected_result,
                                     corrid,
                                     value,
                                     trial,
                                     flag_str=None):
        return expected_result

    def test_simple_sequence(self):
        # Send one sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                self.clear_deferred_exceptions()
                try:
                    dtype = self.get_datatype(trial)
                    model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                    self.check_setup(model_name)
                    self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                     os.environ)

                    if "string" in trial:
                        corrid = '52'
                    else:
                        corrid = 52

                    expected_result = self.get_expected_result(
                        45 + int(corrid), corrid, 9, trial, "end"
                    ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                        45, corrid, 9, trial, "end")

                    self.check_sequence(
                        trial,
                        model_name,
                        dtype,
                        corrid,
                        (4000, None),
                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                        (("start", 1, None, None), (None, 2, None, None),
                         (None, 3, None, None), (None, 4, None, None),
                         (None, 5, None, None), (None, 6, None, None),
                         (None, 7, None, None), (None, 8, None, None),
                         ("end", 9, None, None)),
                        expected_result,
                        protocol,
                        sequence_name="{}_{}".format(self._testMethodName,
                                                     protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, {1: 9 * (idx + 1)},
                                      9 * (idx + 1), 9 * (idx + 1))
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_length1_sequence(self):
        # Send a length-1 sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                self.clear_deferred_exceptions()
                try:
                    dtype = self.get_datatype(trial)
                    model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                    self.check_setup(model_name)
                    self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                     os.environ)

                    if "string" in trial:
                        corrid = '99'
                    else:
                        corrid = 99

                    expected_result = self.get_expected_result(
                        42 + int(corrid), corrid, 42, trial, "start,end"
                    ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                        42, corrid, 42, trial, "start,end")

                    self.check_sequence(
                        trial,
                        model_name,
                        dtype,
                        corrid,
                        (4000, None),
                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                        (
                            ("start,end", 42, None, None),),
                        expected_result,
                        protocol,
                        sequence_name="{}_{}".format(self._testMethodName,
                                                     protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, {1: (idx + 1)}, (idx + 1),
                                      (idx + 1))
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def _multi_sequence_impl(self, trials, expected_batch_exec,
                             expected_exec_cnt, sleep_secs, tensor_shapes):
        for trial in trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions(
                (1, 3), dtype, 0, tensor_shape=(tensor_shapes[0],))
            precreated_shm1_handles = self.precreate_register_regions(
                (11, 12, 13), dtype, 1, tensor_shape=(tensor_shapes[1],))
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 112, 113), dtype, 2, tensor_shape=(tensor_shapes[2],))
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113), dtype, 3, tensor_shape=(tensor_shapes[3],))
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                 os.environ)

                if "string" in trial:
                    corrids = ['1001', '1002', '1003', '1004']
                else:
                    corrids = [1001, 1002, 1003, 1004]

                expected_result = self.get_expected_result(
                    4 * tensor_shapes[0] +
                    int(corrids[0]), corrids[0], 3, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    4, corrids[0], 3, trial, "end")

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
                            expected_result,
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name':
                                "{}_{}".format(self._testMethodName,
                                               corrids[0]),
                            'tensor_shape': (tensor_shapes[0],)
                        }))

                expected_result = self.get_expected_result(
                    36 * tensor_shapes[1] +
                    int(corrids[1]), corrids[1], 13, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    36, corrids[1], 13, trial, "end")
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
                            (("start", 11, None), (None, 12, None), ("end", 13,
                                                                     None)),
                            expected_result,
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name':
                                "{}_{}".format(self._testMethodName,
                                               corrids[1]),
                            'tensor_shape': (tensor_shapes[1],)
                        }))

                expected_result = self.get_expected_result(
                    336 * tensor_shapes[2] +
                    int(corrids[2]), corrids[2], 113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    336, corrids[2], 113, trial, "end")
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
                            (("start", 111, None), (None, 112, None),
                             ("end", 113, None)),
                            expected_result,
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name':
                                "{}_{}".format(self._testMethodName,
                                               corrids[2]),
                            'tensor_shape': (tensor_shapes[2],)
                        }))
                expected_result = self.get_expected_result(
                    3336 * tensor_shapes[3] +
                    int(corrids[3]), corrids[3], 1113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    3336, corrids[3], 1113, trial, "end")
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
                             ("end", 1113, None)),
                            expected_result,
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name':
                                "{}_{}".format(self._testMethodName,
                                               corrids[3]),
                            'tensor_shape': (tensor_shapes[3],)
                        }))

                for t in threads:
                    t.start()
                    if sleep_secs > 0:
                        time.sleep(sleep_secs)
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, expected_batch_exec,
                                  expected_exec_cnt, 11)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)

    def test_multi_sequence(self):
        # Send four sequences in series and make sure they get
        # batched correctly.
        self._multi_sequence_impl(_trials, {4: 2, 3: 1}, 3, 1, (1, 1, 1, 1))

    def test_multi_parallel_sequence(self):
        # Send four sequences in parallel and make sure they get
        # batched correctly.
        self._multi_sequence_impl(_trials, {4: 2, 3: 1}, 3, 0, (1, 1, 1, 1))

    def test_multi_sequence_different_shape(self):
        # Send four sequences in parallel where the requests in each
        # sequence have different shape. Sequences should not be
        # batched due to input tensor size differences.
        self._multi_sequence_impl(_ragged_batch_supported_trials, {1: 11}, 11,
                                  0, (4, 3, 1, 2))

    def test_multi_sequence_different_shape_allow_ragged(self):
        # Send four sequences in parallel where the requests in each
        # sequence have different shape. Input is marked as allowing
        # ragged and so sequences should be batched even with input
        # tensor size differences.
        self._multi_sequence_impl(_ragged_batch_supported_trials, {
            4: 2,
            3: 1
        }, 3, 1, (4, 3, 1, 2))

    def test_backlog(self):
        # Send 5 equal-length sequences in parallel and make sure they
        # get completely batched into batch-size 4 inferences plus the
        # 5th should go in the backlog and then get handled once there
        # is a free slot.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 2, 3),
                                                                      dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions(
                (11, 12, 13), dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 112, 113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113), dtype, 3)
            precreated_shm4_handles = self.precreate_register_regions(
                (11111, 11112, 11113), dtype, 4)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                 os.environ)

                if "string" in trial:
                    corrids = ['1001', '1002', '1003', '1004', '1005']
                else:
                    corrids = [1001, 1002, 1003, 1004, 1005]

                expected_result = self.get_expected_result(
                    6 + int(corrids[0]), corrids[0], 3, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    6, corrids[0], 3, trial, "end")

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
                            (("start", 1, None), (None, 2, None), ("end", 3,
                                                                   None)),
                            expected_result,
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                expected_result = self.get_expected_result(
                    36 + int(corrids[1]), corrids[1], 13, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    36, corrids[1], 13, trial, "end")
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
                            (("start", 11, None), (None, 12, None), ("end", 13,
                                                                     None)),
                            expected_result,
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                expected_result = self.get_expected_result(
                    336 + int(corrids[2]), corrids[2], 113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    336, corrids[2], 113, trial, "end")
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
                            (("start", 111, None), (None, 112, None),
                             ("end", 113, None)),
                            expected_result,
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                expected_result = self.get_expected_result(
                    3336 + int(corrids[3]), corrids[3], 1113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    3336, corrids[3], 1113, trial, "end")
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
                             ("end", 1113, None)),
                            expected_result,
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                expected_result = self.get_expected_result(
                    33336 + int(corrids[4]), corrids[4], 11113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    33336, corrids[4], 11113, trial, "end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[4],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (("start", 11111, None), (None, 11112, None),
                             ("end", 11113, None)),
                            expected_result,
                            precreated_shm4_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 3, 1: 3}, 6, 15)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)
                    self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_fill(self):
        # Send 4 sequences in parallel, two of which are shorter. Send
        # 2 additional sequences that should go into backlog but
        # should immediately fill into the short sequences.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 2, 3),
                                                                      dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions((11, 13),
                                                                      dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113), dtype, 3)
            precreated_shm4_handles = self.precreate_register_regions((11111,),
                                                                      dtype, 4)
            precreated_shm5_handles = self.precreate_register_regions((22222,),
                                                                      dtype, 5)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                 os.environ)
                if "string" in trial:
                    corrids = ['1001', '1002', '1003', '1004', '1005', '1006']
                else:
                    corrids = [1001, 1002, 1003, 1004, 1005, 1006]
                threads = []

                expected_result = self.get_expected_result(
                    6 + int(corrids[0]), corrids[0], 3, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    6, corrids[0], 3, trial, "end")
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
                            (("start", 1, None), (None, 2, None), ("end", 3,
                                                                   None)),
                            expected_result,
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    24 + int(corrids[1]), corrids[1], 13, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    24, corrids[1], 13, trial, "end")
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
                            (("start", 11, None), ("end", 13, None)),
                            expected_result,
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    224 + int(corrids[2]), corrids[2], 113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    224, corrids[2], 113, trial, "end")
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
                            expected_result,
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    3336 + int(corrids[3]), corrids[3], 1113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    3336, corrids[3], 1113, trial, "end")
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
                            (("start", 1111, None), (None, 1112, 3000),
                             ("end", 1113, None)),
                            expected_result,
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    11111 +
                    int(corrids[4]), corrids[4], 11111, trial, "start,end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    11111, corrids[4], 11111, trial, "start,end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[4],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start,end", 11111, None),),
                            expected_result,
                            precreated_shm4_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    22222 +
                    int(corrids[5]), corrids[5], 22222, trial, "start,end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    22222, corrids[5], 22222, trial, "start,end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[5],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start,end", 22222, None),),
                            expected_result,
                            precreated_shm5_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                threads[5].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 3}, 3, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)
                    self.cleanup_shm_regions(precreated_shm4_handles)
                    self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_fill_no_end(self):
        # Send 4 sequences in parallel, two of which are shorter. Send
        # 2 additional sequences that should go into backlog but
        # should immediately fill into the short sequences. One of
        # those sequences is filled before it gets its end request.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 2, 3),
                                                                      dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions((11, 13),
                                                                      dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1113), dtype, 3)
            precreated_shm4_handles = self.precreate_register_regions((11111,),
                                                                      dtype, 4)
            precreated_shm5_handles = self.precreate_register_regions(
                (22222, 22223, 22224), dtype, 5)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                 os.environ)

                if "string" in trial:
                    corrids = ['1001', '1002', '1003', '1004', '1005', '1006']
                else:
                    corrids = [1001, 1002, 1003, 1004, 1005, 1006]
                threads = []
                expected_result = self.get_expected_result(
                    6 + int(corrids[0]), corrids[0], 3, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    6, corrids[0], 3, trial, "end")
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
                            (("start", 1, None), (None, 2, None), ("end", 3,
                                                                   None)),
                            expected_result,
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    24 + int(corrids[1]), corrids[1], 13, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    24, corrids[1], 13, trial, "end")
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
                            (("start", 11, None), ("end", 13, None)),
                            expected_result,
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    224 + int(corrids[2]), corrids[2], 113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    224, corrids[2], 113, trial, "end")
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
                            expected_result,
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    3336 + int(corrids[3]), corrids[3], 1113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    3336, corrids[3], 1113, trial, "end")
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
                            (("start", 1111, None), (None, 1112, 3000),
                             ("end", 1113, None)),
                            expected_result,
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    11111 +
                    int(corrids[4]), corrids[4], 11111, trial, "start,end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    11111, corrids[4], 11111, trial, "start,end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[4],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start,end", 11111, None),),
                            expected_result,
                            precreated_shm4_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    66669 + int(corrids[5]), corrids[5], 22224, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    66669, corrids[5], 22224, trial, "end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[5],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (
                                ("start", 22222, None),
                                (None, 22223, None),
                                ("end", 22224, 2000),
                            ),
                            expected_result,
                            precreated_shm5_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                threads[5].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                # Expecting the requests of the same sequence to be in the same
                # slot, so the execution for thelast long sequence will be
                # padded to a batch.
                self.check_status(model_name, {4: 3, 1: 2}, 5, 14)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)
                    self.cleanup_shm_regions(precreated_shm4_handles)
                    self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_sequence_timeout(self):
        # Send 4 sequences in parallel and make sure they get
        # completely batched into batch-size 4 inferences. One of the
        # sequences has a long delay that causes it to timeout and
        # that allows a 5th sequence to come out of the backlog and
        # finish. The timed-out sequence will then send the delayed
        # inference but it will appear as a new sequence and so fail
        # because it doesn't have the START flag.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1, 3),
                                                                      dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions(
                (11, 12, 12, 13), dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions(
                (111, 112, 112, 113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions(
                (1111, 1112, 1112, 1113), dtype, 3)
            precreated_shm4_handles = self.precreate_register_regions(
                (11111, 11113), dtype, 4)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER",
                                 os.environ)

                if "string" in trial:
                    corrids = ['1001', '1002', '1003', '1004', '1005']
                else:
                    corrids = [1001, 1002, 1003, 1004, 1005]
                threads = []
                expected_result = self.get_expected_result(
                    4 + int(corrids[0]), corrids[0], 3, trial, None
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    4, corrids[0], 3, trial, None)
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
                            (("start", 1, None),
                             (None, 3, _max_sequence_idle_ms + 1000)),
                            expected_result,
                            precreated_shm0_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    48 + int(corrids[1]), corrids[1], 13, trial, None
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    48, corrids[1], 13, trial, None)
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
                            (("start", 11, None), (None, 12,
                                                   _max_sequence_idle_ms / 2),
                             (None, 12, _max_sequence_idle_ms / 2),
                             ("end", 13, _max_sequence_idle_ms / 2)),
                            expected_result,
                            precreated_shm1_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    448 + int(corrids[2]), corrids[2], 113, trial, None
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    448, corrids[2], 113, trial, None)
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
                            (("start", 111, None), (None, 112,
                                                    _max_sequence_idle_ms / 2),
                             (None, 112, _max_sequence_idle_ms / 2),
                             ("end", 113, _max_sequence_idle_ms / 2)),
                            expected_result,
                            precreated_shm2_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    4448 + int(corrids[3]), corrids[3], 1113, trial, None
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    4448, corrids[3], 1113, trial, None)
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
                            (("start", 1111, None), (None, 1112,
                                                     _max_sequence_idle_ms / 2),
                             (None, 1112, _max_sequence_idle_ms / 2),
                             ("end", 1113, _max_sequence_idle_ms / 2)),
                            expected_result,
                            precreated_shm3_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))
                expected_result = self.get_expected_result(
                    22224 + int(corrids[4]), corrids[4], 11113, trial, "end"
                ) if not IMPLICIT_STATE else self.get_expected_result_implicit(
                    22224, corrids[4], 11113, trial, "end")
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_async,
                        args=(
                            trial,
                            model_name,
                            dtype,
                            corrids[4],
                            (None, None),
                            # (flag_str, value, pre_delay_ms)
                            (("start", 11111, None), ("end", 11113, None)),
                            expected_result,
                            precreated_shm4_handles),
                        kwargs={
                            'sequence_name': "{}".format(self._testMethodName)
                        }))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                for t in threads:
                    t.join()

                self.check_deferred_exception()
                self.assertTrue(False, "expected error")
            except Exception as ex:
                self.assertTrue(ex.message().startswith(
                    str("inference request for sequence 1001 to " +
                        "model '{}' must specify the START flag on the first " +
                        "request of the sequence").format(model_name)))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)
                    self.cleanup_shm_regions(precreated_shm4_handles)


if __name__ == '__main__':
    unittest.main()
