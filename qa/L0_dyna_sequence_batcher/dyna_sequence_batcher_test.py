# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from builtins import str
from future.utils import iteritems
import os
import time
import threading
import traceback
import unittest
import numpy as np
import test_util as tu
import sequence_util as su
from tensorrtserver.api import *

_test_system_shared_memory = bool(int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
_test_cuda_shared_memory = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY', 0)))

_trials = ("custom", "savedmodel", "graphdef", "netdef", "plan", "onnx", "libtorch")
_protocols = ("http", "grpc")


class DynaSequenceBatcherTest(su.SequenceBatcherTestUtil):
    def get_datatype(self, trial):
        return np.int32

    def get_expected_result(self, expected_result, corrid, value, trial, flag_str=None):
        # Adjust the expected_result for models that
        # couldn't implement the full accumulator. See
        # qa/common/gen_qa_dyna_sequence_models.py for more
        # information.
        if "custom" not in trial:
            expected_result = value
            if flag_str is not None:
                if "start" in flag_str:
                    expected_result += 1
                if "end" in flag_str:
                    expected_result += corrid
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
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    corrid = 52
                    self.check_sequence(trial, model_name, dtype, corrid,
                                        (4000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         (None, 2, None, None),
                                         (None, 3, None, None),
                                         (None, 4, None, None),
                                         (None, 5, None, None),
                                         (None, 6, None, None),
                                         (None, 7, None, None),
                                         (None, 8, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(45 + corrid, corrid, 9, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), 9 * (idx + 1), 9 * (idx + 1))
                except InferenceServerException as ex:
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
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    corrid = 99
                    self.check_sequence(trial, model_name, dtype, corrid,
                                        (4000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start,end", 42, None, None),),
                                        self.get_expected_result(42 + corrid, corrid, 42,
                                                                 trial, "start,end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), (idx + 1), (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_sequence(self):
        # Send four sequences in parallel and make sure they get
        # completely batched into batch-size 4 inferences.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1,2,3), dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions((11,12,13), dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions((111,112,113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions((1111,1112,1113), dtype, 3)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)
                protocol = "streaming"

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for all sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 11)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                corrids = [ 1001, 1002, 1003, 1004 ]
                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[0],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           ("end", 3, None)),
                          self.get_expected_result(4 + corrids[0], corrids[0], 3, trial, "end"),
                          protocol, precreated_shm0_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[0])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[1],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(36 + corrids[1], corrids[1], 13, trial, "end"),
                          protocol, precreated_shm1_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[1])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[2],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(336 + corrids[2], corrids[2], 113, trial, "end"),
                          protocol, precreated_shm2_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[2])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[3],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336 + corrids[3], corrids[3], 1113, trial, "end"),
                          protocol, precreated_shm3_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[3])}))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), 3, 11)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)


if __name__ == '__main__':
    unittest.main()
