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
from future.utils import iteritems
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import time
import threading
import traceback

from tensorrtserver.api import *
import os

TEST_SYSTEM_SHARED_MEMORY = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY',
                                                  0)))

_max_queue_delay_ms = 10000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class InferShapeTensorTest(unittest.TestCase):

    def setUp(self):
        global _deferred_exceptions
        _deferred_exceptions = []

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        # Just raise one of the exceptions...
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_response(self,
                       bs,
                       thresholds,
                       shape_values,
                       dummy_input_shapes,
                       shm_region_names=None,
                       precreated_shm_regions=None,
                       shm_suffix=""):
        try:
            start_ms = int(round(time.time() * 1000))

            iu.infer_shape_tensor(self,
                                  'plan',
                                  bs,
                                  np.float32,
                                  shape_values,
                                  dummy_input_shapes,
                                  use_grpc=False,
                                  use_streaming=False,
                                  shm_suffix=shm_suffix)

            end_ms = int(round(time.time() * 1000))

            lt_ms = thresholds[0]
            gt_ms = thresholds[1]
            if lt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) < lt_ms,
                    "expected less than " + str(lt_ms) +
                    "ms response time, got " + str(end_ms - start_ms) + " ms")
            if gt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) > gt_ms,
                    "expected greater than " + str(gt_ms) +
                    "ms response time, got " + str(end_ms - start_ms) + " ms")
        except Exception as ex:
            self.add_deferred_exception(ex)

    def check_setup(self, url, protocol, model_name):
        # Make sure test.sh set up the correct batcher settings
        ctx = ServerStatusContext(url, protocol, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        bconfig = ss.model_status[model_name].config.dynamic_batching
        self.assertTrue(2 in bconfig.preferred_batch_size)
        self.assertTrue(6 in bconfig.preferred_batch_size)
        self.assertEqual(bconfig.max_queue_delay_microseconds,
                         _max_queue_delay_ms * 1000)  # 10 secs

    def check_status(self, url, protocol, model_name, static_bs, exec_cnt,
                     infer_cnt):
        ctx = ServerStatusContext(url, protocol, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        vs = ss.model_status[model_name].version_status
        self.assertEqual(len(vs), 1)
        self.assertTrue(1 in vs, "expected status for version 1")
        infer = vs[1].infer_stats
        self.assertEqual(
            len(infer), len(static_bs), "expected batch-sizes (" +
            ",".join(str(b) for b in static_bs) + "), got " + str(vs[1]))
        for b in static_bs:
            self.assertTrue(
                b in infer,
                "expected batch-size " + str(b) + ", got " + str(vs[1]))
        self.assertEqual(
            vs[1].model_execution_count, exec_cnt,
            "expected model-execution-count " + str(exec_cnt) + ", got " +
            str(vs[1].model_execution_count))
        self.assertEqual(
            vs[1].model_inference_count, infer_cnt,
            "expected model-inference-count " + str(infer_cnt) + ", got " +
            str(vs[1].model_inference_count))

    def test_static_batch(self):
        iu.infer_shape_tensor(self, 'plan', 8, np.float32, [[32, 32]], [[4, 4]])
        iu.infer_shape_tensor(self, 'plan', 8, np.float32, [[4, 4]], [[32, 32]])
        iu.infer_shape_tensor(self, 'plan', 8, np.float32, [[4, 4]], [[4, 4]])

    def test_nobatch(self):
        iu.infer_shape_tensor(self, 'plan_nobatch', 1, np.float32, [[32, 32]],
                              [[4, 4]])
        iu.infer_shape_tensor(self, 'plan_nobatch', 1, np.float32, [[4, 4]],
                              [[32, 32]])
        iu.infer_shape_tensor(self, 'plan_nobatch', 1, np.float32, [[4, 4]],
                              [[4, 4]])

    def test_wrong_shape_values(self):
        over_shape_values = [[32, 33]]
        try:
            iu.infer_shape_tensor(self, 'plan', 8, np.float32,
                                  over_shape_values, [[4, 4]])
        except InferenceServerException as ex:
            self.assertEqual("inference:0", ex.server_id())
            self.assertTrue(
                "The shape value at index 2 is expected to be in range from 1 to 32, Got: 33"
                in ex.message())

    # Dynamic Batcher tests
    def test_multi_batch_different_shape_values(self):
        # Send two requests with sum of static batch sizes ==
        # preferred size, but with different shape values. This
        # should cause the requests to not be batched. The first
        # response will come back immediately and the second
        # delayed by the max batch queue delay
        try:
            url = "localhost:8000"
            protocol = ProtocolType.HTTP
            model_name = tu.get_zero_model_name("plan", 1, np.float32)
            self.check_setup(url, protocol, model_name)
            self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(3, (6000, None)),
                                 kwargs={
                                     'shape_values': [[2, 2]],
                                     'dummy_input_shapes': [[16, 16]],
                                     'shm_suffix': '{}'.format(len(threads))
                                 }))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(3, (_max_queue_delay_ms * 1.5,
                                           _max_queue_delay_ms)),
                                 kwargs={
                                     'shape_values': [[4, 4]],
                                     'dummy_input_shapes': [[16, 16]],
                                     'shm_suffix': '{}'.format(len(threads))
                                 }))
            threads[0].start()
            time.sleep(1)
            threads[1].start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(url, protocol, model_name, (3,), 2, 6)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_same_shape_values(self):
        # Send two requests with sum of static batch sizes ==
        # preferred size, but with identical shape values. This
        # should cause the requests to get batched. Both
        # responses should come back immediately.
        try:
            url = "localhost:8000"
            protocol = ProtocolType.HTTP
            model_name = tu.get_zero_model_name("plan", 1, np.float32)
            self.check_setup(url, protocol, model_name)
            self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(4, (6000, None)),
                                 kwargs={
                                     'shape_values': [[4, 4]],
                                     'dummy_input_shapes': [[16, 16]],
                                     'shm_suffix': '{}'.format(len(threads))
                                 }))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, (6000, None)),
                                 kwargs={
                                     'shape_values': [[4, 4]],
                                     'dummy_input_shapes': [[16, 16]],
                                     'shm_suffix': '{}'.format(len(threads))
                                 }))
            threads[0].start()
            time.sleep(1)
            threads[1].start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            self.check_status(url, protocol, model_name, (4, 2), 1, 6)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
