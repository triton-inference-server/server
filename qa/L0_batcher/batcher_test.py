# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
import os
import time
import threading
import traceback
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

if os.environ['BATCHER_TYPE'] == "VARIABLE":
    _trials = ("savedmodel", "graphdef", "netdef", "custom", "libtorch", "onnx")
else:
    _trials = ("savedmodel", "graphdef", "plan", "netdef", "custom", "libtorch", "onnx")

_max_queue_delay_ms = 10000
_check_exception = None

class BatcherTest(unittest.TestCase):
    def setUp(self):
        global _check_exception
        _check_exception = None

    def check_deferred_exception(self):
        if _check_exception is not None:
            raise _check_exception

    def check_response(self, trial, bs, thresholds,
                       requested_outputs=("OUTPUT0", "OUTPUT1"), input_size=16,
                       use_shared_memory=False):
        global _check_exception
        try:
            start_ms = int(round(time.time() * 1000))

            if trial == "savedmodel" or trial == "graphdef" or trial == "netdef" \
                    or trial == "custom" or trial == "libtorch" or trial == "onnx":
                tensor_shape = (input_size,)
                iu.infer_exact(self, trial, tensor_shape, bs,
                               np.float32, np.float32, np.float32, swap=False,
                               model_version=1, outputs=requested_outputs,
                               use_grpc=False, skip_request_id_check=True,
                               use_streaming=False, use_shared_memory)
            elif trial == "plan":
                tensor_shape = (input_size,1,1)
                iu.infer_exact(self, trial, tensor_shape, bs,
                               np.float32, np.float32, np.float32, swap=False,
                               model_version=1, outputs=requested_outputs,
                               use_grpc=False, skip_request_id_check=True,
                               use_streaming=False, use_shared_memory)
            else:
                self.assertFalse(True, "unknown trial type: " + trial)

            end_ms = int(round(time.time() * 1000))

            lt_ms = thresholds[0]
            gt_ms = thresholds[1]
            if lt_ms is not None:
                self.assertTrue((end_ms - start_ms) < lt_ms,
                                "expected less than " + str(lt_ms) +
                                "ms response time, got " + str(end_ms - start_ms) + " ms")
            if gt_ms is not None:
                self.assertTrue((end_ms - start_ms) > gt_ms,
                                "expected greater than " + str(gt_ms) +
                                "ms response time, got " + str(end_ms - start_ms) + " ms")
        except Exception as ex:
            _check_exception = ex

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
        self.assertEqual(bconfig.max_queue_delay_microseconds, _max_queue_delay_ms * 1000) # 10 secs

    def check_status(self, url, protocol, model_name, static_bs, exec_cnt, infer_cnt):
        ctx = ServerStatusContext(url, protocol, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        vs = ss.model_status[model_name].version_status
        self.assertEqual(len(vs), 1)
        self.assertTrue(1 in vs, "expected status for version 1")
        infer = vs[1].infer_stats
        self.assertEqual(len(infer), len(static_bs),
                         "expected batch-sizes (" + ",".join(str(b) for b in static_bs) +
                         "), got " + str(vs[1]))
        for b in static_bs:
            self.assertTrue(b in infer,
                            "expected batch-size " + str(b) + ", got " + str(vs[1]))
        self.assertEqual(vs[1].model_execution_count, exec_cnt,
                        "expected model-execution-count " + str(exec_cnt) + ", got " +
                        str(vs[1].model_execution_count))
        self.assertEqual(vs[1].model_inference_count, infer_cnt,
                        "expected model-inference-count " + str(infer_cnt) + ", got " +
                        str(vs[1].model_inference_count))

    def test_static_batch_preferred(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(trial, 2, (3000, None))
                self.check_response(trial, 6, (3000, None))
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (2,6), 2, 8)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_preferred_shm(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(trial, 2, (3000, None), use_shared_memory=True)
                self.check_response(trial, 6, (3000, None), use_shared_memory=True)
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (2,6), 2, 8)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_lt_any_preferred(self):
        # Send a request with a static batch size < any preferred
        # size. This should cause the response to be delayed by the
        # max batch queue delay
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(trial, 1, (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 1)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_not_preferred(self):
        # Send a request with a static batch size in between preferred
        # sizes. This should cause the response to be delayed by the
        # max batch queue delay
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(trial, 3, (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (3,), 1, 3)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_gt_max_preferred(self):
        # Send a request with a static batch size > maximum preferred
        # size. This should cause the request to be issued immediately
        # (even though the maximum batching queue delay is very high).
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(trial, 7, (3000, None))
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (7,), 1, 7)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_different_shape(self):
        # Send two requests with sum of static batch sizes ==
        # preferred size, but with different shapes (using model with
        # variable-size tensors). This should cause the requests to
        # not be batched. The first response will come back
        # immediately and the second delayed by the max batch queue
        # delay
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'input_size': 16}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms)),
                                                kwargs={'input_size': 8}))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 2, 2)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred(self):
        # Send two requests with total static batch size in between
        # preferred sizes. This should cause the first response to be
        # delayed by the max batch queue delay, and the second by max
        # delay (minus the difference in time that they arrived in the
        # queue)
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms - 2000))))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,3), 1, 4)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. This should cause the first
        # two requests to be immediately responded to and the third
        # response to be delayed by the max batch queue delay.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms)),
                                                kwargs={'input_size': 8}))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,3), 2, 5)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. This should cause the first
        # two requests to be immediately responded to. Send a forth
        # request with the same shape as the third that causes a
        # preferred size so that third and forth response are sent
        # immediately.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'input_size': 8}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 5, (3000, None)),
                                                kwargs={'input_size': 8}))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,3,5), 2, 10)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_gt_max_preferred(self):
        # Send two requests with first not having preferred size and
        # second being larger than max preferred size. Delay the
        # second request so that it arrives after the first is already
        # be processed by the dynamic batcher. This should cause both
        # responses to be returned immediately.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 7, (3000, None))))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (3, 7), 2, 10)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_sum_gt_max_preferred(self):
        # Send two requests with first not having preferred size and
        # second being smaller than max preferred size but the sum of
        # the requests being larger than max preferred size. Delay the
        # second request so that it arrives after the first is already
        # be processed by the dynamic batcher. This should cause first
        # response to be returned immediately but the second response,
        # since it alone is not greater than max preferred size, will
        # be delayed.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)
                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 4,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (3,4), 2, 7)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output0(self):
        # Send two requests where both ask for OUTPUT0. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT0",)}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT0",)}))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 2)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output1(self):
        # Send two requests where both ask for OUTPUT1. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT1",)}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT1",)}))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 2)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_outputs(self):
        # Send two requests where one request asks for one output and
        # the other request asks for the other output. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT0",)}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT1",)}))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 2)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_output_order(self):
        # Send two requests that ask for both outputs, but in a
        # different order. They should be batched and get the correct
        # response even though they use different order.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT0","OUTPUT1")}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'requested_outputs': ("OUTPUT1","OUTPUT0")}))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 2)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_sum_gt_max_preferred(self):
        # Send two requests with first not having preferred size and
        # second being smaller than max preferred size but the sum of
        # the requests being larger than max preferred size. Use
        # TRTSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. This should cause first response to be returned
        # immediately but the second response, since it alone is not
        # greater than max preferred size, will be delayed.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                # Need scheduler to wait for queue to contain 2 requests
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 2)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 4,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (3,4), 2, 7)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. Use
        # TRTSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. This should cause the first two requests to be
        # immediately responded to. Send a forth request with the same
        # shape as the third that causes a preferred size so that
        # third and forth response are sent immediately.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                # Need scheduler to wait for queue to contain 4 requests
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 4)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 3, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None)),
                                                kwargs={'input_size': 8}))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 5, (3000, None)),
                                                kwargs={'input_size': 8}))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,3,5), 2, 10)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_biggest_preferred(self):
        # Send multiple requests that sum to multiple preferred sizes
        # and make sure the largest preferred size if used for the
        # batch. Use TRTSERVER_DELAY_SCHEDULER in the environment so
        # that requests can be queued up before scheduler starts
        # servicing.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                # Need scheduler to wait for queue to contain 6 request
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 6)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 1, 6)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_best_preferred(self):
        # Send multiple requests where the initial ones sum to a
        # preferred size and then extra request goes beyond that. The
        # initial requests should be handled immediately at the
        # preferred batch size and then the other one after
        # timeout. Use TRTSERVER_DELAY_SCHEDULER in the environment so
        # that requests can be queued up before scheduler starts
        # servicing.
        for trial in _trials:
            try:
                url = "localhost:8000"
                protocol = ProtocolType.HTTP
                model_name = tu.get_model_name(trial, np.float32, np.float32, np.float32)

                self.check_setup(url, protocol, model_name)

                # Need scheduler to wait for queue to contain 3 requests
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1, (3000, None))))
                threads.append(threading.Thread(target=self.check_response,
                                                args=(trial, 1,
                                                      (_max_queue_delay_ms * 1.5, _max_queue_delay_ms))))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(url, protocol, model_name, (1,), 2, 3)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
