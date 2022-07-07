# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import time
import threading
import unittest
import numpy as np
import infer_util as iu
import test_util as tu

import tritonclient.grpc as grpcclient

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get('TRITONSERVER_IPADDR', 'localhost')

TEST_SYSTEM_SHARED_MEMORY = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY',
                                                  0)))

if TEST_SYSTEM_SHARED_MEMORY:
    import tritonclient.utils.shared_memory as shm
if TEST_CUDA_SHARED_MEMORY:
    import tritonclient.utils.cuda_shared_memory as cudashm

# Test with either GRPC of HTTP, but not both since when we check
# results we expect only one to run
USE_GRPC = (os.environ.get('USE_GRPC', 1) != "0")
USE_HTTP = (os.environ.get('USE_HTTP', 1) != "0")
if USE_GRPC and USE_HTTP:
    USE_GRPC = False
assert USE_GRPC or USE_HTTP, "USE_GRPC or USE_HTTP must be non-zero"

BACKENDS = os.environ.get('BACKENDS',
                          "graphdef savedmodel onnx libtorch plan python")

_trials = BACKENDS.split(" ")

_ragged_batch_supported_trials = ["custom"]
if "plan" in _trials:
    _ragged_batch_supported_trials.append("plan")
if "onnx" in _trials:
    _ragged_batch_supported_trials.append("onnx")

_max_queue_delay_ms = 10000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class BatcherTest(tu.TestResultCollector):

    def setUp(self):
        # The helper client for setup will be GRPC for simplicity.
        self.triton_client_ = grpcclient.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001")
        self.precreated_shm_regions_ = []
        global _deferred_exceptions
        _deferred_exceptions = []

    def tearDown(self):
        if TEST_SYSTEM_SHARED_MEMORY:
            self.triton_client_.unregister_system_shared_memory()
        if TEST_CUDA_SHARED_MEMORY:
            self.triton_client_.unregister_cuda_shared_memory()
        for precreated_shm_region in self.precreated_shm_regions_:
            if TEST_SYSTEM_SHARED_MEMORY:
                shm.destroy_shared_memory_region(precreated_shm_region)
            elif TEST_CUDA_SHARED_MEMORY:
                cudashm.destroy_shared_memory_region(precreated_shm_region)
        super().tearDown()

    # FIXME why only used for outputs
    def create_advance(self, shm_regions=None):
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            precreated_shm_regions = []
            if shm_regions is None:
                shm_regions = ['output0', 'output1']
            for shm_region in shm_regions:
                if TEST_SYSTEM_SHARED_MEMORY:
                    shm_handle = shm.create_shared_memory_region(
                        shm_region + '_data', '/' + shm_region, 512)
                    self.triton_client_.register_system_shared_memory(
                        shm_region + '_data', '/' + shm_region, 512)
                else:
                    shm_handle = cudashm.create_shared_memory_region(
                        shm_region + '_data', 512, 0)
                    self.triton_client_.register_cuda_shared_memory(
                        shm_region + '_data',
                        cudashm.get_raw_handle(shm_handle), 0, 512)
                # Collect precreated handles for cleanup
                self.precreated_shm_regions_.append(shm_handle)
                precreated_shm_regions.append(shm_handle)
            return precreated_shm_regions
        return []

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
                       trial,
                       bs,
                       thresholds,
                       requested_outputs=("OUTPUT0", "OUTPUT1"),
                       input_size=16,
                       shm_region_names=None,
                       precreated_shm_regions=None):
        try:
            start_ms = int(round(time.time() * 1000))

            if trial == "savedmodel" or trial == "graphdef" or trial == "libtorch" \
                    or trial == "onnx" or trial == "plan" or trial == "python":
                tensor_shape = (bs, input_size)
                iu.infer_exact(
                    self,
                    trial,
                    tensor_shape,
                    bs,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=False,
                    model_version=1,
                    outputs=requested_outputs,
                    use_http_json_tensors=False,
                    use_grpc=USE_GRPC,
                    use_http=USE_HTTP,
                    skip_request_id_check=True,
                    use_streaming=False,
                    shm_region_names=shm_region_names,
                    precreated_shm_regions=precreated_shm_regions,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            else:
                self.assertFalse(True, "unknown trial type: " + trial)

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

    def check_setup(self, model_name, preferred_batch_sizes,
                    max_queue_delay_us):
        # Make sure test.sh set up the correct batcher settings
        config = self.triton_client_.get_model_config(model_name).config
        bconfig = config.dynamic_batching
        self.assertEqual(len(bconfig.preferred_batch_size),
                         len(preferred_batch_sizes))
        for i in preferred_batch_sizes:
            self.assertTrue(i in bconfig.preferred_batch_size)
        self.assertEqual(bconfig.max_queue_delay_microseconds,
                         max_queue_delay_us)

    def check_status(self, model_name, batch_exec, request_cnt, infer_cnt,
                     exec_count):
        stats = self.triton_client_.get_inference_statistics(model_name, "1")
        self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
        self.assertEqual(stats.model_stats[0].name, model_name,
                         "expect model stats for model {}".format(model_name))
        self.assertEqual(
            stats.model_stats[0].version, "1",
            "expect model stats for model {} version 1".format(model_name))

        if batch_exec:
            batch_stats = stats.model_stats[0].batch_stats
            self.assertEqual(
                len(batch_stats), len(batch_exec),
                "expected {} different batch-sizes, got {}".format(
                    len(batch_exec), len(batch_stats)))

            for batch_stat in batch_stats:
                bs = batch_stat.batch_size
                bc = batch_stat.compute_infer.count
                self.assertTrue(bs in batch_exec,
                                "unexpected batch-size {}".format(bs))
                # Get count from one of the stats
                self.assertEqual(
                    bc, batch_exec[bs],
                    "expected model-execution-count {} for batch size {}, got {}"
                    .format(batch_exec[bs], bs, bc))

        actual_request_cnt = stats.model_stats[0].inference_stats.success.count
        self.assertEqual(
            actual_request_cnt, request_cnt,
            "expected model-request-count {}, got {}".format(
                request_cnt, actual_request_cnt))

        actual_exec_cnt = stats.model_stats[0].execution_count
        self.assertIn(
            actual_exec_cnt, exec_count,
            "expected model-exec-count {}, got {}".format(
                request_cnt, actual_exec_cnt))

        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt, infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt))

    def test_static_batch_preferred(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    2, (3000, None),
                    precreated_shm_regions=precreated_shm_regions)
                self.check_response(
                    trial,
                    6, (3000, None),
                    precreated_shm_regions=precreated_shm_regions)
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1, 6: 1}, 2, 8, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_lt_any_preferred(self):
        # Send a request with a static batch size < any preferred
        # size. This should cause the response to be delayed by the
        # max batch queue delay
        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    1, (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                    precreated_shm_regions=precreated_shm_regions)
                self.check_deferred_exception()
                self.check_status(model_name, {1: 1}, 1, 1, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_not_preferred(self):
        # Send a request with a static batch size in between preferred
        # sizes. This should cause the response to be delayed by the
        # max batch queue delay
        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    3, (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                    precreated_shm_regions=precreated_shm_regions)
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1}, 1, 3, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_gt_max_preferred(self):
        # Send a request with a static batch size > maximum preferred
        # size. This should cause the request to be issued immediately
        # (even though the maximum batching queue delay is very high).
        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    7, (3000, None),
                    precreated_shm_regions=precreated_shm_regions)
                self.check_deferred_exception()
                self.check_status(model_name, {7: 1}, 1, 7, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_different_shape_allow_ragged(self):
        # Send two requests with static batch sizes == preferred size,
        # but with different shapes (using model with variable-size
        # tensors). Input tensors are marked as allowing ragged batch
        # so requests should be batched.
        for trial in _ragged_batch_supported_trials:
            try:
                dtype = np.float32
                model_name = tu.get_zero_model_name(trial, 1, dtype)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(target=iu.infer_zero,
                                     args=(self, trial, 1, dtype, ([1, 16],),
                                           ([1, 16],)),
                                     kwargs={
                                         'use_grpc': USE_GRPC,
                                         'use_http': USE_HTTP,
                                         'use_http_json_tensors': False,
                                         'use_streaming': False
                                     }))
                threads.append(
                    threading.Thread(target=iu.infer_zero,
                                     args=(self, trial, 1, dtype, ([1, 8],),
                                           ([1, 8],)),
                                     kwargs={
                                         'use_grpc': USE_GRPC,
                                         'use_http': USE_HTTP,
                                         'use_http_json_tensors': False,
                                         'use_streaming': False
                                     }))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_different_shape(self):
        # Send two requests with sum of static batch sizes ==
        # preferred size, but with different shapes (using model with
        # variable-size tensors). This should cause the requests to
        # not be batched. The first response will come back
        # immediately and the second delayed by the max batch queue
        # delay
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'input_size': 16,
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {1: 2}, 2, 2, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred(self):
        # Send two requests with total static batch size in between
        # preferred sizes. This should cause the first response to be
        # delayed by the max batch queue delay, and the second by max
        # delay (minus the difference in time that they arrived in the
        # queue)
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms - 2000)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1}, 2, 4, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. This should cause the first
        # two requests to be immediately responded to and the third
        # response to be delayed by the max batch queue delay.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {1: 1, 4: 1}, 3, 5, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. This should cause the first
        # two requests to be immediately responded to. Send a forth
        # request with the same shape as the third that causes a
        # preferred size so that third and forth response are sent
        # immediately.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 5, (6000, None)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 6: 1}, 4, 10, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_gt_max_preferred(self):
        # Send two requests with first not having preferred size and
        # second being larger than max preferred size. Delay the
        # second request so that it arrives after the first is already
        # be processed by the dynamic batcher. This should cause both
        # responses to be returned immediately.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 7, (3000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 7: 1}, 2, 10, (2,))
            except Exception as ex:
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
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 4, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 4: 1}, 2, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output0(self):
        # Send two requests where both ask for OUTPUT0. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00']
            shm1_region_names = ['ip10', 'ip11', 'op10']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00'])
        precreated_shm1_regions = self.create_advance(['op10'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT0",),
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT0",),
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output1(self):
        # Send two requests where both ask for OUTPUT1. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op01'])
        precreated_shm1_regions = self.create_advance(['op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT1",),
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT1",),
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_outputs(self):
        # Send two requests where one request asks for one output and
        # the other request asks for the other output. They should be
        # batched and get the correct response even though they don't
        # request both outputs.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00']
            shm1_region_names = ['ip10', 'ip11', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00'])
        precreated_shm1_regions = self.create_advance(['op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT0",),
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'requested_outputs': ("OUTPUT1",),
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_output_order(self):
        # Send two requests that ask for both outputs, but in a
        # different order. They should be batched and get the correct
        # response even though they use different order.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op11', 'op10']
        else:
            shm0_region_names = None
            shm1_region_names = None
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(trial, 1, (6000, None)),
                                     kwargs={
                                         'requested_outputs':
                                             ("OUTPUT0", "OUTPUT1"),
                                         'shm_region_names': shm0_region_names
                                     }))
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(trial, 1, (6000, None)),
                                     kwargs={
                                         'requested_outputs':
                                             ("OUTPUT1", "OUTPUT0"),
                                         'shm_region_names': shm1_region_names
                                     }))
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_sum_gt_max_preferred(self):
        # Send two requests with first not having preferred size and
        # second being smaller than max preferred size but the sum of
        # the requests being larger than max preferred size. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. This should cause first response to be returned
        # immediately but the second response, since it alone is not
        # greater than max preferred size, will be delayed.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                # Need scheduler to wait for queue to contain 2 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 4, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 4: 1}, 2, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_use_max_batch(self):
        # Send three requests with first not having preferred size,
        # second being smaller than max preferred size but the sum of
        # the requests being larger than max preferred size and thrid
        # is sent after the first two requests exceeds the queue delay
        # and the sum of the requests to be in full batch. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. This should cause all response to be returned together,
        # while it appears that the first two responses to be returned
        # after being delayed and the third response to be returned immediately.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                # Need scheduler to wait for queue to contain 3 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 4, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads[0].start()
                threads[1].start()
                time.sleep(11)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {8: 1}, 3, 8, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_preferred_different_shape(self):
        # Send two requests with total static batch size in between
        # preferred sizes. Then send a request with a different shape
        # and a non-preferred batch size. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. This should cause the first two requests to be
        # immediately responded to. Send a forth request with the same
        # shape as the third that causes a preferred size so that
        # third and forth response are sent immediately.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                # Need scheduler to wait for queue to contain 4 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 5, (3000, None)),
                        kwargs={
                            'input_size': 8,
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 6: 1}, 4, 10, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_biggest_preferred(self):
        # Send multiple requests that sum to multiple preferred sizes
        # and make sure the largest preferred size is used for the
        # batch. Use TRITONSERVER_DELAY_SCHEDULER in the environment so
        # that requests can be queued up before scheduler starts
        # servicing.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
            shm4_region_names = ['ip40', 'ip41', 'op40', 'op41']
            shm5_region_names = ['ip50', 'ip51', 'op50', 'op51']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
            shm5_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        precreated_shm4_regions = self.create_advance(['op40', 'op41'])
        precreated_shm5_regions = self.create_advance(['op50', 'op51'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                # Need scheduler to wait for queue to contain 6 request
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 6)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm4_region_names,
                            'precreated_shm_regions': precreated_shm4_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm5_region_names,
                            'precreated_shm_regions': precreated_shm5_regions
                        }))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {6: 1}, 6, 6, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_best_preferred(self):
        # Send multiple requests where the initial ones sum to a
        # preferred size and then extra request goes beyond that. The
        # initial requests should be handled immediately at the
        # preferred batch size and then the other one after
        # timeout. Use TRITONSERVER_DELAY_SCHEDULER in the environment so
        # that requests can be queued up before scheduler starts
        # servicing.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                # Need scheduler to wait for queue to contain 3 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (_max_queue_delay_ms * 1.5,
                                         _max_queue_delay_ms)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1, 1: 1}, 3, 3, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_preserve_ordering(self):
        model_base = "custom"
        dtype = np.float32
        shapes = ([
            1,
            1,
        ],)

        try:
            # use threads to send 12 requests without waiting for response
            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = [
                        "input" + str(i), "output" + str(i)
                    ]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(target=iu.infer_zero,
                                     args=(self, model_base, 1, dtype, shapes,
                                           shapes),
                                     kwargs={
                                         'use_grpc':
                                             USE_GRPC,
                                         'use_http':
                                             USE_HTTP,
                                         'use_http_json_tensors':
                                             False,
                                         'use_streaming':
                                             False,
                                         'shm_region_name_prefix':
                                             shm_region_name_prefix,
                                         'use_system_shared_memory':
                                             TEST_SYSTEM_SHARED_MEMORY,
                                         'use_cuda_shared_memory':
                                             TEST_CUDA_SHARED_MEMORY
                                     }))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, {4: 3}, 12, 12, (3,))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_aligned(self):
        # Send 4 requests with batch size 1. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. The batcher should form a batch of preferred
        # size 4.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [4, 6], 0)

                # Need scheduler to wait for queue to contain 4 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1}, 4, 4, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_unaligned(self):
        # Send 5 requests with batch size 1. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. The batcher should form a batch of preferred
        # size 4 followed by a batch of size 1.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
            shm4_region_names = ['ip40', 'ip41', 'op40', 'op41']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        precreated_shm4_regions = self.create_advance(['op40', 'op41'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [4, 6], 0)

                # Need scheduler to wait for queue to contain 3 requests
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 5)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm4_region_names,
                            'precreated_shm_regions': precreated_shm4_regions
                        }))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 1: 1}, 5, 5, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_use_biggest_preferred(self):
        # Send 7 requests with batch size 1. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. The batcher should form a batch of largest preferred
        # size 6 followed by a batch of size 1.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
            shm3_region_names = ['ip30', 'ip31', 'op30', 'op31']
            shm4_region_names = ['ip40', 'ip41', 'op40', 'op41']
            shm5_region_names = ['ip50', 'ip51', 'op50', 'op51']
            shm6_region_names = ['ip60', 'ip61', 'op60', 'op61']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
            shm5_region_names = None
            shm6_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        precreated_shm3_regions = self.create_advance(['op30', 'op31'])
        precreated_shm4_regions = self.create_advance(['op40', 'op41'])
        precreated_shm5_regions = self.create_advance(['op50', 'op51'])
        precreated_shm6_regions = self.create_advance(['op60', 'op61'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [4, 6], 0)

                # Need scheduler to wait for queue to contain 6 request
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 7)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm3_region_names,
                            'precreated_shm_regions': precreated_shm3_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm4_region_names,
                            'precreated_shm_regions': precreated_shm4_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm5_region_names,
                            'precreated_shm_regions': precreated_shm5_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm6_region_names,
                            'precreated_shm_regions': precreated_shm6_regions
                        }))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {6: 1, 1: 1}, 7, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_use_no_preferred_size(self):
        # Send 3 requests with batch size 1. Use
        # TRITONSERVER_DELAY_SCHEDULER in the environment so that
        # requests can be queued up before scheduler starts
        # servicing. The batcher should form a batch of of 3.
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ['ip00', 'ip01', 'op00', 'op01']
            shm1_region_names = ['ip10', 'ip11', 'op10', 'op11']
            shm2_region_names = ['ip20', 'ip21', 'op20', 'op21']
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(['op00', 'op01'])
        precreated_shm1_regions = self.create_advance(['op10', 'op11'])
        precreated_shm2_regions = self.create_advance(['op20', 'op21'])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_setup(model_name, [4, 6], 0)

                # Need scheduler to wait for queue to contain 3 request
                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm0_region_names,
                            'precreated_shm_regions': precreated_shm0_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm1_region_names,
                            'precreated_shm_regions': precreated_shm1_regions
                        }))
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            'shm_region_names': shm2_region_names,
                            'precreated_shm_regions': precreated_shm2_regions
                        }))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1}, 3, 3, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_queue_delay_only_non_default(self):
        # Send 12 requests with batch size 1. The max_queue_delay is set
        # to non-zero. Depending upon the timing of the requests arrival
        # there can be either 1 or 2 model executions.
        model_base = "custom"
        dtype = np.float32
        shapes = ([
            1,
            1,
        ],)

        try:
            # use threads to send 12 requests without waiting for response
            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = [
                        "input" + str(i), "output" + str(i)
                    ]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(target=iu.infer_zero,
                                     args=(self, model_base, 1, dtype, shapes,
                                           shapes),
                                     kwargs={
                                         'use_grpc':
                                             USE_GRPC,
                                         'use_http':
                                             USE_HTTP,
                                         'use_http_json_tensors':
                                             False,
                                         'use_streaming':
                                             False,
                                         'shm_region_name_prefix':
                                             shm_region_name_prefix,
                                         'use_system_shared_memory':
                                             TEST_SYSTEM_SHARED_MEMORY,
                                         'use_cuda_shared_memory':
                                             TEST_CUDA_SHARED_MEMORY
                                     }))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (1, 2))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_queue_delay_only_default(self):
        # Send 12 requests with batch size 1. The max_queue_delay is set
        # to default value of 0. There should be two distinct model
        # executions. The first few requests will form a first batch
        # and the remaining requests will form the second batch.
        model_base = "custom"
        dtype = np.float32
        shapes = ([
            1,
            1,
        ],)

        try:
            # use threads to send 12 requests without waiting for response
            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = [
                        "input" + str(i), "output" + str(i)
                    ]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(target=iu.infer_zero,
                                     args=(self, model_base, 1, dtype, shapes,
                                           shapes),
                                     kwargs={
                                         'use_grpc':
                                             USE_GRPC,
                                         'use_http':
                                             USE_HTTP,
                                         'use_http_json_tensors':
                                             False,
                                         'use_streaming':
                                             False,
                                         'shm_region_name_prefix':
                                             shm_region_name_prefix,
                                         'use_system_shared_memory':
                                             TEST_SYSTEM_SHARED_MEMORY,
                                         'use_cuda_shared_memory':
                                             TEST_CUDA_SHARED_MEMORY
                                     }))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (2,))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
