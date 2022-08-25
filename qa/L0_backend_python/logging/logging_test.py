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

sys.path.append("../../common")

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

_trials = ["python"]

_max_queue_delay_ms = 10000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class LogTest(tu.TestResultCollector):

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

if __name__ == '__main__':
    unittest.main()
