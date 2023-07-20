#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from builtins import range
from collections.abc import Iterable

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class BatcherTest(tu.TestResultCollector):
    def setUp(self):
        # The helper client for setup will be GRPC for simplicity.
        self.triton_client_ = grpcclient.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001"
        )
        self.precreated_shm_regions_ = []
        global _deferred_exceptions
        _deferred_exceptions = []

    def tearDown(self):
        super().tearDown()

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        # Just raise one of the exceptions...
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_response(
        self,
        trial,
        bs,
        thresholds,
        requested_outputs=("OUTPUT0", "OUTPUT1"),
        input_size=16,
        shm_region_names=None,
        precreated_shm_regions=None,
    ):
        try:
            start_ms = int(round(time.time() * 1000))

            if (
                trial == "savedmodel"
                or trial == "graphdef"
                or trial == "libtorch"
                or trial == "onnx"
                or trial == "plan"
                or trial == "python"
            ):
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
                    use_http=False,
                    use_grpc=False,
                    use_http_json_tensors=False,
                    skip_request_id_check=True,
                    use_streaming=False,
                )
            else:
                self.assertFalse(True, "unknown trial type: " + trial)

            end_ms = int(round(time.time() * 1000))

            lt_ms = thresholds[0]
            gt_ms = thresholds[1]
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

    def check_status(self, model_name, batch_exec, request_cnt, infer_cnt, exec_count):
        # There is a time window between when responses are returned and statistics are updated.
        # To prevent intermittent test failure during that window, wait up to 10 seconds for the
        # inference statistics to be ready.
        num_tries = 10
        for i in range(num_tries):
            stats = self.triton_client_.get_inference_statistics(model_name, "1")
            self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
            actual_exec_cnt = stats.model_stats[0].execution_count
            if actual_exec_cnt == exec_count:
                break
            print(
                "WARNING: expect {} executions, got {} (attempt {})".format(
                    exec_count, actual_exec_cnt, i
                )
            )
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

        if batch_exec:
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
        if isinstance(exec_count, Iterable):
            self.assertIn(
                actual_exec_cnt,
                exec_count,
                "expected model-exec-count {}, got {}".format(
                    exec_count, actual_exec_cnt
                ),
            )
        else:
            self.assertEqual(
                actual_exec_cnt,
                exec_count,
                "expected model-exec-count {}, got {}".format(
                    exec_count, actual_exec_cnt
                ),
            )
        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt,
            infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt
            ),
        )

    def test_volume_batching(self):
        # Send 12 requests with batch size 1. The max_queue_delay is set
        # to non-zero. Depending upon the timing of the requests arrival
        # there can be either 4-6 model executions.
        model_base = "onnx"
        dtype = np.float16
        shapes = (
            [
                1,
                4,
                4,
            ],
        )

        try:
            # use threads to send 12 requests without waiting for response
            threads = []
            for i in range(12):
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, model_base, 1, dtype, shapes, shapes),
                        kwargs={
                            "use_http": True,
                            "use_grpc": False,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                        },
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (4, 5, 6))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == "__main__":
    unittest.main()
