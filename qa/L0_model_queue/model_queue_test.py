# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
import time
import threading
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tritonclientutils import InferenceServerException
from ctypes import *

_max_queue_delay_ms = 10000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class ModelQueueTest(tu.TestResultCollector):

    def setUp(self):
        self.trials_ = []
        for base in ["custom", "ensemble"]:
            for is_http_trial in [True, False]:
                self.trials_.append({
                    "base": base,
                    "is_http_trial": is_http_trial
                })
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
                first_exception = _deferred_exceptions[0]
                _deferred_exceptions.pop(0)
                raise first_exception

    def check_response(self,
                       bs,
                       dtype,
                       shapes,
                       priority,
                       timeout_us,
                       thresholds,
                       base="custom",
                       is_http_trial=True):
        full_shapes = [[
            bs,
        ] + shape for shape in shapes]
        try:
            start_ms = int(round(time.time() * 1000))
            iu.infer_zero(self,
                          base,
                          bs,
                          dtype,
                          full_shapes,
                          full_shapes,
                          model_version=1,
                          use_http_json_tensors=False,
                          use_http=is_http_trial,
                          use_grpc=(not is_http_trial),
                          use_streaming=False,
                          priority=priority,
                          timeout_us=timeout_us)

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

    def test_max_queue_size(self):
        # Send a request with a static batch size == preferred size to trigger
        # model execution. Then sends 10 requests to overload the model queue,
        # expecting 2 of the requests are returned with error code immediately.
        dtype = np.float32
        shapes = ([16],)

        for trial in self.trials_:
            preceding_thread = threading.Thread(
                target=self.check_response,
                args=(8, dtype, shapes, 0, 0, (1999, 1000)),
            )
            threads = []
            for i in range(10):
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(1, dtype, shapes, 0, 0, (None,
                                                                    None)),
                                     kwargs=trial))
            preceding_thread.start()
            time.sleep(0.5)
            for t in threads:
                t.start()

            preceding_thread.join()
            for t in threads:
                t.join()

            # Expect at most two exception with exceeding max queue size error
            for i in range(2):
                try:
                    self.check_deferred_exception()
                except InferenceServerException as ex:
                    self.assertTrue(
                        "Exceeds maximum queue size" in ex.message(),
                        "Expected error message \"Exceeds maximum queue size\", got: {}"
                        .format(ex))
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_policy_delay(self):
        # Send requests with batch sizes 1, 1, 3 where the second and third
        # requests are sent after 'default_timeout_microseconds'.
        # Expect the first request is timed-out and delayed, which makes the
        # second and third request be batched together and executed. While the
        # first request must wait for 'max_queue_delay_microseconds' until it
        # can be executed.
        dtype = np.float32
        shapes = ([16],)
        for trial in self.trials_:
            try:
                threads = []
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(1, dtype, shapes, 0, 0, (15000,
                                                                    10000)),
                                     kwargs=trial))
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(2, dtype, shapes, 0, 0, (100, 0)),
                                     kwargs=trial))
                threads.append(
                    threading.Thread(target=self.check_response,
                                     args=(2, dtype, shapes, 0, 0, (100, 0)),
                                     kwargs=trial))
                threads[0].start()
                time.sleep(0.2)
                threads[1].start()
                threads[2].start()

                for t in threads:
                    t.join()
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_policy_reject(self):
        # Send requests with batch sizes 1, 1, 3 where the second and third
        # requests are sent after 'default_timeout_microseconds'.
        # Expect the first request is timed-out and rejected, which makes the
        # second and third request be batched together and executed.
        dtype = np.float32
        shapes = ([16],)
        for trial in self.trials_:
            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 0, 0, (None, None)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (100, 0)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (100, 0)),
                                 kwargs=trial))
            threads[0].start()
            time.sleep(0.2)
            threads[1].start()
            threads[2].start()

            for t in threads:
                t.join()

            # Expect only one error for rejection
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(
                    "Request timeout expired" in ex.message(),
                    "Expected error message \"Request timeout expired\", got: {}"
                    .format(ex))

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_timeout_override(self):
        # Send requests with batch sizes 1, 1, 3 where the first request
        # overrides the timout to be less than 'default_timeout_microseconds',
        # and the second and third requests are sent after the overridden
        # timeout. Expect the first request is timed-out and rejected before
        # 'default_timeout_microseconds', which makes the second and third
        # request be batched together and executed earlier than
        # 'default_timeout_microseconds'.

        dtype = np.float32
        shapes = ([16],)
        for trial in self.trials_:
            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 0, 100000, (None,
                                                                     None)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (100, 0)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (100, 0)),
                                 kwargs=trial))
            threads[0].start()
            time.sleep(0.2)
            threads[1].start()
            threads[2].start()

            for t in threads:
                t.join()

            # Expect only one error for rejection
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(
                    "Request timeout expired" in ex.message(),
                    "Expected error message \"Request timeout expired\", got: {}"
                    .format(ex))

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Check that timeout larger than 'default_timeout_microseconds' will not
            # override, the last two requests will be processed only after
            # 'default_timeout_microseconds' and before queue delay.
            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 0, 10000000, (None,
                                                                       None)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (1100, 700)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (1100, 700)),
                                 kwargs=trial))
            threads[0].start()
            time.sleep(0.2)
            threads[1].start()
            threads[2].start()

            for t in threads:
                t.join()

            # Expect only one error for rejection
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(
                    "Request timeout expired" in ex.message(),
                    "Expected error message \"Request timeout expired\", got: {}"
                    .format(ex))

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Sanity check that without override, the last two requests will be
            # processed only after 'default_timeout_microseconds'
            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 0, 0, (None, None)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (1100, 700)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (1100, 700)),
                                 kwargs=trial))
            threads[0].start()
            time.sleep(0.2)
            threads[1].start()
            threads[2].start()

            for t in threads:
                t.join()

            # Expect only one error for rejection
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(
                    "Request timeout expired" in ex.message(),
                    "Expected error message \"Request timeout expired\", got: {}"
                    .format(ex))

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_priority_levels(self):
        # Send 2 requests with batch sizes 2, 1 in default priority. Then send
        # 1 request with batch size 2 in priority 1. Expect the third request is
        # place in the front of the queue and form a preferred batch with the
        # first request.
        dtype = np.float32
        shapes = ([16],)
        for trial in self.trials_:
            threads = []
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 0, 0, (500, 200)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 0, 0, (15000, 10000)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 1, 0, (100, 0)),
                                 kwargs=trial))
            threads[0].start()
            # wait to make sure the order is correct
            time.sleep(0.1)
            threads[1].start()
            time.sleep(0.2)
            threads[2].start()

            for t in threads:
                t.join()

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_priority_with_policy(self):
        # Two set of requests are being sent at different priority levels
        # in sequence:
        # priority 1:
        #     batch size 2, default timeout
        #     batch size 1, short timeout
        #     batch size 2, default timeout
        # priority 2:
        #     batch size 2, medium timeout
        #     batch size 3, default timeout
        #     batch size 6, default timeout
        # Expecting that by the time when the last request, second request in
        # priority 2, is sent, the requests with short timeout will be handled
        # accordingly, and the queue becomes:
        # priority 1:
        #     batch size 2, default timeout (1st batch)
        #     batch size 2, default timeout (1st batch)
        #     batch size 1, short timeout (delayed, will be 2nd batch)
        # priority 2:
        #     batch size 2, medium timeout (will be rejected)
        #     batch size 3, default timeout (will be 2nd batch)
        #     batch size 6, default timeout (will be 3rd batch)

        dtype = np.float32
        shapes = ([16],)
        for trial in self.trials_:
            threads = []
            # The expected ranges may not be rounded to accommodate
            # the sleep between sending requests
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 1, 0, (2000, 1000)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(1, dtype, shapes, 1, 1000000, (3400,
                                                                      2400)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 1, 0, (1700, 700)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(2, dtype, shapes, 2, 2000000, (None,
                                                                      None)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(3, dtype, shapes, 2, 0, (2700, 1700)),
                                 kwargs=trial))
            threads.append(
                threading.Thread(target=self.check_response,
                                 args=(6, dtype, shapes, 2, 0, (15000, 10000)),
                                 kwargs=trial))
            for t in threads:
                t.start()
                time.sleep(0.2)

            for t in threads:
                t.join()

            # Expect only one error for rejection
            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(
                    "Request timeout expired" in ex.message(),
                    "Expected error message \"Request timeout expired\", got: {}"
                    .format(ex))

            try:
                self.check_deferred_exception()
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
