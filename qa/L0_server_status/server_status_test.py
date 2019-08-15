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
import infer_util as iu
import unittest
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status


def _get_server_status(url="localhost:8000", protocol=ProtocolType.HTTP, model_name=None):
   ctx = ServerStatusContext(url, protocol, model_name, True)
   return (ctx.get_server_status(), ctx.get_last_request_id())


class ServerStatusTest(unittest.TestCase):

    def test_basic(self):
        try:
            for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                model_name0 = "graphdef_int32_int8_int8"
                server_status0, req_id0 = _get_server_status(pair[0], pair[1], model_name0)
                self.assertEqual(os.environ["TENSORRT_SERVER_VERSION"],
                                server_status0.version)
                self.assertEqual("inference:0", server_status0.id)
                uptime0 = server_status0.uptime_ns
                self.assertGreater(uptime0, 0)
                self.assertEqual(len(server_status0.model_status), 1)
                self.assertTrue(model_name0 in server_status0.model_status,
                                "expected status for model " + model_name0)

                model_name1 = "graphdef_float32_float32_float32"
                server_status1, req_id1 = _get_server_status(pair[0], pair[1], model_name1)
                self.assertEqual(os.environ["TENSORRT_SERVER_VERSION"],
                                server_status1.version)
                self.assertEqual("inference:0", server_status1.id)
                uptime1 = server_status1.uptime_ns
                self.assertEqual(len(server_status1.model_status), 1)
                self.assertTrue(model_name1 in server_status1.model_status,
                                "expected status for model " + model_name1)

                self.assertGreater(uptime1, uptime0)
                self.assertNotEqual(req_id0, req_id1)

                server_status2, req_id2 = _get_server_status(pair[0], pair[1])
                self.assertEqual(os.environ["TENSORRT_SERVER_VERSION"],
                                server_status2.version)
                self.assertEqual("inference:0", server_status2.id)
                uptime2 = server_status2.uptime_ns
                for mn in (model_name0, model_name1, "netdef_float32_float32_float32",
                        "plan_float32_float32_float32"):
                    self.assertTrue(mn in server_status2.model_status,
                                    "expected status for model " + model_name1)

                self.assertGreater(uptime2, uptime1)
                self.assertEqual(req_id2, req_id1 + 1)

        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_unknown_model(self):
        try:
            for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                server_status = _get_server_status(pair[0], pair[1], "foo")
                self.assertTrue(False, "expected unknown model failure")
        except InferenceServerException as ex:
            self.assertEqual("inference:0", ex.server_id())
            self.assertGreater(ex.request_id(), 0)
            self.assertTrue(
                ex.message().startswith("no status available for unknown model"))

    def test_model_latest_infer(self):
        input_size = 16
        tensor_shape = (input_size,)

        # There are 3 versions of *_int32_int32_int32 and all
        # should be available.
        for platform in ('graphdef', 'netdef'):
            model_name = platform + "_int32_int32_int32"

            # Initially there should be no version stats..
            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status0, req_id0 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status0.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status0.model_status[model_name].version_status), 3,
                                    "expected status for 3 versions for model " + model_name)
                    for v in (1, 2, 3):
                        self.assertTrue(v in server_status0.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        self.assertEqual(server_status0.model_status[model_name].version_status[v].ready_state,
                                        server_status.MODEL_READY)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Infer using latest version (which is 3)...
            iu.infer_exact(self, platform, tensor_shape, 1,
                           np.int32, np.int32, np.int32,
                           model_version=None, swap=True)

            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status1, req_id1 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status1.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status1.model_status[model_name].version_status), 3,
                                    "expected status for 3 versions for model " + model_name)
                    for v in (1, 2, 3):
                        self.assertTrue(v in server_status1.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        self.assertEqual(server_status1.model_status[model_name].version_status[v].ready_state,
                                        server_status.MODEL_READY)

                    # Only version 3 should have infer stats
                    for v in (1, 2, 3):
                        version_status = server_status1.model_status[model_name].version_status[v]
                        if v == 3:
                            self.assertEqual(len(version_status.infer_stats), 1,
                                            "expected 1 infer stats for v" + str(v) + " model " + model_name)
                            self.assertTrue(1 in version_status.infer_stats,
                                            "expected batch 1 status for v" + str(v) + " model " + model_name)
                            infer_stats = version_status.infer_stats[1]
                            self.assertTrue(infer_stats.success.count, 1)
                        else:
                            self.assertEqual(len(version_status.infer_stats), 0,
                                            "unexpected infer stats for v" + str(v) + " model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_specific_infer(self):
        input_size = 16

        # There are 3 versions of *_float32_float32_float32 but only
        # versions 1 and 3 should be available.
        for platform in ('graphdef', 'netdef', 'plan'):
            tensor_shape = (input_size, 1, 1) if platform == 'plan' else (input_size,)
            model_name = platform + "_float32_float32_float32"

            # Initially there should be no version status...
            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status0, req_id0 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status0.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status0.model_status[model_name].version_status), 2,
                                    "expected status for 2 versions for model " + model_name)
                    for v in (1, 3):
                        self.assertTrue(v in server_status0.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        self.assertEqual(server_status0.model_status[model_name].version_status[v].ready_state,
                                        server_status.MODEL_READY)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Infer using version 1...
            iu.infer_exact(self, platform, tensor_shape, 1,
                           np.float32, np.float32, np.float32,
                           model_version=1, swap=False)

            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status1, req_id1 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status1.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status1.model_status[model_name].version_status), 2,
                                    "expected status for 2 versions for model " + model_name)
                    for v in (1, 3):
                        self.assertTrue(v in server_status1.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        self.assertEqual(server_status1.model_status[model_name].version_status[v].ready_state,
                                        server_status.MODEL_READY)

                    # Only version 1 should have infer stats
                    for v in (1, 3):
                        version_status = server_status1.model_status[model_name].version_status[v]
                        if v == 1:
                            self.assertEqual(len(version_status.infer_stats), 1,
                                            "expected 1 infer stats for v" + str(v) + " model " + model_name)
                            self.assertTrue(1 in version_status.infer_stats,
                                            "expected batch 1 status for v" + str(v) + " model " + model_name)
                            infer_stats = version_status.infer_stats[1]
                            self.assertTrue(infer_stats.success.count, 1)
                        else:
                            self.assertEqual(len(version_status.infer_stats), 0,
                                            "unexpected infer stats for v" + str(v) + " model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


class ModelStatusTest(unittest.TestCase):
    '''
    These tests must be run after the ServerStatusTest. See test.sh
    file for correct test running.
    '''
    def test_model_versions_deleted(self):
        # Originally There were 3 versions of *_int32_int32_int32 and
        # version 3 was executed once. Version 2 and 3 models were
        # deleted from the model repository so now only expect version 1 to
        # be ready and version 3 to show stats but not be ready.
        for platform in ('graphdef', 'netdef'):
            model_name = platform + "_int32_int32_int32"

            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status1, req_id1 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status1.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status1.model_status[model_name].version_status), 3,
                                    "expected status for 3 versions for model " + model_name)

                    # Only version 3 should have infer stats, only 1 is ready
                    for v in (1, 2, 3):
                        self.assertTrue(v in server_status1.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        version_status = server_status1.model_status[model_name].version_status[v]
                        if v == 1:
                            self.assertEqual(version_status.ready_state, server_status.MODEL_READY)
                            self.assertEqual(len(version_status.infer_stats), 0,
                                            "unexpected infer stats for v" + str(v) + " model " + model_name)
                        else:
                            self.assertEqual(version_status.ready_state, server_status.MODEL_UNAVAILABLE)
                            if v == 2:
                                self.assertEqual(len(version_status.infer_stats), 0,
                                                "unexpected infer stats for v" + str(v) + " model " + model_name)
                            else:
                                self.assertEqual(len(version_status.infer_stats), 1,
                                                "expected 1 infer stats for v" + str(v) + " model " + model_name)
                                self.assertTrue(1 in version_status.infer_stats,
                                            "expected batch 1 status for v" + str(v) + " model " + model_name)
                                infer_stats = version_status.infer_stats[1]
                                self.assertTrue(infer_stats.success.count, 1)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_versions_added(self):
        # Originally There was version 1 of *_float16_float32_float32.
        # Version 7 was added so now expect just version 7 to be ready.
        for platform in ('graphdef',):
            model_name = platform + "_float16_float32_float32"

            try:
                for pair in [("localhost:8000", ProtocolType.HTTP), ("localhost:8001", ProtocolType.GRPC)]:
                    server_status1, req_id1 = _get_server_status(pair[0], pair[1], model_name)
                    self.assertTrue(model_name in server_status1.model_status,
                                    "expected status for model " + model_name)
                    self.assertEqual(len(server_status1.model_status[model_name].version_status), 2,
                                    "expected status for 2 versions for model " + model_name)

                    for v in (1,):
                        self.assertTrue(v in server_status1.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        version_status = server_status1.model_status[model_name].version_status[v]
                        self.assertEqual(version_status.ready_state, server_status.MODEL_UNAVAILABLE)
                        self.assertEqual(len(version_status.infer_stats), 0,
                                        "unexpected infer stats for v" + str(v) + " model " + model_name)

                    for v in (7,):
                        self.assertTrue(v in server_status1.model_status[model_name].version_status,
                                        "expected version " + str(v) + " status for model " + model_name)
                        version_status = server_status1.model_status[model_name].version_status[v]
                        self.assertEqual(version_status.ready_state, server_status.MODEL_READY)
                        self.assertEqual(len(version_status.infer_stats), 0,
                                        "unexpected infer stats for v" + str(v) + " model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
