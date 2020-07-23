# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
import shutil
import time
import unittest
import numpy as np
import infer_util as iu
import test_util as tu

import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class LifeCycleTest(tu.TestResCollector):

    def _infer_success_models(self,
                              model_base_names,
                              versions,
                              tensor_shape,
                              swap=False):
        for base_name in model_base_names:
            try:
                model_name = tu.get_model_name(base_name, np.float32,
                                               np.float32, np.float32)
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    # FIXME is_server_ready should be true here DLIS-1296
                    # self.assertTrue(triton_client.is_server_ready())
                    for v in versions:
                        self.assertTrue(
                            triton_client.is_model_ready(model_name, str(v)))

                for v in versions:
                    iu.infer_exact(self,
                                   base_name,
                                   tensor_shape,
                                   1,
                                   np.float32,
                                   np.float32,
                                   np.float32,
                                   model_version=v,
                                   swap=(swap or (v == 3)))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_noexit(self):
        # Server was started with invalid args and
        # --exit-on-error=false so expect it to be running with
        # SERVER_FAILED_TO_INITIALIZE status.
        # Server is not live and not ready regardless of --strict-readiness
        try:
            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            self.assertFalse(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            md = triton_client.get_server_metadata()
            self.assertEqual(os.environ["TRITON_SERVER_VERSION"], md.version)
            self.assertEqual("triton", md.name)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            self.assertFalse(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            md = triton_client.get_server_metadata()
            self.assertEqual(os.environ["TRITON_SERVER_VERSION"], md['version'])
            self.assertEqual("triton", md['name'])
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_modelfail(self):
        # --strict-readiness=true so server is live but not ready
        tensor_shape = (1, 16)

        # Server was started but with a model that fails to load
        try:
            model_name = tu.get_model_name('graphdef', np.float32, np.float32,
                                           np.float32)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            self.assertTrue(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))

            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            self.assertTrue(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Inferencing with the missing model should fail.
        try:
            iu.infer_exact(self, 'graphdef', tensor_shape, 1, np.float32,
                           np.float32, np.float32)
            self.assertTrue(
                False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'graphdef_float32_float32_float32' has no available versions"
            ))

        # And other models should be loaded successfully
        try:
            for base_name in ["savedmodel", 'netdef']:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    model_name = tu.get_model_name(base_name, np.float32,
                                                   np.float32, np.float32)
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(self,
                               base_name,
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_modelfail_nostrict(self):
        # --strict-readiness=false so server is live and ready
        tensor_shape = (1, 16)

        # Server was started but with a model that fails to load
        try:
            model_name = tu.get_model_name('graphdef', np.float32, np.float32,
                                           np.float32)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))

            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Inferencing with the missing model should fail.
        try:
            iu.infer_exact(self, 'graphdef', tensor_shape, 1, np.float32,
                           np.float32, np.float32)
            self.assertTrue(
                False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'graphdef_float32_float32_float32' has no available versions"
            ))

        # And other models should be loaded successfully
        try:
            for base_name in ["savedmodel", 'netdef']:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    model_name = tu.get_model_name(base_name, np.float32,
                                                   np.float32, np.float32)
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(self,
                               base_name,
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_no_model_config(self):
        tensor_shape = (1, 16)

        # Server was started but with a model that fails to be polled
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                model_name = tu.get_model_name('graphdef', np.float32,
                                               np.float32, np.float32)

                # expecting ready because not strict readiness
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                md = triton_client.get_model_metadata(model_name, "1")
                self.assertTrue(
                    False, "expected model '" + model_name +
                    "' to be ignored due to polling failure")

            except Exception as ex:
                self.assertTrue(ex.message().startswith(
                    "Request for unknown model: 'graphdef_float32_float32_float32' is not found"
                ))

        # And other models should be loaded successfully
        try:
            for base_name in ["savedmodel", 'netdef']:
                model_name = tu.get_model_name(base_name, np.float32,
                                               np.float32, np.float32)
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(self,
                               base_name,
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_init_error_modelfail(self):
        # --strict-readiness=true so server is live but not ready

        # Server was started but with models that fail to load
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertFalse(triton_client.is_server_ready())

                # one model uses sequence batcher while the other uses dynamic batcher
                model_names = [
                    "custom_sequence_int32", "custom_int32_int32_int32"
                ]
                for model_name in model_names:
                    self.assertFalse(triton_client.is_model_ready(model_name))

            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # And other models should be loaded successfully
            try:
                for base_name in ["graphdef", "savedmodel", 'netdef']:
                    model_name = tu.get_model_name(base_name, np.float32,
                                                   np.float32, np.float32)
                    self.assertTrue(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            tensor_shape = (1, 16)
            for base_name in ["graphdef", "savedmodel", 'netdef']:
                iu.infer_exact(self,
                               base_name,
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_model_no_version(self):
        # --strict-readiness=true so server is live but not ready
        tensor_shape = (1, 16)

        # Server was started but with a model that fails to load
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertFalse(triton_client.is_server_ready())

                model_name = tu.get_model_name('graphdef', np.float32,
                                               np.float32, np.float32)
                self.assertFalse(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Sanity check that other models are loaded properly
            try:
                for base_name in ["savedmodel", "netdef"]:
                    model_name = tu.get_model_name(base_name, np.float32,
                                                   np.float32, np.float32)
                    self.assertTrue(triton_client.is_model_ready(model_name))
                for version in ["1", "3"]:
                    model_name = tu.get_model_name("plan", np.float32,
                                                   np.float32, np.float32)
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, version))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            for base_name in ["savedmodel", "netdef"]:
                iu.infer_exact(self,
                               base_name,
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=True)
            for version in [1, 3]:
                iu.infer_exact(self,
                               'plan',
                               tensor_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=(version == 3),
                               model_version=version)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            iu.infer_exact(self, 'graphdef', tensor_shape, 1, np.float32,
                           np.float32, np.float32)
            self.assertTrue(
                False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'graphdef_float32_float32_float32' has no available versions"
            ))

    def test_parse_ignore_zero_prefixed_version(self):
        tensor_shape = (1, 16)

        # Server was started but only version 1 is loaded
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                model_name = tu.get_model_name('savedmodel', np.float32,
                                               np.float32, np.float32)
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            # swap=False for version 1
            iu.infer_exact(self,
                           'savedmodel',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=False)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_model_load_unload(self):
        tensor_shape = (1, 16)
        savedmodel_name = tu.get_model_name('savedmodel', np.float32,
                                            np.float32, np.float32)
        netdef_name = tu.get_model_name('netdef', np.float32, np.float32,
                                        np.float32)

        # Make sure savedmodel model is not in the status (because
        # initially it is not in the model repository)
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Add savedmodel model to the model repository and give it time to
        # load. Make sure that it has a status and is ready.
        try:
            shutil.copytree(savedmodel_name, "models/" + savedmodel_name)
            time.sleep(5)  # wait for model to load
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference on the just loaded model
        try:
            iu.infer_exact(self,
                           'savedmodel',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=True)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Make sure savedmodel has execution stats
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(savedmodel_name)
            self.assertEqual(len(stats["model_stats"]), 2)
            for idx in range(len(stats["model_stats"])):
                self.assertEqual(stats["model_stats"][idx]["name"],
                                 savedmodel_name)
                if stats["model_stats"][idx]["version"] == "1":
                    self.assertEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"]
                        ["count"], 0)
                else:
                    self.assertNotEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"]
                        ["count"], 0)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(savedmodel_name)
            self.assertEqual(len(stats.model_stats), 2)
            for idx in range(len(stats.model_stats)):
                self.assertEqual(stats.model_stats[idx].name, savedmodel_name)
                if stats.model_stats[idx].version == "1":
                    self.assertEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0)
                else:
                    self.assertNotEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Remove savedmodel model from the model repository and give it
        # time to unload. Make sure that it is no longer available.
        try:
            shutil.rmtree("models/" + savedmodel_name)
            time.sleep(5)  # wait for model to unload
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Model is removed so inference should fail
        try:
            iu.infer_exact(self,
                           'savedmodel',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=True)
            self.assertTrue(
                False,
                "expected error for unavailable model " + savedmodel_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'savedmodel_float32_float32_float32' has no available versions"
            ))

        # Add back the same model. The status/stats should be reset.
        try:
            shutil.copytree(savedmodel_name, "models/" + savedmodel_name)
            time.sleep(5)  # wait for model to load
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))

            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(savedmodel_name)
            self.assertEqual(len(stats["model_stats"]), 2)
            self.assertEqual(stats["model_stats"][0]["name"], savedmodel_name)
            self.assertEqual(stats["model_stats"][1]["name"], savedmodel_name)
            self.assertEqual(
                stats["model_stats"][0]["inference_stats"]["success"]["count"],
                0)
            self.assertEqual(
                stats["model_stats"][1]["inference_stats"]["success"]["count"],
                0)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(savedmodel_name)
            self.assertEqual(len(stats.model_stats), 2)
            self.assertEqual(stats.model_stats[0].name, savedmodel_name)
            self.assertEqual(stats.model_stats[1].name, savedmodel_name)
            self.assertEqual(stats.model_stats[0].inference_stats.success.count,
                             0)
            self.assertEqual(stats.model_stats[1].inference_stats.success.count,
                             0)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Remove netdef model from the model repository and give it
        # time to unload. Make sure that it is unavailable.
        try:
            shutil.rmtree("models/" + netdef_name)
            time.sleep(5)  # wait for model to unload
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertTrue(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertFalse(triton_client.is_model_ready(netdef_name, "1"))
                self.assertFalse(triton_client.is_model_ready(netdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Model is removed so inference should fail
        try:
            iu.infer_exact(self,
                           'netdef',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=True)
            self.assertTrue(
                False, "expected error for unavailable model " + netdef_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'netdef_float32_float32_float32' has no available versions"
            ))

    def test_dynamic_model_load_unload_disabled(self):
        tensor_shape = (1, 16)
        savedmodel_name = tu.get_model_name('savedmodel', np.float32,
                                            np.float32, np.float32)
        netdef_name = tu.get_model_name('netdef', np.float32, np.float32,
                                        np.float32)

        # Make sure savedmodel model is not in the status (because
        # initially it is not in the model repository)
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Add savedmodel model to the model repository and give it time to
        # load. But it shouldn't load because dynamic loading is disabled.
        try:
            shutil.copytree(savedmodel_name, "models/" + savedmodel_name)
            time.sleep(5)  # wait for model to load
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference which should fail because the model isn't there
        try:
            iu.infer_exact(self,
                           'savedmodel',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=True)
            self.assertTrue(
                False,
                "expected error for unavailable model " + savedmodel_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'savedmodel_float32_float32_float32' is not found"
            ))

        # Remove one of the original models from the model repository.
        # Unloading is disabled so it should remain available in the status.
        try:
            shutil.rmtree("models/" + netdef_name)
            time.sleep(5)  # wait for model to unload (but it shouldn't)
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(netdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference to make sure model still being served even
        # though deleted from model repository
        try:
            iu.infer_exact(self,
                           'netdef',
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           swap=True)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_version_load_unload(self):
        tensor_shape = (1, 16)
        graphdef_name = tu.get_model_name('graphdef', np.int32, np.int32,
                                          np.int32)

        # There are 3 versions. Make sure that all have status and are
        # ready.
        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "2"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference on version 1 to make sure it is available
        try:
            iu.infer_exact(self,
                           'graphdef',
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           swap=False,
                           model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Make sure only version 1 has execution stats in the status.
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(graphdef_name)
            self.assertEqual(len(stats["model_stats"]), 3)
            for idx in range(len(stats["model_stats"])):
                self.assertEqual(stats["model_stats"][idx]["name"],
                                 graphdef_name)
                if stats["model_stats"][idx]["version"] == "1":
                    self.assertNotEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"]
                        ["count"], 0)
                else:
                    self.assertEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"]
                        ["count"], 0)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            stats = triton_client.get_inference_statistics(graphdef_name)
            self.assertEqual(len(stats.model_stats), 3)
            for idx in range(len(stats.model_stats)):
                self.assertEqual(stats.model_stats[idx].name, graphdef_name)
                if stats.model_stats[idx].version == "1":
                    self.assertNotEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0)
                else:
                    self.assertEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Remove version 1 from the model repository and give it time to
        # unload. Make sure that it is unavailable.
        try:
            shutil.rmtree("models/" + graphdef_name + "/1")
            time.sleep(5)  # wait for version to unload
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(graphdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "2"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Version is removed so inference should fail
        try:
            iu.infer_exact(self,
                           'graphdef',
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           swap=False,
                           model_version=1)
            self.assertTrue(
                False, "expected error for unavailable model " + graphdef_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'graphdef_int32_int32_int32' version 1 is not at ready state"
            ))

        # Add another version to the model repository.
        try:
            shutil.copytree("models/" + graphdef_name + "/2",
                            "models/" + graphdef_name + "/7")
            time.sleep(5)  # wait for version to load
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(graphdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "2"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "3"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_version_load_unload_disabled(self):
        tensor_shape = (1, 16)
        graphdef_name = tu.get_model_name('graphdef', np.int32, np.int32,
                                          np.int32)

        # Add a new version to the model repository and give it time to
        # load. But it shouldn't load because dynamic loading is
        # disabled.
        try:
            shutil.copytree("models/" + graphdef_name + "/2",
                            "models/" + graphdef_name + "/7")
            time.sleep(5)  # wait for model to load
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "2"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "3"))
                self.assertFalse(
                    triton_client.is_model_ready(graphdef_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Remove one of the original versions from the model repository.
        # Unloading is disabled so it should remain available
        # in the status.
        try:
            shutil.rmtree("models/" + graphdef_name + "/1")
            time.sleep(5)  # wait for version to unload (but it shouldn't)
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "1"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "2"))
                self.assertTrue(triton_client.is_model_ready(
                    graphdef_name, "3"))
                self.assertFalse(
                    triton_client.is_model_ready(graphdef_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference to make sure model still being served even
        # though version deleted from model repository
        try:
            iu.infer_exact(self,
                           'graphdef',
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           swap=False,
                           model_version=1)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_model_modify(self):
        models_base = ('savedmodel', 'plan')
        models_shape = ((1, 16), (1, 16))
        models = list()
        for m in models_base:
            models.append(
                tu.get_model_name(m, np.float32, np.float32, np.float32))

        # Make sure savedmodel and plan are in the status
        for model_name in models:
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference on the model, both versions 1 and 3
        for version in (1, 3):
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(self,
                                   model_name,
                                   model_shape,
                                   1,
                                   np.float32,
                                   np.float32,
                                   np.float32,
                                   swap=(version == 3),
                                   model_version=version)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

        # Change the model configuration to use wrong label file
        for base_name, model_name in zip(models_base, models):
            shutil.copyfile("config.pbtxt.wrong." + base_name,
                            "models/" + model_name + "/config.pbtxt")

        time.sleep(5)  # wait for models to reload
        for model_name in models:
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(self,
                                   model_name,
                                   model_shape,
                                   1,
                                   np.float32,
                                   np.float32,
                                   np.float32,
                                   swap=(version == 3),
                                   model_version=version,
                                   output0_raw=False)
                    self.assertTrue(
                        False,
                        "expected error for wrong label for " + model_name)
                except AssertionError as ex:
                    self.assertTrue("'label9" in str(ex) and "!=" in str(ex),
                                    str(ex))

        # Change the model configuration to use correct label file and to have
        # the default version policy (so that only version 3) is available.
        for base_name, model_name in zip(models_base, models):
            shutil.copyfile("config.pbtxt." + base_name,
                            "models/" + model_name + "/config.pbtxt")

        time.sleep(5)  # wait for models to reload
        for model_name in models:
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Attempt inferencing using version 1, should fail since
        # change in model policy makes that no longer available.
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(self,
                               model_name,
                               model_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=False,
                               model_version=1)
                self.assertTrue(
                    False, "expected error for unavailable model " + model_name)
            except Exception as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

        # Version 3 should continue to work...
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(self,
                               model_name,
                               model_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=True,
                               model_version=3)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_file_delete(self):
        models_base = ('savedmodel', 'plan')
        models_shape = ((1, 16), (1, 16))
        models = list()
        for m in models_base:
            models.append(
                tu.get_model_name(m, np.float32, np.float32, np.float32))

        # Make sure savedmodel and plan are in the status
        for model_name in models:
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Run inference on the model, both versions 1 and 3
        for version in (1, 3):
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(self,
                                   model_name,
                                   model_shape,
                                   1,
                                   np.float32,
                                   np.float32,
                                   np.float32,
                                   swap=(version == 3),
                                   model_version=version)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

        # Delete model configuration, which cause model to be
        # re-loaded and use autofilled config, which means that
        # version policy will be latest and so only version 3 will be
        # available
        for model_name in models:
            os.remove("models/" + model_name + "/config.pbtxt")

        time.sleep(5)  # wait for models to reload
        for model_name in models:
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Only version 3 (latest) should work...
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(self,
                               model_name,
                               model_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=True,
                               model_version=3)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            try:
                iu.infer_exact(self,
                               model_name,
                               model_shape,
                               1,
                               np.float32,
                               np.float32,
                               np.float32,
                               swap=False,
                               model_version=1)
                self.assertTrue(
                    False,
                    "expected error for unavailable model " + graphdef_name)
            except Exception as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

    def test_multiple_model_repository_polling(self):
        model_shape = (1, 16)
        savedmodel_name = tu.get_model_name('savedmodel', np.float32,
                                            np.float32, np.float32)

        # Models should be loaded successfully and infer
        # successfully. Initially savedmodel only has version 1.
        self._infer_success_models([
            "savedmodel",
        ], (1,), model_shape)
        self._infer_success_models(["graphdef", 'netdef'], (1, 3), model_shape)

        # Add the savedmodel to the second model repository, should cause
        # it to be unloaded due to duplication
        shutil.copytree(savedmodel_name, "models_0/" + savedmodel_name)
        time.sleep(5)  # wait for models to reload
        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(["graphdef", 'netdef'], (1, 3), model_shape)

        # Remove the savedmodel from the first model repository, the
        # model from the second model repository should be loaded
        # properly. In the second model repository savedmodel should
        # have versions 1 and 3.
        shutil.rmtree("models/" + savedmodel_name)
        time.sleep(5)  # wait for model to unload
        self._infer_success_models(["savedmodel", "graphdef", 'netdef'], (1, 3),
                                   model_shape)

    def test_multiple_model_repository_control(self):
        # similar to test_multiple_model_repository_polling, but the
        # model load/unload is controlled by the API
        model_shape = (1, 16)
        savedmodel_name = tu.get_model_name("savedmodel", np.float32,
                                            np.float32, np.float32)
        model_bases = ['savedmodel', "graphdef", 'netdef']

        # Initially models are not loaded
        for base in model_bases:
            try:
                model_name = tu.get_model_name(base, np.float32, np.float32,
                                               np.float32)
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Load all models, here we use GRPC
        for base in model_bases:
            try:
                model_name = tu.get_model_name(base, np.float32, np.float32,
                                               np.float32)
                triton_client = grpcclient.InferenceServerClient(
                    "localhost:8001", verbose=True)
                triton_client.load_model(model_name)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Models should be loaded successfully and infer
        # successfully. Initially savedmodel only has version 1.
        self._infer_success_models([
            "savedmodel",
        ], (1,), model_shape)
        self._infer_success_models(["graphdef", 'netdef'], (1, 3), model_shape)

        # Add the savedmodel to the second model repository. Because
        # not polling this doesn't change any model state, all models
        # are still loaded and available.
        shutil.copytree(savedmodel_name, "models_0/" + savedmodel_name)
        self._infer_success_models([
            "savedmodel",
        ], (1,), model_shape)
        self._infer_success_models(["graphdef", 'netdef'], (1, 3), model_shape)

        # Reload savedmodel which will cause it to unload because it
        # is in 2 model repositories. Use HTTP here.
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.load_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "failed to load '{}'".format(savedmodel_name)))

        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(["graphdef", 'netdef'], (1, 3), model_shape)

        # Remove the savedmodel from the first model repository and
        # explicitly load savedmodel. The savedmodel from the second
        # model repository should be loaded properly. In the second
        # model repository savedmodel should have versions 1 and 3.
        shutil.rmtree("models/" + savedmodel_name)
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.load_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(ex.message().startswith(
                "failed to load '{}'".format(savedmodel_name)))

        self._infer_success_models(["savedmodel", "graphdef", 'netdef'], (1, 3),
                                   model_shape)

    def test_model_control(self):
        model_shape = (1, 16)
        savedmodel_name = tu.get_model_name('savedmodel', np.float32,
                                            np.float32, np.float32)

        ensemble_prefix = "simple_"
        ensemble_name = ensemble_prefix + savedmodel_name

        # Make sure no models are loaded
        for model_name in (savedmodel_name, ensemble_name):
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Load non-existent model
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                triton_client.load_model("unknown_model")
                self.assertTrue(False, "expected unknown model failure")
            except Exception as ex:
                self.assertTrue(ex.message().startswith(
                    "failed to load 'unknown_model', no version is available"))

        # Load ensemble model, the dependent model should be polled and loaded
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.load_model(ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "savedmodel",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_savedmodel",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        # Delete model configuration for savedmodel, which will cause
        # the autofiller to use the latest version policy so that only
        # version 3 will be available if the models are re-loaded
        for model_name in (savedmodel_name,):
            os.remove("models/" + model_name + "/config.pbtxt")

        self._infer_success_models([
            "savedmodel",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_savedmodel",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        # Reload models, only version 3 should be available for savedmodel
        for model_name in (savedmodel_name, ensemble_name):
            try:
                triton_client = grpcclient.InferenceServerClient(
                    "localhost:8001", verbose=True)
                triton_client.load_model(model_name)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "savedmodel",
        ], (3,), model_shape)
        self._infer_success_models([
            "simple_savedmodel",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        for model_name in (savedmodel_name,):
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Unload non-existing model, nothing should happen
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                triton_client.unload_model("unknown_model")
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Unload the depending model, as side effect, the ensemble model will be
        # forced to be unloaded
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.unload_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        for model_name in (savedmodel_name, ensemble_name):
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Explicitly unload the ensemble and load the depending
        # model. The ensemble model should not be reloaded because it
        # was explicitly unloaded.
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.unload_model(ensemble_name)
            triton_client.load_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "savedmodel",
        ], (3,), model_shape)

        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(ensemble_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(ensemble_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multiple_model_repository_control_startup_models(self):
        model_shape = (1, 16)
        savedmodel_name = tu.get_model_name('savedmodel', np.float32,
                                            np.float32, np.float32)
        graphdef_name = tu.get_model_name('graphdef', np.float32, np.float32,
                                          np.float32)

        ensemble_prefix = "simple_"
        savedmodel_ensemble_name = ensemble_prefix + savedmodel_name
        graphdef_ensemble_name = ensemble_prefix + graphdef_name

        # Make sure unloaded models are not in the status
        for base in ("netdef",):
            model_name = tu.get_model_name(base, np.float32, np.float32,
                                           np.float32)
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # And loaded models work properly
        self._infer_success_models([
            "savedmodel",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_savedmodel",
        ], (1, 3),
                                   model_shape,
                                   swap=True)
        self._infer_success_models([
            "graphdef",
        ], (1, 3), model_shape)

        # Load non-existing model
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                triton_client.load_model("unknown_model")
                self.assertTrue(False, "expected unknown model failure")
            except Exception as ex:
                self.assertTrue(ex.message().startswith(
                    "failed to load 'unknown_model', no version is available"))

        # Load graphdef ensemble model, the dependent model is already
        # loaded via command-line
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.load_model(graphdef_ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "graphdef",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_graphdef",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        # Delete model configuration, which will cause the autofiller
        # to use the latest version policy so that only version 3 will
        # be available if the models are re-loaded
        os.remove("models/" + savedmodel_name + "/config.pbtxt")

        self._infer_success_models([
            "graphdef",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_graphdef",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        # Reload savedmodel, only version 3 should be available
        try:
            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            triton_client.load_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "savedmodel",
        ], (3,), model_shape)
        self._infer_success_models([
            "simple_savedmodel",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        # Unload non-existing model, nothing should happen
        for triton_client in (httpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True),
                              grpcclient.InferenceServerClient("localhost:8001",
                                                               verbose=True)):
            try:
                triton_client.unload_model("unknown_model")
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Unload the savedmodel, as side effect, the ensemble model
        # will be forced to be unloaded
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.unload_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        for model_name in [savedmodel_name, savedmodel_ensemble_name]:
            try:
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Explicitly unload the savedmodel ensemble and load the
        # depending model. The ensemble model should not be reloaded
        # because it was explicitly unloaded.
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            triton_client.unload_model(savedmodel_ensemble_name)
            triton_client.load_model(savedmodel_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models([
            "savedmodel",
        ], (3,), model_shape)
        self._infer_success_models([
            "graphdef",
        ], (1, 3), model_shape)
        self._infer_success_models([
            "simple_graphdef",
        ], (1, 3),
                                   model_shape,
                                   swap=True)

        try:
            for triton_client in (httpclient.InferenceServerClient(
                    "localhost:8000", verbose=True),
                                  grpcclient.InferenceServerClient(
                                      "localhost:8001", verbose=True)):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_ensemble_name, "1"))
                self.assertFalse(
                    triton_client.is_model_ready(savedmodel_ensemble_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_repository_index(self):
        # use model control EXPLIT and --load-model to load a subset of models
        # in model repository
        tensor_shape = (1, 16)
        model_bases = ["graphdef", "savedmodel", "simple_savedmodel"]

        # Sanity check on loaded models
        # 3 models should be loaded:
        #     simple_savedmodel_float32_float32_float32
        #     savedmodel_float32_float32_float32
        #     graphdef_float32_float32_float32
        for model_base in model_bases:
            try:
                model_name = tu.get_model_name(model_base, np.float32,
                                               np.float32, np.float32)
                for triton_client in (httpclient.InferenceServerClient(
                        "localhost:8000", verbose=True),
                                      grpcclient.InferenceServerClient(
                                          "localhost:8001", verbose=True)):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        # Check model repository index
        # All models should be in ready state except netdef_float32_float32_float32
        # which appears in two repositories.
        model_bases.append("simple_graphdef")
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            index = triton_client.get_model_repository_index()
            indexed = list()
            self.assertEqual(len(index), 8)
            for i in index:
                indexed.append(i["name"])
                if i["name"] == "netdef_float32_float32_float32":
                    self.assertEqual(i["state"], "UNAVAILABLE")
                    self.assertEqual(
                        i["reason"],
                        "model appears in two or more repositories")
            for model_base in model_bases:
                model_name = tu.get_model_name(model_base, np.float32,
                                               np.float32, np.float32)
                self.assertTrue(model_name in indexed)

            triton_client = grpcclient.InferenceServerClient("localhost:8001",
                                                             verbose=True)
            index = triton_client.get_model_repository_index()
            indexed = list()
            self.assertEqual(len(index.models), 8)
            for i in index.models:
                indexed.append(i.name)
                if i.name == "netdef_float32_float32_float32":
                    self.assertEqual(i.state, "UNAVAILABLE")
                    self.assertEqual(
                        i.reason, "model appears in two or more repositories")
            for model_base in model_bases:
                model_name = tu.get_model_name(model_base, np.float32,
                                               np.float32, np.float32)
                self.assertTrue(model_name in indexed)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
