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

import numpy as np
import os
import unittest
import infer_util as iu
import test_util as tu
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import *


class ServerMetadataTest(tu.TestResultCollector):

    def test_basic(self):
        try:
            for pair in [("localhost:8000", "http"),
                         ("localhost:8001", "grpc")]:
                model_name = "graphdef_int32_int8_int8"
                extensions = [
                    'classification', 'sequence', 'model_repository',
                    'schedule_policy', 'model_configuration',
                    'system_shared_memory', 'cuda_shared_memory',
                    'binary_tensor_data', 'statistics'
                ]
                if pair[1] == "http":
                    triton_client = httpclient.InferenceServerClient(
                        url=pair[0], verbose=True)
                else:
                    triton_client = grpcclient.InferenceServerClient(
                        url=pair[0], verbose=True)

                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                server_metadata = triton_client.get_server_metadata()
                model_metadata = triton_client.get_model_metadata(model_name)

                if pair[1] == "http":
                    self.assertEqual(os.environ["TRITON_SERVER_VERSION"],
                                     server_metadata['version'])
                    self.assertEqual("triton", server_metadata['name'])
                    for ext in extensions:
                        self.assertIn(ext, server_metadata['extensions'])

                    self.assertEqual(model_name, model_metadata['name'])
                else:
                    self.assertEqual(os.environ["TRITON_SERVER_VERSION"],
                                     server_metadata.version)
                    self.assertEqual("triton", server_metadata.name)
                    for ext in extensions:
                        self.assertIn(ext, server_metadata.extensions)

                    self.assertEqual(model_name, model_metadata.name)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_unknown_model(self):
        try:
            for pair in [("localhost:8000", "http"),
                         ("localhost:8001", "grpc")]:
                model_name = "foo"
                if pair[1] == "http":
                    triton_client = httpclient.InferenceServerClient(
                        url=pair[0], verbose=True)
                else:
                    triton_client = grpcclient.InferenceServerClient(
                        url=pair[0], verbose=True)

                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                server_metadata = triton_client.get_server_metadata()
                if pair[1] == "http":
                    self.assertEqual(os.environ["TRITON_SERVER_VERSION"],
                                     server_metadata['version'])
                    self.assertEqual("triton", server_metadata['name'])
                else:
                    self.assertEqual(os.environ["TRITON_SERVER_VERSION"],
                                     server_metadata.version)
                    self.assertEqual("triton", server_metadata.name)

                model_metadata = triton_client.get_model_metadata(model_name)
                self.assertTrue(False, "expected unknown model failure")
        except InferenceServerException as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'foo' is not found"))

    def test_unknown_model_version(self):
        try:
            for pair in [("localhost:8000", "http"),
                         ("localhost:8001", "grpc")]:
                model_name = "graphdef_int32_int8_int8"
                if pair[1] == "http":
                    triton_client = httpclient.InferenceServerClient(
                        url=pair[0], verbose=True)
                else:
                    triton_client = grpcclient.InferenceServerClient(
                        url=pair[0], verbose=True)

                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                model_metadata = triton_client.get_model_metadata(
                    model_name, model_version="99")
                self.assertTrue(False, "expected unknown model version failure")
        except InferenceServerException as ex:
            self.assertTrue(ex.message().startswith(
                "Request for unknown model: 'graphdef_int32_int8_int8' version 99 is not found"
            ))

    def test_model_latest_infer(self):
        input_size = 16
        tensor_shape = (1, input_size)
        platform_name = {
            'graphdef': 'tensorflow_graphdef',
            'onnx': 'onnxruntime_onnx'
        }

        # There are 3 versions of *_int32_int32_int32 and all
        # should be available.
        for platform in ('graphdef', 'onnx'):
            model_name = platform + "_int32_int32_int32"

            # Initially there should be no version stats..
            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    model_metadata = triton_client.get_model_metadata(
                        model_name)
                    # verify all versions are reported when no model version is specified
                    if pair[1] == "http":
                        self.assertEqual(model_name, model_metadata['name'])
                        self.assertEqual(len(model_metadata['versions']), 3)
                        for v in (1, 2, 3):
                            self.assertIn(str(v), model_metadata['versions'])
                    else:
                        self.assertEqual(model_name, model_metadata.name)
                        self.assertEqual(len(model_metadata.versions), 3)
                        for v in (1, 2, 3):
                            self.assertIn(str(v), model_metadata.versions)

                    # verify contents of model metadata
                    if pair[1] == "http":
                        model_platform = model_metadata['platform']
                        model_inputs = model_metadata['inputs']
                        model_outputs = model_metadata['outputs']
                    else:
                        model_platform = model_metadata.platform
                        model_inputs = model_metadata.inputs
                        model_outputs = model_metadata.outputs

                    self.assertEqual(platform_name[platform], model_platform)
                    self.assertEqual(len(model_inputs), 2)
                    self.assertEqual(len(model_outputs), 2)

                    for model_input in model_inputs:
                        if pair[1] == "http":
                            input_dtype = model_input['datatype']
                            input_shape = model_input['shape']
                            input_name = model_input['name']
                        else:
                            input_dtype = model_input.datatype
                            input_shape = model_input.shape
                            input_name = model_input.name
                        self.assertIn(input_name, ["INPUT0", "INPUT1"])
                        self.assertEqual(input_dtype, "INT32")
                        self.assertEqual(input_shape, [-1, 16])

                    for model_output in model_outputs:
                        if pair[1] == "http":
                            output_dtype = model_output['datatype']
                            output_shape = model_output['shape']
                            output_name = model_output['name']
                        else:
                            output_dtype = model_output.datatype
                            output_shape = model_output.shape
                            output_name = model_output.name
                        self.assertIn(output_name, ["OUTPUT0", "OUTPUT1"])
                        self.assertEqual(output_dtype, "INT32")
                        self.assertEqual(output_shape, [-1, 16])

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Infer using latest version (which is 3)...
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           model_version=None,
                           swap=True)

            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    for v in (1, 2, 3):
                        self.assertTrue(
                            triton_client.is_model_ready(model_name,
                                                         model_version=str(v)))

                    # Only version 3 should have infer stats
                    infer_stats = triton_client.get_inference_statistics(
                        model_name)
                    if pair[1] == "http":
                        stats = infer_stats['model_stats']
                    else:
                        stats = infer_stats.model_stats
                    self.assertEqual(
                        len(stats), 3,
                        "expected 3 infer stats for model " + model_name)
                    for s in stats:
                        if pair[1] == "http":
                            v = s['version']
                            stat = s['inference_stats']
                        else:
                            v = s.version
                            stat = s.inference_stats

                        if v == "3":
                            if pair[1] == "http":
                                self.assertTrue(stat['success']['count'], 3)
                            else:
                                self.assertTrue(stat.success.count, 3)
                        else:
                            if pair[1] == "http":
                                self.assertEqual(
                                    stat['success']['count'], 0,
                                    "unexpected infer success counts for version "
                                    + str(v) + " of model " + model_name)
                            else:
                                self.assertEqual(
                                    stat.success.count, 0,
                                    "unexpected infer success counts for version "
                                    + str(v) + " of model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_specific_infer(self):
        input_size = 16

        # There are 3 versions of *_float32_float32_float32 but only
        # versions 1 and 3 should be available.
        for platform in ('graphdef', 'onnx', 'plan'):
            tensor_shape = (1, input_size)
            model_name = platform + "_float32_float32_float32"

            # Initially there should be no version status...
            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(
                        triton_client.is_model_ready(model_name,
                                                     model_version="1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name,
                                                     model_version="2"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name,
                                                     model_version="3"))
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            # Infer using version 1...
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           model_version=1,
                           swap=False)

            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(
                        triton_client.is_model_ready(model_name,
                                                     model_version="1"))
                    self.assertFalse(
                        triton_client.is_model_ready(model_name,
                                                     model_version="2"))
                    self.assertTrue(
                        triton_client.is_model_ready(model_name,
                                                     model_version="3"))

                    # Only version 1 should have infer stats
                    infer_stats = triton_client.get_inference_statistics(
                        model_name, model_version='1')
                    if pair[1] == "http":
                        self.assertEqual(
                            len(infer_stats['model_stats']), 1,
                            "expected 1 infer stats for version 1"
                            " of model " + model_name)
                        stats = infer_stats['model_stats'][0]['inference_stats']
                        self.assertTrue(stats['success']['count'], 3)
                    else:
                        self.assertEqual(
                            len(infer_stats.model_stats), 1,
                            "expected 1 infer stats for version 1"
                            " of model " + model_name)
                        stats = infer_stats.model_stats[0].inference_stats
                        self.assertTrue(stats.success.count, 3)
                    infer_stats = triton_client.get_inference_statistics(
                        model_name, model_version='3')
                    if pair[1] == "http":
                        stats = infer_stats['model_stats'][0]['inference_stats']
                        self.assertEqual(
                            stats['success']['count'], 0,
                            "unexpected infer stats for version 3"
                            " of model " + model_name)
                    else:
                        stats = infer_stats.model_stats[0].inference_stats
                        self.assertEqual(
                            stats.success.count, 0,
                            "unexpected infer stats for version 3"
                            " of model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


class ModelMetadataTest(tu.TestResultCollector):
    '''
    These tests must be run after the ServerMetadataTest. See test.sh
    file for correct test running.
    '''

    def test_model_versions_deleted(self):
        # Originally There were 3 versions of *_int32_int32_int32 and
        # version 3 was executed once. Version 2 and 3 models were
        # deleted from the model repository so now only expect version 1 to
        # be ready and show stats.
        for platform in ('graphdef', 'onnx'):
            model_name = platform + "_int32_int32_int32"

            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    model_metadata = triton_client.get_model_metadata(
                        model_name)
                    if pair[1] == "http":
                        self.assertEqual(model_name, model_metadata['name'])
                        self.assertEqual(len(model_metadata['versions']), 1)
                        self.assertEqual("1", model_metadata['versions'][0])
                    else:
                        self.assertEqual(model_name, model_metadata.name)
                        self.assertEqual(len(model_metadata.versions), 1)
                        self.assertEqual("1", model_metadata.versions[0])

                    # Only version 3 should have infer stats, only 1 is ready
                    for v in (1, 2, 3):
                        if v == 1:
                            self.assertTrue(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))
                            infer_stats = triton_client.get_inference_statistics(
                                model_name, model_version=str(v))
                            if pair[1] == "http":
                                self.assertEqual(
                                    len(infer_stats['model_stats']), 1,
                                    "expected 1 infer stats for version " +
                                    str(v) + " of model " + model_name)
                                stats = infer_stats['model_stats'][0][
                                    'inference_stats']
                                self.assertEqual(stats['success']['count'], 0)
                            else:
                                self.assertEqual(
                                    len(infer_stats.model_stats), 1,
                                    "expected 1 infer stats for version " +
                                    str(v) + " of model " + model_name)
                                stats = infer_stats.model_stats[
                                    0].inference_stats
                                self.assertEqual(stats.success.count, 0)

                        else:
                            self.assertFalse(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_versions_added(self):
        # Originally There was version 1 of *_float16_float32_float32.
        # Version 7 was added so now expect just version 7 to be ready
        # and provide infer stats.
        for platform in ('graphdef',):
            model_name = platform + "_float16_float32_float32"

            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    model_metadata = triton_client.get_model_metadata(
                        model_name)
                    if pair[1] == "http":
                        self.assertEqual(
                            model_name, model_metadata['name'],
                            "expected status for model " + model_name)
                        self.assertEqual(
                            len(model_metadata['versions']), 1,
                            "expected status for 1 versions for model " +
                            model_name)
                        self.assertEqual("7", model_metadata['versions'][0])
                    else:
                        self.assertEqual(
                            model_name, model_metadata.name,
                            "expected status for model " + model_name)
                        self.assertEqual(
                            len(model_metadata.versions), 1,
                            "expected status for 1 versions for model " +
                            model_name)
                        self.assertEqual("7", model_metadata.versions[0])

                    # Only version 7 should be ready and show infer stat.
                    for v in (1, 7):
                        if v == 7:
                            self.assertTrue(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))
                            infer_stats = triton_client.get_inference_statistics(
                                model_name, model_version=str(v))
                            if pair[1] == "http":
                                stats = infer_stats['model_stats'][0][
                                    'inference_stats']
                                self.assertEqual(
                                    stats['success']['count'], 0,
                                    "unexpected infer stats for version " +
                                    str(v) + " of model " + model_name)
                            else:
                                stats = infer_stats.model_stats[
                                    0].inference_stats
                                self.assertEqual(
                                    stats.success.count, 0,
                                    "unexpected infer stats for version " +
                                    str(v) + " of model " + model_name)

                        else:
                            self.assertFalse(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))
                            try:
                                infer_stats = triton_client.get_inference_statistics(
                                    model_name, model_version=str(v))
                                self.assertTrue(
                                    False,
                                    "unexpected infer stats for the model that is not ready"
                                )
                            except InferenceServerException as ex:
                                self.assertIn(
                                    "requested model version is not available for model",
                                    str(ex))

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_infer_stats_no_model_version(self):
        # Originally There were 3 versions of *_int32_int32_int32 and
        # version 3 was executed once. Version 2 and 3 models were
        # deleted from the model repository so now only expect version 1 to
        # be ready and show infer stats.
        for platform in ('graphdef', 'onnx'):
            model_name = platform + "_int32_int32_int32"

            try:
                for pair in [("localhost:8000", "http"),
                             ("localhost:8001", "grpc")]:
                    if pair[1] == "http":
                        triton_client = httpclient.InferenceServerClient(
                            url=pair[0], verbose=True)
                    else:
                        triton_client = grpcclient.InferenceServerClient(
                            url=pair[0], verbose=True)

                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    model_metadata = triton_client.get_model_metadata(
                        model_name)
                    if pair[1] == "http":
                        self.assertEqual(model_name, model_metadata['name'])
                        self.assertEqual(len(model_metadata['versions']), 1)
                        self.assertEqual("1", model_metadata['versions'][0])
                    else:
                        self.assertEqual(model_name, model_metadata.name)
                        self.assertEqual(len(model_metadata.versions), 1)
                        self.assertEqual("1", model_metadata.versions[0])

                    # Only version 3 should have infer stats, only 1 is ready
                    for v in (1, 2, 3):
                        if v == 1:
                            self.assertTrue(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))
                        else:
                            self.assertFalse(
                                triton_client.is_model_ready(
                                    model_name, model_version=str(v)))

                    infer_stats = triton_client.get_inference_statistics(
                        model_name)
                    if pair[1] == "http":
                        stats = infer_stats['model_stats']
                    else:
                        stats = infer_stats.model_stats
                    self.assertEqual(
                        len(stats), 1,
                        "expected 1 infer stats for model " + model_name)

                    if pair[1] == "http":
                        version = stats[0]['version']
                        stat = stats[0]['inference_stats']
                    else:
                        version = stats[0].version
                        stat = stats[0].inference_stats

                    if version != "1":
                        self.assertTrue(
                            False,
                            "expected version 1 for infer stat, got " + version)
                    else:
                        if pair[1] == "http":
                            self.assertEqual(
                                stat['success']['count'], 0,
                                "unexpected infer stats for version " +
                                str(version) + " of model " + model_name)
                        else:
                            self.assertEqual(
                                stat.success.count, 0,
                                "unexpected infer stats for version " +
                                str(version) + " of model " + model_name)

            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_infer_stats_no_model(self):
        # Test get_inference_statistics when no model/model_version is passed.
        try:
            for pair in [("localhost:8000", "http"),
                         ("localhost:8001", "grpc")]:
                if pair[1] == "http":
                    triton_client = httpclient.InferenceServerClient(
                        url=pair[0], verbose=True)
                else:
                    triton_client = grpcclient.InferenceServerClient(
                        url=pair[0], verbose=True)

                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                # Returns infer stats for ALL models + ready versions
                infer_stats = triton_client.get_inference_statistics()
                if pair[1] == "http":
                    stats = infer_stats['model_stats']
                else:
                    stats = infer_stats.model_stats
                self.assertEqual(
                    len(stats), 207,
                    "expected 207 infer stats for all ready versions of all model"
                )

        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
