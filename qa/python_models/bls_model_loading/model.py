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

import time
import unittest

import numpy as np
import triton_python_backend_utils as pb_utils


class PBBLSModelLoadingTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "onnx_int32_int32_int32"

    def tearDown(self):
        # The unload call does not wait for the requested model to be fully
        # unloaded before returning.
        pb_utils.unload_model(self.model_name)
        # TODO: Make this more robust to wait until fully unloaded
        print("Sleep 30 seconds to make sure model finishes unloading...")
        time.sleep(30)
        print("Done sleeping.")

    def test_load_unload_model(self):
        self.assertFalse(pb_utils.is_model_ready(model_name=self.model_name))
        pb_utils.load_model(model_name=self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))
        pb_utils.unload_model(self.model_name)
        self.assertFalse(pb_utils.is_model_ready(self.model_name))

    def test_load_with_config_override(self):
        self.assertFalse(pb_utils.is_model_ready(self.model_name))
        pb_utils.load_model(self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))

        # Send the config with the wrong format
        wrong_config = '"parameters": {"config": {{"backend":"onnxruntime", "version_policy":{"specific":{"versions":[2]}}}}}'
        with self.assertRaises(pb_utils.TritonModelException):
            pb_utils.load_model(model_name=self.model_name, config=wrong_config)
        # The model should not be changed after a failed load model request
        for version in ["2", "3"]:
            self.assertTrue(
                pb_utils.is_model_ready(
                    model_name=self.model_name, model_version=version
                )
            )

        # Send the config with the correct format
        config = (
            '{"backend":"onnxruntime", "version_policy":{"specific":{"versions":[2]}}}'
        )
        pb_utils.load_model(self.model_name, config=config)
        # The model should be changed after a successful load model request
        self.assertTrue(pb_utils.is_model_ready(self.model_name, "2"))
        self.assertFalse(pb_utils.is_model_ready(self.model_name, "3"))

    def test_load_with_file_override(self):
        self.assertFalse(pb_utils.is_model_ready(self.model_name))
        pb_utils.load_model(self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))

        override_name = "override_model"
        config = '{"backend":"onnxruntime"}'
        with open("models/onnx_int32_int32_int32/3/model.onnx", "rb") as file:
            data = file.read()
        files = {"file:1/model.onnx": data}

        # Request to load the model with override file, should fail without
        # providing override config.
        with self.assertRaises(pb_utils.TritonModelException):
            pb_utils.load_model(self.model_name, "", files)

        # Request to load the model with override file and config in a different name
        pb_utils.load_model(model_name=override_name, config=config, files=files)
        # Sanity check that the model with original name is unchanged
        self.assertFalse(pb_utils.is_model_ready(self.model_name, "1"))
        self.assertTrue(pb_utils.is_model_ready(self.model_name, "3"))

        # Check the override model readiness
        self.assertTrue(pb_utils.is_model_ready(override_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(override_name, "3"))

        # Request to load the model with override file and config in original name
        pb_utils.load_model(self.model_name, config, files)
        # Check that the model with original name is changed
        self.assertTrue(pb_utils.is_model_ready(self.model_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(self.model_name, "3"))

        # Sanity check readiness of the different named model
        self.assertTrue(pb_utils.is_model_ready(override_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(override_name, "3"))


class TritonPythonModel:
    def initialize(self, args):
        # Run the unittest during initialization
        test = unittest.main("model", exit=False)
        self.result = test.result.wasSuccessful()

    def execute(self, requests):
        responses = []
        for _ in requests:
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0", np.array([self.result], dtype=np.float16)
                        )
                    ]
                )
            )
        return responses
