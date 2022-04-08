#!/usr/bin/python

# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest
import test_util as tu
from mlflow.deployments import get_deploy_client
import json
import numpy as np


class PluginTest(tu.TestResultCollector):

    def setUp(self):
        self.client_ = get_deploy_client('triton')

    def test_onnx_flavor(self):
        # Log the ONNX model to MLFlow
        import mlflow.onnx
        import onnx
        model = onnx.load(
            "./mlflow-triton-plugin/examples/onnx_float32_int32_int32/1/model.onnx"
        )
        # Use a different name to ensure the plugin operates on correct model
        mlflow.onnx.log_model(model,
                              "triton",
                              registered_model_name="onnx_model")

        # create
        self.client_.create_deployment("onnx_model",
                                       "models:/onnx_model/1",
                                       flavor="onnx")

        # list
        deployment_list = self.client_.list_deployments()
        self.assertEqual(len(deployment_list), 1)
        self.assertEqual(deployment_list[0]['name'], "onnx_model")

        # get
        deployment = self.client_.get_deployment("onnx_model")
        self.assertEqual(deployment['name'], "onnx_model")

        # predict
        inputs = {}
        with open("./mlflow-triton-plugin/examples/input.json", "r") as f:
            input_json = json.load(f)
            for key, value in input_json['inputs'].items():
                inputs[key] = np.array(value, dtype=np.float32)

        output = self.client_.predict("onnx_model", inputs)
        with open("./mlflow-triton-plugin/examples/expected_output.json",
                  "r") as f:
            output_json = json.load(f)
            for key, value in output_json['outputs'].items():
                np.testing.assert_allclose(
                    output['outputs'][key],
                    np.array(value, dtype=np.int32),
                    err_msg='Inference result is not correct')

        # delete
        self.client_.delete_deployment("onnx_model")


if __name__ == '__main__':
    unittest.main()
