#!/usr/bin/python
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import shutil
import time
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import tritonclient.http as httpclient

import argparse
import csv
import json
import os
import requests
import socket
import sys


class SageMakerMultiModelTest(tu.TestResultCollector):

    def setUp(self):

        SAGEMAKER_BIND_TO_PORT = os.getenv("SAGEMAKER_BIND_TO_PORT", "8080")
        self.url_mme_ = "http://localhost:{}/models".format(
            SAGEMAKER_BIND_TO_PORT)

        # model_1 setup
        self.model1_name = "sm_mme_model_1"
        self.model1_url = "/opt/ml/models/123456789abcdefghi/model"

        self.model1_input_data_ = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]
        self.model1_expected_output0_data_ = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
        ]
        self.model1_expected_output1_data_ = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.model1_expected_result_ = {
            "model_name":
                "sm_mme_model_1",
            "model_version":
                "1",
            "outputs": [
                {
                    "name": "OUTPUT0",
                    "datatype": "INT32",
                    "shape": [1, 16],
                    "data": self.model1_expected_output0_data_
                },
                {
                    "name": "OUTPUT1",
                    "datatype": "INT32",
                    "shape": [1, 16],
                    "data": self.model1_expected_output1_data_
                },
            ],
        }

        # model_2 setup
        self.model2_name = "sm_mme_model_2"
        self.model2_url = "/opt/ml/models/987654321ihgfedcba/model"

        # Output is same as input since this is an identity model
        self.model2_input_data_ = [0, 1, 2, 3, 4, 5, 6, 7]

    def test_sm_0_environment_variables_set(self):
        self.assertEqual(os.getenv("SAGEMAKER_MULTI_MODEL"), "true",
                         "Variable SAGEMAKER_MULTI_MODEL must be set to true")

    def test_sm_1_model_load(self):
        # Load model_1
        request_body = {"model_name": self.model1_name, "url": self.model1_url}
        headers = {"Content-Type": "application/json"}
        r = requests.post(self.url_mme_,
                          data=json.dumps(request_body),
                          headers=headers)
        time.sleep(5)  # wait for model to load
        self.assertEqual(
            r.status_code, 200,
            "Expected status code 200, received {}".format(r.status_code))

        # Load the same model again, expect a 409
        request_body = {"model_name": self.model1_name, "url": self.model1_url}
        headers = {"Content-Type": "application/json"}
        r = requests.post(self.url_mme_,
                          data=json.dumps(request_body),
                          headers=headers)
        time.sleep(5)  # wait for model to load
        self.assertEqual(
            r.status_code, 409,
            "Expected status code 409, received {}".format(r.status_code))

        # Load model_2
        request_body = {"model_name": self.model2_name, "url": self.model2_url}
        headers = {"Content-Type": "application/json"}
        r = requests.post(self.url_mme_,
                          data=json.dumps(request_body),
                          headers=headers)
        time.sleep(5)  # wait for model to load
        self.assertEqual(
            r.status_code, 200,
            "Expected status code 200, received {}".format(r.status_code))

    def test_sm_2_model_list(self):
        r = requests.get(self.url_mme_)
        time.sleep(3)
        expected_response_1 = {
            "models": [
                {
                    "modelName": self.model1_name,
                    "modelUrl": self.model1_url
                },
                {
                    "modelName": self.model2_name,
                    "modelUrl": self.model2_url
                },
            ]
        }
        expected_response_2 = {
            "models": [
                {
                    "modelName": self.model2_name,
                    "modelUrl": self.model2_url
                },
                {
                    "modelName": self.model1_name,
                    "modelUrl": self.model1_url
                },
            ]
        }

        # Returned list response's order is not deterministic
        self.assertIn(
            r.json(),
            [expected_response_1, expected_response_2],
            "Expected one of {}, received: {}".format(
                [expected_response_1, expected_response_2], r.json()),
        )

    def test_sm_3_model_get(self):
        get_url = "{}/{}".format(self.url_mme_, self.model1_name)
        r = requests.get(get_url)
        time.sleep(3)
        expected_response = {
            "modelName": self.model1_name,
            "modelUrl": self.model1_url
        }
        self.assertEqual(
            r.json(), expected_response,
            "Expected response: {}, received: {}".format(
                expected_response, r.json()))

    def test_sm_4_model_invoke(self):
        # Invoke model_1
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
        inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.model1_input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput("OUTPUT0", binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput("OUTPUT1", binary_data=False))
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {"Content-Type": "application/json"}
        invoke_url = "{}/{}/invoke".format(self.url_mme_, self.model1_name)
        r = requests.post(invoke_url, data=request_body, headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.model1_expected_result_,
            r.json(),
            "Expected response : {}, received: {}".format(
                self.model1_expected_result_, r.json()),
        )

        # Invoke model_2
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput(
            "INPUT0",
            [1, 8],
            "FP32",
        ))
        input_data = np.array(self.model2_input_data_, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)

        outputs.append(
            httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))

        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        invoke_url = "{}/{}/invoke".format(self.url_mme_, self.model2_name)
        headers = {
            "Content-Type":
                "application/vnd.sagemaker-triton.binary+json;json-header-size={}"
                .format(header_length)
        }
        r = requests.post(invoke_url, data=request_body, headers=headers)

        header_length_prefix = "application/vnd.sagemaker-triton.binary+json;json-header-size="
        header_length_str = r.headers["Content-Type"][len(header_length_prefix
                                                         ):]
        result = httpclient.InferenceServerClient.parse_response_body(
            r._content, header_length=int(header_length_str))

        # Get the inference header size so we can locate the output binary data
        output_data = result.as_numpy("OUTPUT0")

        for i in range(8):
            self.assertEqual(output_data[0][i], input_data[0][i],
                             "Tensor Value Mismatch")

    def test_sm_5_model_unload(self):
        # Unload model_1
        unload_url = "{}/{}".format(self.url_mme_, self.model1_name)
        r = requests.delete(unload_url)
        time.sleep(3)
        self.assertEqual(
            r.status_code, 200,
            "Expected status code 200, received {}".format(r.status_code))

        # Unload model_2
        unload_url = "{}/{}".format(self.url_mme_, self.model2_name)
        r = requests.delete(unload_url)
        time.sleep(3)
        self.assertEqual(
            r.status_code, 200,
            "Expected status code 200, received {}".format(r.status_code))

        # Unload a non-loaded model, expect a 404
        unload_url = "{}/sm_non_loaded_model".format(self.url_mme_)
        r = requests.delete(unload_url)
        time.sleep(3)
        self.assertEqual(
            r.status_code, 404,
            "Expected status code 404, received {}".format(r.status_code))


if __name__ == "__main__":
    unittest.main()
