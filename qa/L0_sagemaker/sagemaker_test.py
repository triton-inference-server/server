#!/usr/bin/python
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


class SageMakerTest(tu.TestResultCollector):

    def setUp(self):
        SAGEMAKER_BIND_TO_PORT = os.getenv('SAGEMAKER_BIND_TO_PORT', '8080')
        self.url_ = "http://localhost:{}/invocations".format(
            SAGEMAKER_BIND_TO_PORT)
        self.input_data_ = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]
        self.expected_output0_data_ = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
        ]
        self.expected_output1_data_ = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.expected_result_ = {
            "model_name":
                "sm_model",
            "model_version":
                "1",
            "outputs": [{
                "name": "OUTPUT0",
                "datatype": "INT32",
                "shape": [1, 16],
                "data": self.expected_output0_data_
            }, {
                "name": "OUTPUT1",
                "datatype": "INT32",
                "shape": [1, 16],
                "data": self.expected_output1_data_
            }]
        }

    def test_direct_inference(self):
        request = {
            "inputs": [{
                "name": "INPUT0",
                "datatype": "INT32",
                "shape": [1, 16],
                "data": self.input_data_
            }, {
                "name": "INPUT1",
                "datatype": "INT32",
                "shape": [1, 16],
                "data": self.input_data_
            }]
        }
        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.url_, data=json.dumps(request), headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.expected_result_, r.json(),
            "Expected response body: {}; got: {}".format(
                self.expected_result_, r.json()))

    def test_inference_client_generated_request(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.expected_result_, r.json(),
            "Expected response body: {}; got: {}".format(
                self.expected_result_, r.json()))

    def test_inference_client_generated_request_binary(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'application/vnd.sagemaker-triton.binary+json;json-header-size={}'
                .format(header_length)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.expected_result_, r.json(),
            "Expected response body: {}; got: {}".format(
                self.expected_result_, r.json()))

    def test_inference_client_generated_response(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        result = httpclient.InferenceServerClient.parse_response_body(
            r._content)

        output0_data = result.as_numpy('OUTPUT0')
        output1_data = result.as_numpy('OUTPUT1')
        for i in range(16):
            self.assertEqual(output0_data[0][i], self.expected_output0_data_[i])
            self.assertEqual(output1_data[0][i], self.expected_output1_data_[i])

    def test_inference_client_generated_response_binary(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        header_length_prefix = "application/vnd.sagemaker-triton.binary+json;json-header-size="
        header_length_str = r.headers['Content-Type'][len(header_length_prefix
                                                         ):]
        result = httpclient.InferenceServerClient.parse_response_body(
            r._content, header_length=int(header_length_str))

        output0_data = result.as_numpy('OUTPUT0')
        output1_data = result.as_numpy('OUTPUT1')
        for i in range(16):
            self.assertEqual(output0_data[0][i], self.expected_output0_data_[i])
            self.assertEqual(output1_data[0][i], self.expected_output1_data_[i])

    def test_malformed_binary_header(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'additional-string/application/vnd.sagemaker-triton.binary+json;json-header-size={}'
                .format(header_length)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))

    def test_malformed_binary_header_not_number(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'application/vnd.sagemaker-triton.binary+json;json-header-size=additional-string{}'
                .format(header_length)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))

    def test_malformed_binary_header_negative_number(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'application/vnd.sagemaker-triton.binary+json;json-header-size=-123'
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))

    def test_malformed_binary_header_large_number(self):
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Initialize the data
        input_data = np.array(self.input_data_, dtype=np.int32)
        input_data = np.expand_dims(input_data, axis=0)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)
        inputs[1].set_data_from_numpy(input_data, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'application/vnd.sagemaker-triton.binary+json;json-header-size=12345'
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))


if __name__ == '__main__':
    unittest.main()
