#!/usr/bin/python
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class VertexAiTest(tu.TestResultCollector):

    def setUp(self):
        port = os.getenv('AIP_HTTP_PORT', '8080')
        predict_endpoint = os.getenv('AIP_PREDICT_ROUTE', '/predict')
        self.model_ = os.getenv('TEST_EXPLICIT_MODEL_NAME', 'addsub')
        self.url_ = "http://localhost:{}{}".format(port, predict_endpoint)
        self.input_data_ = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]
        self.expected_output0_data_ = [x * 2 for x in self.input_data_]
        self.expected_output1_data_ = [0 for x in self.input_data_]

    def test_predict(self):
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

    def test_predict_specified_model(self):
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

        headers = {
            'Content-Type':
                'application/json',
            "X-Vertex-Ai-Triton-Redirect":
                "v2/models/{}/infer".format(self.model_)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        result = httpclient.InferenceServerClient.parse_response_body(
            r._content)

        output0_data = result.as_numpy('OUTPUT0')
        output1_data = result.as_numpy('OUTPUT1')
        if self.model_ == "addsub":
            expected_output0_data = [x * 2 for x in self.input_data_]
            expected_output1_data = [0 for x in self.input_data_]
        else:
            expected_output0_data = [0 for x in self.input_data_]
            expected_output1_data = [x * 2 for x in self.input_data_]
        for i in range(16):
            self.assertEqual(output0_data[0][i], expected_output0_data[i])
            self.assertEqual(output1_data[0][i], expected_output1_data[i])

    def test_predict_request_binary(self):
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
                'application/vnd.vertex-ai-triton.binary+json;json-header-size={}'
                .format(header_length)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        result = httpclient.InferenceServerClient.parse_response_body(
            r._content)
        output0_data = result.as_numpy('OUTPUT0')
        output1_data = result.as_numpy('OUTPUT1')
        for i in range(16):
            self.assertEqual(output0_data[0][i], self.expected_output0_data_[i])
            self.assertEqual(output1_data[0][i], self.expected_output1_data_[i])

    def test_predict_response_binary(self):
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

        header_length_str = r.headers['Inference-Header-Content-Length']
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
                'additional-string/application/vnd.vertex-ai-triton.binary+json;json-header-size={}'
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
                'application/vnd.vertex-ai-triton.binary+json;json-header-size=additional-string{}'
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
                'application/vnd.vertex-ai-triton.binary+json;json-header-size=-123'
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
                'application/vnd.vertex-ai-triton.binary+json;json-header-size=12345'
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))


if __name__ == '__main__':
    unittest.main()
