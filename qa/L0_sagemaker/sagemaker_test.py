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
from tritonclientutils import InferenceServerException

import argparse
import csv
import json
import os
import requests
import socket
import sys

FLAGS = None


def post_to_url(url, data):
    headers = {'Content-Type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=data, headers=headers)
    r.raise_for_status()


class SageMakerTest(tu.TestResultCollector):

    def setUp(self):
        self.url_ = "http://localhost:8080/invocations"
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
                "model",
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

    def test_inference_client_generated_body(self):
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
        request_body, _ = httpclient.InferenceServerClient.make_body(
            inputs, outputs=outputs)

        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.expected_result_, r.json(),
            "Expected response body: {}; got: {}".format(
                self.expected_result_, r.json()))

    def test_inference_client_generated_body_binary(self):
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
        request_body, header_length = httpclient.InferenceServerClient.make_body(
            inputs, outputs=outputs)

        headers = {
            'Content-Type':
                'application/application/vnd.sagemaker-triton.binary+json;json-header-size={}'
                .format(header_length)
        }
        r = requests.post(self.url_, data=request_body, headers=headers)
        r.raise_for_status()

        self.assertEqual(
            self.expected_result_, r.json(),
            "Expected response body: {}; got: {}".format(
                self.expected_result_, r.json()))


if __name__ == '__main__':
    unittest.main()
