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

import unittest
import numpy as np
import test_util as tu
import requests


class HttpTest(tu.TestResultCollector):

    def _get_infer_url(self, model_name):
        return "http://localhost:8000/v2/models/{}/infer".format(model_name)

    def test_raw_binary(self):
        # Select model that satisfies constraints for raw binary request
        model = "onnx_zero_1_float32"
        input = np.arange(8, dtype=np.float32)
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input.tobytes(),
                          headers=headers)
        r.raise_for_status()

        # Get the inference header size so we can locate the output binary data
        header_size = int(r.headers["Inference-Header-Content-Length"])
        self.assertEqual(
            input.tobytes(), r.content[header_size:],
            "Expected response body contains correct output binary data: {}; got: {}"
            .format(input.tobytes(), r.content[header_size:]))

    def test_raw_binary_longer(self):
        # Similar to test_raw_binary but test with different data size
        model = "onnx_zero_1_float32"
        input = np.arange(32, dtype=np.float32)
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input.tobytes(),
                          headers=headers)
        r.raise_for_status()

        # Get the inference header size so we can locate the output binary data
        header_size = int(r.headers["Inference-Header-Content-Length"])
        self.assertEqual(
            input.tobytes(), r.content[header_size:],
            "Expected response body contains correct output binary data: {}; got: {}"
            .format(input.tobytes(), r.content[header_size:]))

    def test_byte(self):
        # Select model that satisfies constraints for raw binary request
        # i.e. BYTE type the element count must be 1
        model = "onnx_zero_1_object_1_element"
        input = "427"
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input,
                          headers=headers)
        r.raise_for_status()

        # Get the inference header size so we can locate the output binary data
        header_size = int(r.headers["Inference-Header-Content-Length"])
        # Triton returns BYTES tensor with byte size prepended
        output = r.content[header_size + 4:].decode()
        self.assertEqual(
            input, output,
            "Expected response body contains correct output binary data: {}; got: {}"
            .format(input, output))

    def test_byte_too_many_elements(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. BYTE type the element count must be 1
        model = "onnx_zero_1_object"
        input = "427"
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input,
                          headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))
        self.assertIn(
            "For BYTE datatype raw input, the model must have input shape [1]",
            r.content.decode())

    def test_multi_variable_dimensions(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. this model has multiple variable-sized dimensions
        model = "onnx_zero_1_float16"
        input = np.ones([2, 2], dtype=np.float16)
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input.tobytes(),
                          headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))
        self.assertIn(
            "The shape of the raw input 'INPUT0' can not be deduced because there are more than one variable-sized dimension",
            r.content.decode())

    def test_multi_inputs(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. input count must be 1
        model = "onnx_zero_3_float32"
        # Use one numpy array, after tobytes() it can be seen as three inputs
        # each with 8 elements (this ambiguity is why this is not allowed)
        input = np.arange(24, dtype=np.float32)
        headers = {'Inference-Header-Content-Length': '0'}
        r = requests.post(self._get_infer_url(model),
                          data=input.tobytes(),
                          headers=headers)
        self.assertEqual(
            400, r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code))
        self.assertIn(
            "Raw request must only have 1 input to be deduced but got 3 inputs for model",
            r.content.decode())


if __name__ == '__main__':
    unittest.main()
