#!/usr/bin/python

# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import requests
import unittest
import test_util as tu


class OutputValidationTest(tu.TestResultCollector):
    # for datatype mismatch
    def test_datatype(self):
        url = 'http://localhost:8000/v2/models/libtorch_datatype_1_float32/infer'
        body = '{"inputs":[{"name":"INPUT__0","shape":[1,1],"datatype":"FP32","data":[1.0]}],"outputs":[{"name":"OUTPUT__0"}]}'
        response = requests.post(url, data=body)
        msg = response.json()["error"]
        self.assertTrue(
            msg.startswith(
                "configuration expects datatype TYPE_INT32 for output 'OUTPUT__0', model provides TYPE_FP32"
            ))

    # for output mismatch
    def test_index(self):
        url = 'http://localhost:8000/v2/models/libtorch_index_1_float32/infer'
        body = '{"inputs":[{"name":"INPUT__0","shape":[1,1],"datatype":"FP32","data":[1.0]}],"outputs":[{"name":"OUTPUT__1"}]}'
        response = requests.post(url, data=body)
        msg = response.json()["error"]
        self.assertTrue(
            msg.startswith(
                "The output OUTPUT__1 in the model configuration refers to an output index which doesn't exist. This model has 1 outputs"
            ))

    # successful run
    def test_success(self):
        url = 'http://localhost:8000/v2/models/libtorch_zero_1_float32/infer'
        body = '{"inputs":[{"name":"INPUT__0","shape":[1,1],"datatype":"FP32","data":[1.0]}],"outputs":[{"name":"OUTPUT__0"}]}'
        response = requests.post(url, data=body)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
