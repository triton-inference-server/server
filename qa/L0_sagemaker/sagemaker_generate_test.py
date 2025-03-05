#!/usr/bin/python
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import sys
import unittest

import requests
import test_util as tu


class SageMakerGenerateTest(tu.TestResultCollector):
    def setUp(self):
        SAGEMAKER_BIND_TO_PORT = os.getenv("SAGEMAKER_BIND_TO_PORT", "8080")
        self.url_ = "http://localhost:{}/invocations".format(SAGEMAKER_BIND_TO_PORT)

    def generate(self, inputs):
        return requests.post(
            self.url_, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )

    def test_generate(self):
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": text, "STREAM": False}

        r = self.generate(inputs)
        r.raise_for_status()

        self.assertIn("Content-Type", r.headers)
        self.assertEqual(r.headers["Content-Type"], "application/json")

        data = r.json()
        self.assertIn("TEXT", data)
        self.assertEqual(text, data["TEXT"])


if __name__ == "__main__":
    unittest.main()
