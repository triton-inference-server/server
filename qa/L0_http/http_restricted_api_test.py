#!/usr/bin/python
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

import sys

sys.path.append("../common")

import unittest

import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException


class RestrictedAPITest(unittest.TestCase):
    def setUp(self):
        self.model_name_ = "simple"
        self.client_ = tritonhttpclient.InferenceServerClient("localhost:8000")

    # Other unspecified APIs should not be restricted
    def test_sanity(self):
        self.client_.get_inference_statistics("simple")
        self.client_.get_inference_statistics(
            "simple", headers={"infer-key": "infer-value"}
        )

    # metadata, infer, model repository APIs are restricted.
    # metadata and infer expects "infer-key : infer-value" header,
    # model repository expected "admin-key : admin-value".
    def test_model_repository(self):
        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            self.client_.unload_model(
                self.model_name_, headers={"infer-key": "infer-value"}
            )
        # Request go through and get actual transaction error
        with self.assertRaisesRegex(
            InferenceServerException, "explicit model load / unload is not allowed"
        ):
            self.client_.unload_model(
                self.model_name_, headers={"admin-key": "admin-value"}
            )

    def test_metadata(self):
        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            self.client_.get_server_metadata()
        self.client_.get_server_metadata({"infer-key": "infer-value"})

    def test_infer(self):
        # setup
        inputs = [
            tritonhttpclient.InferInput("INPUT0", [1, 16], "INT32"),
            tritonhttpclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))

        # This test only care if the request goes through
        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            _ = self.client_.infer(
                model_name=self.model_name_, inputs=inputs, headers={"test": "1"}
            )
        self.client_.infer(
            model_name=self.model_name_,
            inputs=inputs,
            headers={"infer-key": "infer-value"},
        )


if __name__ == "__main__":
    unittest.main()
