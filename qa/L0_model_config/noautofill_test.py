#!/usr/bin/python
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class NoAutoFillTest(tu.TestResultCollector):
    def setUp(self):
        self._model_name = "noautofill_noconfig"
        self._triton_client = httpclient.InferenceServerClient("localhost:8000")

    def tearDown(self):
        self._triton_client.unload_model(self._model_name)

    def test_load_no_autofill_model_with_config(self):
        config = '{"max_batch_size":"16"}'
        self._triton_client.load_model(self._model_name, config=config)

        # Check if the model config is correct
        model_config = self._triton_client.get_model_config(self._model_name)
        self.assertEqual(model_config["max_batch_size"], 16)

    def test_load_no_autofill_model_with_no_config(self):
        with self.assertRaises(InferenceServerException) as ex:
            self._triton_client.load_model(self._model_name)
        self.assertIn("model configuration is not provided", str(ex.exception))


if __name__ == "__main__":
    unittest.main()
