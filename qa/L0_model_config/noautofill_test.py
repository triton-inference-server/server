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
import argparse

sys.path.append("../common")

import unittest
import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class NoAutoFillTest(tu.TestResultCollector):

    def setUp(self):
        self._model_name = "noautofill_noconfig"
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-u',
            '--url',
            type=str,
            required=False,
            default='localhost:8000',
            help='Inference server URL. Default is localhost:8000.')
        parser.add_argument('-v',
                            '--verbose',
                            action="store_true",
                            required=False,
                            default=False,
                            help='Enable verbose output')
        FLAGS = parser.parse_args()
        try:
            self._triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit(1)

    def test_load_no_autofill_model_with_config(self):
        try:
            config = "{\"max_batch_size\":\"16\"}"
            self._triton_client.load_model(self._model_name, config=config)
        except InferenceServerException as e:
            print("Failed: unexpected error: ", e.message())
            sys.exit(1)

        # Check if the model config is correct
        model_config = self._triton_client.get_model_config(self._model_name)
        if model_config['max_batch_size'] != 16:
            print("Failed: expect max_batch_size = 16, got: {}".format(
                model_config['max_batch_size']))
            sys.exit(1)

    def test_load_no_autofill_model_with_no_config(self):
        try:
            self._triton_client.load_model(self._model_name)
        except InferenceServerException as e:
            if "unable to find the model configuration file" not in e.message():
                print("Failed: wrong error message", e.message())
                sys.exit(1)
        else:
            print("Failed: expect error occurs.")
            sys.exit(1)


if __name__ == '__main__':
    unittest.main()
