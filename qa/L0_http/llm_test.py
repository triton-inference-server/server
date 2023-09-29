#!/usr/bin/python3
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

import json
import unittest

import requests
import test_util as tu


class HttpTest(tu.TestResultCollector):
    def _get_infer_url(self, model_name, route):
        return f"http://localhost:8000/v2/models/{model_name}/{route}"

    def _simple_infer(self, model_name, inputs, expected_outputs):
        headers = {"Content-Type": "application/json"}
        url = self._get_infer_url(model_name, "infer")
        r = requests.post(url, data=json.dumps(inputs), headers=headers)
        r.raise_for_status()

        content = r.json()
        print(content)

        self.assertEqual(content["model_name"], model_name)
        self.assertIn("outputs", content)
        self.assertEqual(content["outputs"], expected_outputs)

    def _simple_generate_stream(self, model_name, inputs, expected_outputs):
        import sseclient

        headers = {"Accept": "text/event-stream"}
        url = self._get_infer_url(model_name, "generate_stream")
        # stream=True used to indicate response can be iterated over
        r = requests.post(url, data=json.dumps(inputs), headers=headers, stream=True)

        # Validate SSE format
        print(r.headers)
        self.assertIn("Content-Type", r.headers)
        # FIXME: Clarify correct header here.
        # self.assertEqual(r.headers['Content-Type'], 'text/event-stream')
        self.assertEqual(r.headers["Content-Type"], "text/event-stream; charset=utf-8")

        # SSE format (data: []) is hard to parse, use helper library for simplicity
        client = sseclient.SSEClient(r)
        tokens = []
        for i, event in enumerate(client.events()):
            # End of event stream
            if event.data == "[DONE]":
                continue

            # Parse event data, join events into a single response
            data = json.loads(event.data)
            print(f"Event {i}:", data)
            if "TEXT" not in data:
                print("FIXME: EXPECTED OUTPUT FIELD NOT FOUND")
            else:
                tokens.append(data["TEXT"])
        print("TOKENS:", tokens)

    def test_infer(self):
        model_name = "onnx_zero_1_object"
        parameters = {}

        # Setup text-based input
        input0_data = ["hello"]
        input0 = {
            "name": "INPUT0",
            "datatype": "BYTES",
            "shape": [1, 1],
            "data": input0_data,
        }
        inputs = {"inputs": [input0], "parameters": parameters}
        # Identity model, output should match input
        expected_outputs = [
            {
                "name": "OUTPUT0",
                "datatype": "BYTES",
                "shape": [1, 1],
                "data": input0_data,
            }
        ]
        self._simple_infer(model_name, inputs, expected_outputs)

    # def test_generate(self):
    #    pass

    def test_generate_stream(self):
        # TODO: vllm
        model_name = "onnx_zero_1_object"
        parameters = {}
        # Setup text-based input
        input0_data = ["hello"]
        inputs = {"prompt": input0_data, "stream": True, "parameters": parameters}
        # Identity model, output should match input
        expected_outputs = [
            {
                "name": "OUTPUT0",
                "datatype": "BYTES",
                "shape": [1, 1],
                "data": input0_data,
            }
        ]
        self._simple_generate_stream(model_name, inputs, expected_outputs)


if __name__ == "__main__":
    unittest.main()
