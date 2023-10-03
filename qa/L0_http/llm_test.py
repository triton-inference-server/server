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

    def test_generate(self):
        model_name = "vllm_proxy"
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": text, "STREAM": False}

        url = self._get_infer_url(model_name, "generate")
        # stream=True used to indicate response can be iterated over
        r = requests.post(url, data=json.dumps(inputs))

        r.raise_for_status()

        self.assertIn("Content-Type", r.headers)
        self.assertIn("application/json", r.headers["Content-Type"])

        data = r.json()
        self.assertIn("TEXT", data)
        self.assertEqual(text, data["TEXT"])

    def test_generate_stream(self):
        model_name = "vllm_proxy"
        # Setup text-based input
        text = "hello world"
        rep_count = 3
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": rep_count}

        import sseclient

        headers = {"Accept": "text/event-stream"}
        url = self._get_infer_url(model_name, "generate_stream")
        # stream=True used to indicate response can be iterated over
        r = requests.post(url, data=json.dumps(inputs), headers=headers, stream=True)

        r.raise_for_status()

        # Validate SSE format
        self.assertIn("Content-Type", r.headers)
        self.assertIn("text/event-stream", r.headers["Content-Type"])

        # SSE format (data: []) is hard to parse, use helper library for simplicity
        client = sseclient.SSEClient(r)
        res_count = 0
        for i, event in enumerate(client.events()):
            # Parse event data, join events into a single response
            data = json.loads(event.data)
            self.assertIn("TEXT", data)
            self.assertEqual(text, data["TEXT"])
            res_count += 1
        self.assertTrue(rep_count, res_count)


if __name__ == "__main__":
    unittest.main()
