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
import sseclient
import test_util as tu


class SageMakerGenerateStreamTest(tu.TestResultCollector):
    def setUp(self):
        SAGEMAKER_BIND_TO_PORT = os.getenv("SAGEMAKER_BIND_TO_PORT", "8080")
        self.url_ = "http://localhost:{}/invocations".format(SAGEMAKER_BIND_TO_PORT)

    def generate_stream(self, inputs, stream=False):
        headers = {"Accept": "text/event-stream"}
        # stream=True used to indicate response can be iterated over, which
        # should be the common setting for generate_stream.
        # For correctness test case, stream=False so that we can re-examine
        # the response content.
        return requests.post(
            self.url_,
            data=inputs if isinstance(inputs, str) else json.dumps(inputs),
            headers=headers,
            stream=stream,
        )

    def generate_stream_expect_success(self, inputs, expected_output, rep_count):
        r = self.generate_stream(inputs)
        r.raise_for_status()
        self.check_sse_responses(r, [{"TEXT": expected_output}] * rep_count)

    def check_sse_responses(self, res, expected_res):
        # Validate SSE format
        self.assertIn("Content-Type", res.headers)
        self.assertEqual(
            "text/event-stream; charset=utf-8", res.headers["Content-Type"]
        )

        # SSE format (data: []) is hard to parse, use helper library for simplicity
        client = sseclient.SSEClient(res)
        res_count = 0
        for event in client.events():
            # Parse event data, join events into a single response
            data = json.loads(event.data)
            for key, value in expected_res[res_count].items():
                self.assertIn(key, data)
                self.assertEqual(value, data[key])
            res_count += 1
        self.assertEqual(len(expected_res), res_count)
        # Make sure there is no message in the wrong form
        for remaining in client._read():
            self.assertTrue(
                remaining.startswith(b"data:"),
                f"SSE response not formed properly, got: {remaining}",
            )
            self.assertTrue(
                remaining.endswith(b"\n\n"),
                f"SSE response not formed properly, got: {remaining}",
            )

    def test_generate_stream(self):
        # Setup text-based input
        text = "hello world"
        rep_count = 3
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": rep_count}
        self.generate_stream_expect_success(inputs, text, rep_count)


if __name__ == "__main__":
    unittest.main()
