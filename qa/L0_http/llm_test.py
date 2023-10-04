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
import sseclient
import test_util as tu


class HttpTest(tu.TestResultCollector):
    def _get_infer_url(self, model_name, route):
        return f"http://localhost:8000/v2/models/{model_name}/{route}"

    def generate_stream(self, model_name, inputs):
        headers = {"Accept": "text/event-stream"}
        url = self._get_infer_url(model_name, "generate_stream")
        # stream=True used to indicate response can be iterated over
        # r = requests.post(url, data=json.dumps(inputs), headers=headers, stream=True)
        return requests.post(
            url,
            data=inputs if isinstance(inputs, str) else json.dumps(inputs),
            headers=headers,
            stream=False,
        )

    def generate_expect_failure(self, model_name, inputs, msg):
        url = self._get_infer_url(model_name, "generate")
        r = requests.post(
            url, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )
        try:
            r.raise_for_status()
            self.assertTrue(False, f"Expected failure, success for {inputs}")
        except requests.exceptions.HTTPError as e:
            self.assertIn(msg, r.json()["error"])

    def generate_stream_expect_failure(self, model_name, inputs, msg):
        r = self.generate_stream(model_name, inputs)
        try:
            r.raise_for_status()
            self.assertTrue(False, f"Expected failure, success for {inputs}")
        except requests.exceptions.HTTPError as e:
            self.assertIn(msg, r.json()["error"])

    def generate_stream_expect_success(
        self, model_name, inputs, expected_output, rep_count
    ):
        r = self.generate_stream(model_name, inputs)
        r.raise_for_status()
        self.check_sse_responses(r, [{"TEXT": expected_output}] * rep_count)

    def check_sse_responses(self, res, expected_res):
        # Validate SSE format
        self.assertIn("Content-Type", res.headers)
        self.assertIn("text/event-stream", res.headers["Content-Type"])

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
        self.assertTrue(len(expected_res), res_count)
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

    def test_generate(self):
        model_name = "vllm_proxy"
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": text, "STREAM": False}

        url = self._get_infer_url(model_name, "generate")
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
        self.generate_stream_expect_success(model_name, inputs, text, rep_count)

    def test_missing_inputs(self):
        model_name = "vllm_proxy"
        missing_all_inputs = [
            # Missing all inputs
            {},
            {"abc": 123},
        ]
        missing_one_input = [
            # Missing 1 input
            {"PROMPT": "hello"},
            {"STREAM": False},
            {"STREAM": False, "other": "param"},
        ]
        for inputs in missing_all_inputs:
            self.generate_expect_failure(
                model_name, inputs, "expected 2 inputs but got 0"
            )
            self.generate_stream_expect_failure(
                model_name, inputs, "expected 2 inputs but got 0"
            )

        for inputs in missing_one_input:
            self.generate_expect_failure(
                model_name, inputs, "expected 2 inputs but got 1"
            )
            self.generate_stream_expect_failure(
                model_name, inputs, "expected 2 inputs but got 1"
            )

    def test_invalid_input_types(self):
        model_name = "vllm_proxy"
        invalid_bool = "attempt to access JSON non-boolean as boolean"
        invalid_string = "attempt to access JSON non-string as string"
        invalid_type_inputs = [
            # Prompt bad type
            ({"PROMPT": 123, "STREAM": False}, invalid_string),
            # Stream bad type
            ({"PROMPT": "hello", "STREAM": "false"}, invalid_bool),
            # Both bad type, parsed in order
            ({"PROMPT": True, "STREAM": 123}, invalid_string),
            ({"STREAM": 123, "PROMPT": True}, invalid_bool),
        ]

        for inputs, error_msg in invalid_type_inputs:
            self.generate_expect_failure(model_name, inputs, error_msg)
            self.generate_stream_expect_failure(model_name, inputs, error_msg)

    def test_duplicate_inputs(self):
        model_name = "vllm_proxy"
        dupe_prompt = "input 'PROMPT' already exists in request"
        dupe_stream = "input 'STREAM' already exists in request"
        # Use JSON string directly as Python Dict doesn't support duplicate keys
        invalid_type_inputs = [
            # One duplicate
            (
                '{"PROMPT": "hello", "STREAM": false, "PROMPT": "duplicate"}',
                dupe_prompt,
            ),
            ('{"PROMPT": "hello", "STREAM": false, "STREAM": false}', dupe_stream),
            # Multiple duplicates, parsed in order
            (
                '{"PROMPT": "hello", "STREAM": false, "PROMPT": "duplicate", "STREAM": true}',
                dupe_prompt,
            ),
            (
                '{"PROMPT": "hello", "STREAM": false, "STREAM": true, "PROMPT": "duplicate"}',
                dupe_stream,
            ),
        ]
        for inputs, error_msg in invalid_type_inputs:
            self.generate_expect_failure(model_name, inputs, error_msg)
            self.generate_stream_expect_failure(model_name, inputs, error_msg)

    def test_generate_stream_response_error(self):
        model_name = "vllm_proxy"
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": 0, "FAIL_LAST": True}
        r = self.generate_stream(model_name, inputs)

        # With "REPETITION": 0, error will be first response and the HTTP code
        # will be set properly
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.check_sse_responses(r, [{"error": "An Error Occurred"}])

        # With "REPETITION" > 0, the first response is valid response and set
        # HTTP code to success, so user must validate each response
        inputs["REPETITION"] = 1
        r = self.generate_stream(model_name, inputs)
        r.raise_for_status()

        self.check_sse_responses(r, [{"TEXT": text}, {"error": "An Error Occurred"}])

    # NOTE: The below is to reproduce a current segfault, not the real test.
    """
    # First call /generate_stream Stream=False
    curl -s -w "\n%{http_code}\n" -X POST localhost:8000/v2/models/vllm_proxy/generate_stream -d '{"PROMPT": "hi there", "STREAM": false}'

    # Then call /generate_stream Stream=True
    curl -s -w "\n%{http_code}\n" -X POST localhost:8000/v2/models/vllm_proxy/generate_stream -d '{"PROMPT": "hi there", "STREAM": true}'

    Thread 53 "tritonserver" received signal SIGSEGV, Segmentation fault.
    [Switching to Thread 0x7fd11e7fc000 (LWP 3456520)]
    0x000055882d952886 in evbuffer_get_length (buffer=0x0) at /mnt/triton/jira/llm_rest/server/build/_deps/repo-third-party-build/libevent/src/libevent/buffer.c:610
    610             EVBUFFER_LOCK(buffer);
    (gdb) bt
    #0  0x000055882d952886 in evbuffer_get_length (buffer=0x0)
        at /mnt/triton/jira/llm_rest/server/build/_deps/repo-third-party-build/libevent/src/libevent/buffer.c:610
    #1  0x000055882d94935b in evhtp_send_reply_chunk (request=0x7fd10c005f50, buf=0x0)
        at /mnt/triton/jira/llm_rest/server/build/_deps/repo-third-party-src/libevhtp/libevhtp/evhtp.c:3969
    #2  0x000055882d902d27 in triton::server::HTTPAPIServer::GenerateRequestClass::ChunkResponseCallback (thr=0x55882ffc8de0,
        arg=0x7fd10c003810, shared=0x55882ffc8cc0) at /mnt/triton/jira/llm_rest/server/src/http_server.cc:4003
    """

    def test_segfault(self):
        model_name = "vllm_proxy"
        # NOTE with curl on CLI, it seems to segfault consistently, but with
        # requests.post() in this script, it seems intermittent and may require re-run
        input1 = {"PROMPT": "hello", "STREAM": False, "param": "segfault"}
        input2 = {"PROMPT": "hello", "STREAM": True, "param": "segfault"}
        r1 = self.generate_stream(model_name, input1)
        r1.raise_for_status()
        # FIXME: Causes segfault when executed in this order False -> True
        r2 = self.generate_stream(model_name, input2)
        r2.raise_for_status()

        self.assertTrue(
            False,
            "This segfaults consistently with curl, seeems intermittent with requests.post()",
        )

    # TODO
    def test_non_decoupled(self):
        # OK for /generate
        # OK for /generate_stream
        self.assertTrue(False, "Not implemented yet")

    # TODO
    def test_decoupled_one_response(self):
        # OK for /generate
        # OK for /generate_stream
        self.assertTrue(False, "Not implemented yet")

    # TODO
    def test_decoupled_zero_response(self):
        # FAIL for /generate
        # OK for /generate_stream
        self.assertTrue(False, "Not implemented yet")

    # TODO
    def test_decoupled_many_response(self):
        # FAIL for /generate
        # OK for /generate_stream
        self.assertTrue(False, "Not implemented yet")

    # TODO: Use model from L0_parameters to verify some catch-all parameter
    # behavior for JSON keys not found in model config.
    def test_parameters(self):
        self.assertTrue(False, "Not implemented yet")

    # TODO: Test unsupported parameters like floats, nested objects, etc.
    def test_invalid_parameters(self):
        self.assertTrue(False, "Not implemented yet")


if __name__ == "__main__":
    unittest.main()
