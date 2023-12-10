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
import threading
import time
import unittest

import requests
import sseclient
import test_util as tu


class GenerateEndpointTest(tu.TestResultCollector):
    def setUp(self):
        self._model_name = "mock_llm"

    def _get_infer_url(self, model_name, route):
        return f"http://localhost:8000/v2/models/{model_name}/{route}"

    def generate_stream(self, model_name, inputs, stream=False):
        headers = {"Accept": "text/event-stream"}
        url = self._get_infer_url(model_name, "generate_stream")
        # stream=True used to indicate response can be iterated over, which
        # should be the common setting for generate_stream.
        # For correctness test case, stream=False so that we can re-examine
        # the response content.
        return requests.post(
            url,
            data=inputs if isinstance(inputs, str) else json.dumps(inputs),
            headers=headers,
            stream=stream,
        )

    def generate(self, model_name, inputs):
        url = self._get_infer_url(model_name, "generate")
        return requests.post(
            url, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )

    def generate_expect_failure(self, model_name, inputs, msg):
        url = self._get_infer_url(model_name, "generate")
        r = requests.post(
            url, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )
        # Content-Type header should always be JSON for errors
        self.assertEqual(r.headers["Content-Type"], "application/json")

        try:
            r.raise_for_status()
            self.assertTrue(False, f"Expected failure, success for {inputs}")
        except requests.exceptions.HTTPError as e:
            self.assertIn(msg, r.json()["error"])

    def generate_stream_expect_failure(self, model_name, inputs, msg):
        r = self.generate_stream(model_name, inputs)
        # Content-Type header should always be JSON for errors
        self.assertEqual(r.headers["Content-Type"], "application/json")

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

    def test_generate(self):
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": text, "STREAM": False}

        r = self.generate(self._model_name, inputs)
        r.raise_for_status()

        self.assertIn("Content-Type", r.headers)
        self.assertEqual(r.headers["Content-Type"], "application/json")

        data = r.json()
        self.assertIn("TEXT", data)
        self.assertEqual(text, data["TEXT"])

    def test_generate_stream(self):
        # Setup text-based input
        text = "hello world"
        rep_count = 3
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": rep_count}
        self.generate_stream_expect_success(self._model_name, inputs, text, rep_count)

    def test_streaming(self):
        # verify the responses are streamed as soon as it is generated
        text = "hello world"
        rep_count = 3
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": rep_count, "DELAY": 2}
        past = time.time()
        res = self.generate_stream(self._model_name, inputs, stream=True)
        client = sseclient.SSEClient(res)
        # This test does not focus on event content
        for _ in client.events():
            now = time.time()
            self.assertTrue(1 < (now - past) < 3)
            past = now

    def test_missing_inputs(self):
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
                self._model_name, inputs, "expected 2 inputs but got 0"
            )
            self.generate_stream_expect_failure(
                self._model_name, inputs, "expected 2 inputs but got 0"
            )

        for inputs in missing_one_input:
            self.generate_expect_failure(
                self._model_name, inputs, "expected 2 inputs but got 1"
            )
            self.generate_stream_expect_failure(
                self._model_name, inputs, "expected 2 inputs but got 1"
            )

    def test_invalid_input_types(self):
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
            self.generate_expect_failure(self._model_name, inputs, error_msg)
            self.generate_stream_expect_failure(self._model_name, inputs, error_msg)

    def test_duplicate_inputs(self):
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
            self.generate_expect_failure(self._model_name, inputs, error_msg)
            self.generate_stream_expect_failure(self._model_name, inputs, error_msg)

    def test_generate_stream_response_error(self):
        # Setup text-based input
        text = "hello world"
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": 0, "FAIL_LAST": True}
        r = self.generate_stream(self._model_name, inputs)

        # With "REPETITION": 0, error will be first response and the HTTP code
        # will be set properly
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.check_sse_responses(r, [{"error": "An Error Occurred"}])

        # With "REPETITION" > 0, the first response is valid response and set
        # HTTP code to success, so user must validate each response
        inputs["REPETITION"] = 1
        r = self.generate_stream(self._model_name, inputs)
        r.raise_for_status()

        self.check_sse_responses(r, [{"TEXT": text}, {"error": "An Error Occurred"}])

    def test_race_condition(self):
        # In Triton HTTP frontend, the HTTP response is sent in a different
        # thread than Triton response complete thread, both programs have shared
        # access to the same object, so this test is sending sufficient load to
        # the endpoint, in attempt to expose race condition if any  .
        input1 = {"PROMPT": "hello", "STREAM": False, "param": "segfault"}
        input2 = {
            "PROMPT": "hello",
            "STREAM": True,
            "REPETITION": 3,
            "param": "segfault",
        }
        threads = []

        def thread_func(model_name, inputs):
            self.generate_stream(model_name, inputs).raise_for_status()

        for _ in range(50):
            threads.append(
                threading.Thread(target=thread_func, args=((self._model_name, input1)))
            )
            threads.append(
                threading.Thread(target=thread_func, args=((self._model_name, input2)))
            )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def test_one_response(self):
        # In the current 'inputs' setting, the model will send at least 1
        # response, "STREAM" controls model behavior on sending model responses:
        # If True, the model sends two responses, one is the actual infer
        # response and the other contains flag only to signal end of response.
        # 'generate_stream' endpoint is designed for this case so it should send
        # infer response and complete HTTP response appropriately. And
        # 'generate' endpoint will be able to handle this case as at its core
        # only one infer response is received, which is the same as typical HTTP
        # usage.
        # If False, the model sends one response containing infer response and
        # end flag, which is the same as how non-decoupled model responds.
        inputs = {"PROMPT": "hello world", "STREAM": True}
        r = self.generate_stream(self._model_name, inputs)
        r.raise_for_status()
        r = self.generate(self._model_name, inputs)
        r.raise_for_status()

        inputs["STREAM"] = False
        r = self.generate_stream(self._model_name, inputs)
        r.raise_for_status()
        r = self.generate(self._model_name, inputs)
        r.raise_for_status()

    def test_zero_response(self):
        inputs = {"PROMPT": "hello world", "STREAM": True, "REPETITION": 0}
        r = self.generate_stream(self._model_name, inputs)
        r.raise_for_status()
        # Expect generate fails the inference
        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn(
                "generate expects model to produce exactly 1 response",
                r.json()["error"],
            )

    def test_many_response(self):
        inputs = {"PROMPT": "hello world", "STREAM": True, "REPETITION": 2}
        r = self.generate_stream(self._model_name, inputs)
        r.raise_for_status()
        # Expect generate fails the inference
        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn(
                "generate expects model to produce exactly 1 response",
                r.json()["error"],
            )

    def test_complex_schema(self):
        # Currently only the fundamental conversion is supported, nested object
        # in the request will results in parsing error

        # complex object to parameters (specifying non model input)
        inputs = {
            "PROMPT": "hello world",
            "STREAM": True,
            "PARAMS": {"PARAM_0": 0, "PARAM_1": True},
        }
        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn("parameter 'PARAMS' has invalid type", r.json()["error"])

        # complex object to model input
        inputs = {
            "PROMPT": {"USER": "hello world", "BOT": "world hello"},
            "STREAM": True,
        }
        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn(
                "attempt to access JSON non-string as string", r.json()["error"]
            )

    def test_close_connection_during_streaming(self):
        # verify the responses are streamed as soon as it is generated
        text = "hello world"
        rep_count = 3
        inputs = {"PROMPT": [text], "STREAM": True, "REPETITION": rep_count, "DELAY": 2}
        res = self.generate_stream(self._model_name, inputs, stream=True)
        # close connection while the responses are being generated
        res.close()
        # check server healthiness
        health_url = "http://localhost:8000/v2/health/live"
        requests.get(health_url).raise_for_status()

    def test_parameters(self):
        # Test reserved nested object for parameters
        text = "hello world"
        rep_count = 3
        inputs = {
            "PROMPT": [text],
            "STREAM": True,
            "parameters": {"REPETITION": rep_count},
        }
        self.generate_stream_expect_success(self._model_name, inputs, text, rep_count)

        # parameters keyword is not an object
        inputs = {"PROMPT": [text], "STREAM": True, "parameters": 1}

        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn(
                "Expected JSON object for keyword: 'parameters'", r.json()["error"]
            )

        # parameters contains complex object
        inputs = {
            "PROMPT": [text],
            "STREAM": True,
            "parameters": {"nested": {"twice": 1}},
        }

        r = self.generate(self._model_name, inputs)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.assertIn(
                "Converting keyword: 'parameters': parameter 'nested' has invalid type.",
                r.json()["error"],
            )


if __name__ == "__main__":
    unittest.main()
