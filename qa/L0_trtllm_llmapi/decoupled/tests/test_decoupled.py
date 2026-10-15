#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Decoupled LLM API model: streaming, cancellation and stop requests.

The model is served with ``model_transaction_policy { decoupled: true }`` in
config.pbtxt (see common/trtllm_util.sh for why model.yaml alone is not enough).
"""

import time
import unittest

import llmapi_test_util as util

PROMPT = "The capital of France is Paris. The capital of Germany is"
LONG_PROMPT = "Write a very long story about a dragon and a knight."
# Long enough that an uncancelled request outlives the checks below.
LONG_OUTPUT = dict(sampling_param_max_tokens=30000, sampling_param_temperature=0.8)


class LlmapiDecoupledTest(unittest.TestCase):
    # The server logs one line per streamed response (the launcher enables
    # verbose logging), so the count only grows while a request is generating.
    # Decoupled Python-backend requests are never "pending" in Triton's
    # metrics once they start, so the log is what shows a request was released.
    def _responses_logged(self):
        return util.server_log_count("add response output: output: text_output")

    def _assert_generating(self):
        before = self._responses_logged()
        self.assertTrue(
            util.wait_until(lambda: self._responses_logged() > before, timeout=10),
            "the long request is not generating responses",
        )

    def _assert_generation_stopped(self):
        def stopped():
            logged = self._responses_logged()
            time.sleep(2)
            return self._responses_logged() == logged

        self.assertTrue(
            util.wait_until(stopped, timeout=30, interval=0.5),
            "the request is still generating after it was cancelled",
        )

    def _assert_server_healthy(self):
        self.assertTrue(util.server_ready())
        with util.GrpcStream() as stream:
            stream.send(PROMPT, sampling_param_max_tokens=4)
            texts, error, final = stream.collect(timeout=60)
        self.assertIsNone(error)
        self.assertTrue(final)
        self.assertTrue(texts and texts[-1])

    # Streaming ------------------------------------------------------------

    def test_grpc_stream_returns_multiple_responses(self):
        with util.GrpcStream() as stream:
            stream.send(
                PROMPT,
                streaming=True,
                sampling_param_max_tokens=16,
                sampling_param_temperature=0.0,
            )
            texts, error, final = stream.collect()
        self.assertIsNone(error)
        self.assertTrue(final, "the stream did not end with a final response")
        self.assertGreaterEqual(len(texts), 2)

    def test_streamed_text_is_cumulative(self):
        # Every response carries the whole text generated so far, not only the
        # new tokens. Clients that concatenate chunks would duplicate output.
        with util.GrpcStream() as stream:
            stream.send(
                PROMPT,
                streaming=True,
                sampling_param_max_tokens=16,
                sampling_param_temperature=0.0,
                sampling_param_exclude_input_from_output=True,
            )
            texts, error, _ = stream.collect()
        self.assertIsNone(error)
        self.assertGreaterEqual(len(texts), 2)
        for previous, current in zip(texts, texts[1:]):
            self.assertTrue(current.startswith(previous), (previous, current))
            self.assertGreaterEqual(len(current), len(previous))

    def test_non_streaming_request_returns_one_response(self):
        with util.GrpcStream() as stream:
            stream.send(PROMPT, sampling_param_max_tokens=8)
            texts, error, final = stream.collect()
        self.assertIsNone(error)
        self.assertTrue(final)
        self.assertEqual(len(texts), 1)

    def test_http_generate_stream(self):
        status, events = util.http_generate_stream(
            {
                "text_input": PROMPT,
                "streaming": True,
                "sampling_param_max_tokens": 16,
                "sampling_param_temperature": 0.0,
                "sampling_param_exclude_input_from_output": True,
            }
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(events), 2)
        texts = [event["text_output"] for event in events]
        self.assertTrue(texts[-1].startswith(texts[0]))

    # Cancellation ---------------------------------------------------------

    def test_client_cancel_releases_the_request(self):
        stream = util.GrpcStream()
        stream.send(LONG_PROMPT, streaming=True, **LONG_OUTPUT)
        result, error = stream.get()
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self._assert_generating()
        stream.close(cancel_requests=True)
        self._assert_generation_stopped()
        self._assert_server_healthy()

    def test_stop_request_cancels_a_running_request(self):
        request_id = "llmapi-stop-running"
        with util.GrpcStream() as stream, util.GrpcStream() as control:
            stream.send(
                LONG_PROMPT, request_id=request_id, streaming=True, **LONG_OUTPUT
            )
            result, error = stream.get()
            self.assertIsNone(error)
            self._assert_generating()
            control.send("", request_id=request_id, stop=True)
            _, stop_error = control.get()
            self.assertIn("cancelled", str(stop_error).lower())
            _, error, final = stream.collect()
            self.assertTrue(final)
            self.assertIn("cancelled", str(error).lower())
        self._assert_generation_stopped()
        self._assert_server_healthy()

    def test_stop_request_for_unknown_id(self):
        with util.GrpcStream() as control:
            control.send("", request_id="llmapi-no-such-request", stop=True)
            _, error = control.get()
        self.assertIn("cancelled", str(error).lower())

    def test_stop_request_for_completed_id(self):
        request_id = "llmapi-already-done"
        with util.GrpcStream() as stream:
            stream.send(PROMPT, request_id=request_id, sampling_param_max_tokens=4)
            _, error, final = stream.collect()
            self.assertIsNone(error)
            self.assertTrue(final)
            stream.send("", request_id=request_id, stop=True)
            _, error = stream.get()
        self.assertIn("cancelled", str(error).lower())
        self._assert_server_healthy()

    def test_stop_request_without_id_is_rejected(self):
        with util.GrpcStream() as control:
            control.send("", request_id="", stop=True)
            _, error = control.get()
        self.assertIn("request id must be provided", str(error))


if __name__ == "__main__":
    unittest.main()
