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

"""Non-decoupled LLM API model: unary HTTP and gRPC requests."""

import unittest
import uuid

import llmapi_test_util as util
from tritonclient.utils import InferenceServerException

PROMPT = "The capital of France is Paris. The capital of Germany is"


class LlmapiUnaryTest(unittest.TestCase):
    def _expect_error(self, expected, prompt=PROMPT, **params):
        """The request must fail with `expected` in the error message."""
        with self.assertRaises(InferenceServerException) as ctx:
            util.grpc_infer(prompt, **params)
        self.assertIn(expected, str(ctx.exception))

    def _assert_server_healthy(self):
        self.assertTrue(util.server_ready())
        result = util.grpc_infer(PROMPT, sampling_param_max_tokens=2)
        self.assertTrue(util.output_text(result))

    # Basic serving ---------------------------------------------------------

    def test_http_generate(self):
        response = util.http_generate(
            {
                "text_input": PROMPT,
                "sampling_param_max_tokens": 16,
                "sampling_param_temperature": 0.0,
            }
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model_name"], util.MODEL_NAME)
        self.assertTrue(body["text_output"].strip())

    def test_grpc_infer(self):
        result = util.grpc_infer(
            PROMPT, sampling_param_max_tokens=16, sampling_param_temperature=0.0
        )
        self.assertEqual(result.get_output("text_output").datatype, "BYTES")
        self.assertIn("Berlin", util.output_text(result))

    def test_input_is_echoed_unless_excluded(self):
        params = dict(sampling_param_max_tokens=8, sampling_param_temperature=0.0)
        echoed = util.output_text(util.grpc_infer(PROMPT, **params))
        self.assertTrue(echoed.startswith(PROMPT))
        generated = util.output_text(
            util.grpc_infer(
                PROMPT, sampling_param_exclude_input_from_output=True, **params
            )
        )
        self.assertFalse(generated.startswith(PROMPT))
        self.assertEqual(echoed, PROMPT + generated)

    def test_seed_determinism(self):
        params = dict(
            sampling_param_max_tokens=32,
            sampling_param_temperature=1.0,
            sampling_param_exclude_input_from_output=True,
        )

        def generate(seed):
            return util.output_text(
                util.grpc_infer(PROMPT, sampling_param_seed=seed, **params)
            )

        self.assertEqual(generate(1234), generate(1234))
        # Different seeds must not all collapse onto one output.
        self.assertGreater(len({generate(seed) for seed in (1, 2, 3, 4)}), 1)

    # Finish and stop reasons ----------------------------------------------

    def test_finish_reason_length(self):
        result = util.grpc_infer(
            PROMPT,
            sampling_param_max_tokens=4,
            sampling_param_temperature=0.0,
            return_finish_reason=True,
        )
        self.assertEqual(util.output_text(result, "finish_reason"), "length")

    def test_stop_words(self):
        params = dict(
            sampling_param_max_tokens=64,
            sampling_param_temperature=0.0,
            sampling_param_exclude_input_from_output=True,
            sampling_param_stop=["."],
            return_finish_reason=True,
            return_stop_reason=True,
        )
        try:
            result = util.grpc_infer(PROMPT, **params)
        except InferenceServerException as e:
            if "text input must be of type" in str(e):
                # Upstream defect in triton_backend/all_models/llmapi: the stop
                # strings reach SamplingParams as bytes, which the tokenizer
                # rejects, so sampling_param_stop never works. Not a Triton
                # issue; this test enforces the contract once it is fixed.
                self.skipTest(f"known TensorRT-LLM llmapi defect: {e}")
            raise
        text = util.output_text(result)
        self.assertNotIn(".", text)
        self.assertEqual(util.output_text(result, "finish_reason"), "stop")
        self.assertEqual(util.output_text(result, "stop_reason"), ".")

    # Perf metrics ---------------------------------------------------------

    def test_perf_metrics_outputs(self):
        # The outputs are returned as strings; a unique prompt avoids reusing
        # KV-cache blocks from other tests.
        prompt = f"{uuid.uuid4()} " * 200 + "The answer is"
        params = dict(
            sampling_param_max_tokens=4,
            sampling_param_temperature=0.0,
            sampling_param_return_perf_metrics=True,
        )

        def metrics():
            result = util.grpc_infer(prompt, **params)
            values = {}
            for name in util.PERF_METRIC_OUTPUTS:
                output = result.as_numpy(name)
                self.assertIsNotNone(output, f"missing output {name}")
                values[name] = float(output.flatten()[0].decode())
            return values

        first, second = metrics(), metrics()
        for values in (first, second):
            self.assertLessEqual(
                values["arrival_time_ns"], values["first_scheduled_time_ns"]
            )
            self.assertLessEqual(
                values["first_scheduled_time_ns"], values["first_token_time_ns"]
            )
            self.assertLessEqual(
                values["first_token_time_ns"], values["last_token_time_ns"]
            )
        self.assertEqual(first["kv_cache_reused_block"], 0)
        self.assertGreater(second["kv_cache_reused_block"], 0)
        self.assertGreater(second["kv_cache_hit_rate"], first["kv_cache_hit_rate"])

    # Error paths ----------------------------------------------------------

    def test_streaming_requires_decoupled_model(self):
        self._expect_error(
            "Streaming is only supported in decoupled mode",
            streaming=True,
            sampling_param_max_tokens=4,
        )

    def test_missing_text_input(self):
        self._expect_error(
            "text_input is missing", prompt=None, sampling_param_max_tokens=4
        )

    def test_invalid_sampling_values(self):
        self._expect_error("temperature", sampling_param_temperature=-1.0)
        self._expect_error("top_p", sampling_param_top_p=5.0)
        self._expect_error("max_tokens", sampling_param_max_tokens=0)
        self._assert_server_healthy()

    def test_prompt_longer_than_max_num_tokens(self):
        self._expect_error(
            "should not exceed max_num_tokens",
            prompt="hello " * 60000,
            sampling_param_max_tokens=4,
        )
        self._assert_server_healthy()

    def test_lora_inputs_are_validated(self):
        self._expect_error(
            "must all be provided together", lora_id=1, sampling_param_max_tokens=4
        )
        self._expect_error(
            "lora_path does not exist",
            lora_id=1,
            lora_name="adapter",
            lora_path="/nonexistent/adapter",
            sampling_param_max_tokens=4,
        )
        self._expect_error(
            "lora_ckpt_source must be one of",
            lora_id=1,
            lora_name="adapter",
            lora_path="/tmp",
            lora_ckpt_source="bogus",
            sampling_param_max_tokens=4,
        )
        self._assert_server_healthy()

    # Triton core metrics --------------------------------------------------

    def test_core_request_metrics(self):
        before = util.metric_value("nv_inference_request_success")
        count = 3
        for _ in range(count):
            util.grpc_infer(PROMPT, sampling_param_max_tokens=2)
        # The LLM API reports its own errors as error responses, which Triton
        # counts as completed requests; only valid requests are asserted on.
        self.assertEqual(
            util.metric_value("nv_inference_request_success") - before, count
        )
        self.assertEqual(util.metric_value("nv_inference_exec_count") > 0, True)


if __name__ == "__main__":
    unittest.main()
