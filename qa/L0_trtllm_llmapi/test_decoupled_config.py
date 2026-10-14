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

"""Characterize the decoupled configuration trap of the LLM API model.

The model repository is prepared with ``triton_config.decoupled: True`` in
model.yaml but WITHOUT ``model_transaction_policy`` in config.pbtxt.
launch_triton_server.py passes --disable-auto-complete-config, so the
auto_complete_config() that would copy the model.yaml flag into the served
configuration never runs: model.py believes it is decoupled while Triton
serves it as a non-decoupled model. A streaming request then gets no response
and the server logs one error per generated token.

These tests pin that behaviour so a change of it is noticed. If upstream fixes
the launcher or the model, this suite is expected to fail: update it and drop
the config.pbtxt workaround from common/trtllm_util.sh.
"""

import sys
import unittest

import requests

sys.path.append("../common")

import llmapi_test_util as util  # noqa: E402
from test_util import TestResultCollector  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Germany is"
SINGLE_RESPONSE_ERROR = "Non-decoupled model cannot send more than one response"


class LlmapiDecoupledConfigTest(TestResultCollector):
    def test_model_yaml_decoupled_flag_is_not_applied(self):
        config = requests.get(
            f"{util.HTTP_URL}/v2/models/{util.MODEL_NAME}/config", timeout=10
        ).json()
        policy = config.get("model_transaction_policy") or {}
        self.assertFalse(policy.get("decoupled", False), config)

    def test_streaming_request_gets_no_response(self):
        stream = util.GrpcStream()
        try:
            stream.send(
                PROMPT,
                streaming=True,
                sampling_param_max_tokens=8,
                sampling_param_temperature=0.0,
            )
            result, error = stream.get(timeout=20)
            self.assertIsNone(result)
            self.assertIsNone(error)
            self.assertTrue(
                util.wait_until(
                    lambda: SINGLE_RESPONSE_ERROR in util.server_log_text(), timeout=20
                ),
                "the server did not log the single-response error",
            )
        finally:
            stream.close(cancel_requests=True)

    def test_non_streaming_request_is_still_served(self):
        result = util.grpc_infer(PROMPT, sampling_param_max_tokens=8)
        self.assertTrue(util.output_text(result))


if __name__ == "__main__":
    unittest.main()
