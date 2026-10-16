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


"""Streaming through the decoupled ensemble: every response of the decoupled
tensorrt_llm step must reach the client, followed by the final flag."""

import llmapi_test_util as util
import pytest
from ensemble_util import ENSEMBLE_NAME, PREFIX, QUESTION, ensemble_inputs

pytestmark = pytest.mark.usefixtures("streaming_ensemble")


def _stream(streaming):
    with util.GrpcStream() as stream:
        stream.send_inputs(
            ENSEMBLE_NAME, ensemble_inputs(max_tokens=32, streaming=streaming)
        )
        return stream.collect(timeout=120)


def test_streaming_returns_multiple_responses():
    before = util.metric_value("nv_inference_request_success", "prompt_template")
    texts, error, got_final = _stream(streaming=True)
    assert error is None, error
    assert got_final, "no final response; the stream would hang"
    assert len(texts) > 1, texts
    # Streamed responses carry the cumulative generated text, without the
    # prompt, so the template step is checked through its request count.
    for shorter, longer in zip(texts, texts[1:]):
        assert longer.startswith(shorter), texts
    assert len(texts[-1]) > len(texts[0])
    after = util.metric_value("nv_inference_request_success", "prompt_template")
    assert after - before == 1


def test_non_streaming_request_returns_one_response():
    texts, error, got_final = _stream(streaming=False)
    assert error is None, error
    assert got_final
    assert len(texts) == 1, texts
    assert texts[0].startswith(PREFIX + QUESTION), texts


def test_no_single_response_errors_in_server_log():
    assert (
        util.server_log_count("Non-decoupled model cannot send more than one response")
        == 0
    )
