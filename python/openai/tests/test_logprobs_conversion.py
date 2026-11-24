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

import openai_frontend.engine.utils.triton as triton_utils
import pytest

# The vLLM backend sends one dict per generated token. vLLM inserts the
# sampled token first, with its rank in the vocabulary, and then the top-k
# tokens. Here the first and third tokens were the most likely ones, but the
# second was sampled from rank 3, so " elephant" and " the" outrank it. The
# rank 1 token is longer than the sampled one, so building the third offset
# from the wrong token changes it.
VLLM_LOGPROBS = [
    {
        "9906": {"logprob": -0.1, "rank": 1, "decoded_token": "Hello"},
        "13347": {"logprob": -2.5, "rank": 2, "decoded_token": "Hi"},
    },
    {
        "8415": {"logprob": -3.0, "rank": 3, "decoded_token": " cat"},
        "46840": {"logprob": -0.5, "rank": 1, "decoded_token": " elephant"},
        "279": {"logprob": -1.2, "rank": 2, "decoded_token": " the"},
    },
    {
        "7731": {"logprob": -0.3, "rank": 1, "decoded_token": " sat"},
        "10837": {"logprob": -1.8, "rank": 2, "decoded_token": " ran"},
    },
]


@pytest.fixture
def vllm_logprobs(monkeypatch):
    monkeypatch.setattr(
        triton_utils, "_get_logprobs_from_response", lambda response: VLLM_LOGPROBS
    )


def test_chat_logprobs_report_sampled_token(vllm_logprobs):
    content = triton_utils._get_openai_chat_format_logprobs_from_vllm_response(None)

    assert [c.token for c in content] == ["Hello", " cat", " sat"]
    assert [c.logprob for c in content] == [-0.1, -3.0, -0.3]
    assert content[1].bytes == list(" cat".encode("utf-8"))
    # top_logprobs stay ordered from most to least likely
    assert [t.token for t in content[1].top_logprobs] == [" elephant", " the", " cat"]


def test_completion_logprobs_report_sampled_token(vllm_logprobs):
    logprobs = triton_utils._get_openai_completion_format_logprobs_from_vllm_response(
        None
    )

    assert logprobs.tokens == ["Hello", " cat", " sat"]
    assert logprobs.token_logprobs == [-0.1, -3.0, -0.3]
    assert logprobs.text_offset == [0, len("Hello"), len("Hello cat")]
    assert logprobs.top_logprobs[1] == {" elephant": -0.5, " the": -1.2, " cat": -3.0}
