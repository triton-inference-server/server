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


"""OpenAI frontend against the TensorRT-LLM LLM API (Python) model interface.

Runs against tests/test_models/mock_llmapi, which has the real model's inputs,
decoupled policy and cumulative streaming, so no GPU or engine is needed.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import tritonserver
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

sys.path.append(
    os.path.join(str(Path(__file__).resolve().parent.parent), "openai_frontend")
)

from tests.utils import setup_fastapi_app, setup_server  # noqa: E402

_MODEL = "mock_llmapi"
_PROMPT = "What is machine learning?"


@pytest.fixture(scope="module")
def tokenizer(tokenizer_model):
    return AutoTokenizer.from_pretrained(tokenizer_model)


# The model is recognised by its inputs; --backend llmapi must behave the same.
@pytest.fixture(
    scope="module", params=[None, "llmapi"], ids=["autodetect", "backend-flag"]
)
def client(request, tokenizer_model):
    server = setup_server(
        str(Path(__file__).parent / "llmapi_mock_models"),
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
        load_models=[_MODEL],
    )
    try:
        app = setup_fastapi_app(
            tokenizer=tokenizer_model, server=server, backend=request.param
        )
        with TestClient(app) as test_client:
            yield test_client
    finally:
        server.stop()


def _complete(client, **body):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": _PROMPT, **body}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _stream(client, endpoint, **body):
    """Return the decoded data events of a streamed response, without [DONE]."""
    response = client.post(endpoint, json={"model": _MODEL, "stream": True, **body})
    assert response.status_code == 200, response.text
    events = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    return [json.loads(event) for event in events[:-1]]


class TestCompletions:
    def test_prompt_is_not_echoed_by_default(self, client):
        # The model itself echoes unless told to exclude the input.
        text = _complete(client, max_tokens=3)["choices"][0]["text"]
        assert text == " w0 w1 w2"

    def test_echo_includes_the_prompt(self, client):
        text = _complete(client, max_tokens=2, echo=True)["choices"][0]["text"]
        assert text == _PROMPT + " w0 w1"

    def test_usage_is_estimated_from_the_tokenizer(self, client, tokenizer):
        body = _complete(client, max_tokens=3)
        text = body["choices"][0]["text"]
        usage = body["usage"]
        assert usage["prompt_tokens"] == len(tokenizer.encode(_PROMPT))
        assert usage["completion_tokens"] == len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_streaming_returns_new_text_only(self, client):
        # Every response of the model carries the whole text so far.
        events = _stream(client, "/v1/completions", prompt=_PROMPT, max_tokens=3)
        pieces = [event["choices"][0]["text"] for event in events]
        assert pieces == [" w0", " w1", " w2"]
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_usage_matches_non_streaming(self, client):
        expected = _complete(client, max_tokens=3)["usage"]
        events = _stream(
            client,
            "/v1/completions",
            prompt=_PROMPT,
            max_tokens=3,
            stream_options={"include_usage": True},
        )
        assert events[-1]["choices"] == []
        assert events[-1]["usage"] == expected

    def test_sampling_parameters_are_converted(self, client):
        text = _complete(
            client,
            prompt="echo-params:",
            max_tokens=5,
            temperature=0.5,
            top_p=0.75,
            frequency_penalty=0.25,
            presence_penalty=0.5,
            seed=7,
            stop=["END"],
        )["choices"][0]["text"]
        assert text.startswith("PARAMS:")
        received = json.loads(text[len("PARAMS:") :])
        assert received == {
            "sampling_param_exclude_input_from_output": True,
            "sampling_param_frequency_penalty": 0.25,
            "sampling_param_max_tokens": 5,
            "sampling_param_presence_penalty": 0.5,
            "sampling_param_seed": 7,
            "sampling_param_stop": ["END"],
            "sampling_param_temperature": 0.5,
            "sampling_param_top_p": 0.75,
            "streaming": False,
        }

    def test_logprobs_are_rejected(self, client):
        response = client.post(
            "/v1/completions",
            json={"model": _MODEL, "prompt": _PROMPT, "logprobs": 1},
        )
        assert response.status_code == 400
        assert "vLLM" in response.text


class TestChatCompletions:
    def _messages(self):
        return [{"role": "user", "content": _PROMPT}]

    def test_chat_completion(self, client, tokenizer):
        response = client.post(
            "/v1/chat/completions",
            json={"model": _MODEL, "messages": self._messages(), "max_tokens": 2},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["choices"][0]["message"]["content"] == " w0 w1"
        templated = tokenizer.apply_chat_template(
            self._messages(), tokenize=False, add_generation_prompt=True
        )
        assert body["usage"]["prompt_tokens"] == len(tokenizer.encode(templated))
        assert body["usage"]["completion_tokens"] == len(
            tokenizer.encode(" w0 w1", add_special_tokens=False)
        )

    def test_streaming_chat_returns_new_text_only(self, client):
        events = _stream(
            client,
            "/v1/chat/completions",
            messages=self._messages(),
            max_tokens=3,
            stream_options={"include_usage": True},
        )
        content = "".join(
            event["choices"][0]["delta"].get("content") or ""
            for event in events
            if event["choices"]
        )
        assert content == " w0 w1 w2"
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-1]["choices"] == []
        assert events[-1]["usage"]["completion_tokens"] > 0

    def test_schema_constrained_tool_call_is_rejected(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": self._messages(),
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            },
        )
        assert response.status_code == 400
        assert "LLM API" in response.text


class TestUnsupported:
    def test_embeddings_are_rejected(self, client):
        response = client.post(
            "/v1/embeddings", json={"model": _MODEL, "input": "hello"}
        )
        assert response.status_code == 400
