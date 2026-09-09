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

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tritonserver
from engine.triton_engine import TritonLLMEngine, TritonModelMetadata
from engine.utils.triton import (
    _create_vllm_embedding_request,
    _create_vllm_generate_request,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from frontend.fastapi.routers import embeddings
from schemas.openai import CreateEmbeddingRequest
from utils.utils import ServerError


def _response(vector=None, tokens=1, final=True):
    outputs = {}
    if vector is not None:
        outputs["text_output"] = tritonserver.Tensor.from_string_array(
            [json.dumps(vector)]
        )
    if tokens is not None:
        outputs["num_input_tokens"] = tritonserver.Tensor.from_dlpack(
            np.array([tokens], dtype=np.int32)
        )
        outputs["num_output_tokens"] = tritonserver.Tensor.from_dlpack(
            np.array([0], dtype=np.int32)
        )
    return SimpleNamespace(outputs=outputs, final=final)


class _Responses:
    """Native response-handle stand-in with observable cancellation."""

    def __init__(self, items, gate=None):
        self.items = iter(items)
        self.gate = gate
        self.started = asyncio.Event()
        self.cancel = Mock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        item = next(self.items, None)
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def engine_and_model(monkeypatch):
    model = SimpleNamespace(
        create_request=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(request_id=None, **kwargs)
        ),
        async_infer=Mock(
            side_effect=lambda request: _Responses([_response([1.0, 2.0])])
        ),
    )
    metadata = TritonModelMetadata(
        name="test-model",
        backend="vllm",
        model=model,
        tokenizer=None,
        lora_configs=None,
        echo_tensor_name=None,
        create_time=0,
        inference_request_converter=_create_vllm_generate_request,
        embedding_request_converter=_create_vllm_embedding_request,
    )
    monkeypatch.setattr(
        TritonLLMEngine, "_get_model_metadata", lambda self: {metadata.name: metadata}
    )
    engine = TritonLLMEngine(server=None, tokenizer="", default_max_tokens=16)
    return engine, model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, expected_inputs",
    [
        ("hello", ["hello"]),
        ([101, 102], [[101, 102]]),
        ([101, "102"], [[101, 102]]),
        ([True], [[1]]),
        ([101.0], [[101]]),
        (["hello"], ["hello"]),
        (["101", "102"], ["101", "102"]),
    ],
)
async def test_embedding_input_forms(engine_and_model, value, expected_inputs):
    engine, model = engine_and_model
    response = await engine.embedding(
        CreateEmbeddingRequest(model="test-model", input=value)
    )

    submitted = [
        json.loads(call.args[0].inputs["embedding_request"][0])["input"]
        for call in model.async_infer.call_args_list
    ]
    assert submitted == expected_inputs
    assert [item.index for item in response.data] == list(range(len(expected_inputs)))


@pytest.mark.parametrize(
    "value", [["valid", ""], ["valid", 101], ["valid", None], [[101], [102]]]
)
def test_invalid_batch_submits_nothing(engine_and_model, value):
    engine, model = engine_and_model
    app = FastAPI()
    app.include_router(embeddings.router)
    app.engine = engine
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings", json={"model": "test-model", "input": value}
        )

    assert response.status_code == 422
    model.create_request.assert_not_called()
    model.async_infer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["float", "base64"])
async def test_batch_outputs_options_and_usage(engine_and_model, encoding):
    engine, model = engine_and_model
    vectors = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    tokens = {"a": 2, "b": 3}

    def infer(request):
        payload = json.loads(request.inputs["embedding_request"][0])
        assert payload["pooling_params"] == {"dimensions": [2]}
        assert request.inputs["return_num_input_tokens"].tolist() == [True]
        assert request.inputs["return_num_output_tokens"].tolist() == [True]
        return _Responses(
            [_response(vectors[payload["input"]], tokens[payload["input"]])]
        )

    model.async_infer.side_effect = infer
    request = CreateEmbeddingRequest(
        model="test-model",
        input=["a", "b", "a"],
        dimensions=2,
        encoding_format=encoding,
        user="test-user",
    )
    original = request.model_dump()
    response = await engine.embedding(request)

    assert request.model_dump() == original
    assert response.model == "test-model" and response.object == "list"
    assert len(response.data) == 3
    for index, item in enumerate(response.data):
        vector = item.embedding
        if encoding == "base64":
            vector = np.frombuffer(base64.b64decode(vector), dtype=np.float32)
        np.testing.assert_array_equal(vector, vectors[request.input[index]])
        assert item.index == index and item.object == "embedding"
    assert response.usage.prompt_tokens == 7
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("token_counts", [(None, 3), (2, None), (None, None), (0, 0)])
async def test_batch_unknown_and_zero_usage(engine_and_model, token_counts):
    engine, model = engine_and_model
    model.async_infer.side_effect = [
        _Responses([_response([1.0], tokens)]) for tokens in token_counts
    ]
    response = await engine.embedding(
        CreateEmbeddingRequest(model="test-model", input=["a", "b"])
    )
    if None in token_counts:
        assert response.usage is None
    else:
        assert response.usage.prompt_tokens == response.usage.total_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "envelope", ["single", "separate-final", "empty", "non-final", "extra"]
)
async def test_each_child_response_envelope(engine_and_model, envelope):
    engine, model = engine_and_model
    valid = _response([1.0])
    cases = {
        "single": [valid],
        "separate-final": [_response([2.0], final=False), _response(tokens=None)],
        "empty": [],
        "non-final": [_response([2.0], final=False)],
        "extra": [valid, valid, valid],
    }
    handles = [
        _Responses([valid]),
        _Responses(cases[envelope]),
    ]
    model.async_infer.side_effect = handles
    request = CreateEmbeddingRequest(model="test-model", input=["a", "b"])
    if envelope in ("single", "separate-final"):
        response = await engine.embedding(request)
        assert len(response.data) == 2
        assert response.data[1].embedding == ([1.0] if envelope == "single" else [2.0])
        for handle in handles:
            handle.cancel.assert_not_called()
    else:
        with pytest.raises(ServerError):
            await engine.embedding(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["conversion", "submission", "iteration"])
async def test_batch_failure_stops_submission(engine_and_model, failure_point):
    engine, model = engine_and_model
    failure = RuntimeError("injected backend failure")
    completed = _Responses([_response([1.0])])
    responses = _Responses([failure])
    if failure_point == "conversion":
        model.create_request.side_effect = [SimpleNamespace(id="first"), failure]
        model.async_infer.side_effect = [completed]
    elif failure_point == "submission":
        model.async_infer.side_effect = [completed, failure]
    else:
        model.async_infer.side_effect = [completed, responses]

    with pytest.raises(RuntimeError, match="injected backend failure"):
        await engine.embedding(
            CreateEmbeddingRequest(model="test-model", input=["a", "b", "c"])
        )
    assert [
        json.loads(call.kwargs["inputs"]["embedding_request"][0])["input"]
        for call in model.create_request.call_args_list
    ] == ["a", "b"]
    assert model.async_infer.call_count == (1 if failure_point == "conversion" else 2)
    completed.cancel.assert_not_called()
    if failure_point == "iteration":
        responses.cancel.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["a", ["a", "b", "c"]])
async def test_cancellation_reaches_native_request(engine_and_model, value):
    engine, model = engine_and_model
    completed = _Responses([_response([1.0])])
    responses = _Responses([_response([1.0])], gate=asyncio.Event())
    model.async_infer.side_effect = (
        [responses] if isinstance(value, str) else [completed, responses]
    )
    task = asyncio.create_task(
        engine.embedding(CreateEmbeddingRequest(model="test-model", input=value))
    )
    try:
        await asyncio.wait_for(responses.started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    responses.cancel.assert_called_once()
    completed.cancel.assert_not_called()
    assert model.async_infer.call_count == (1 if isinstance(value, str) else 2)
