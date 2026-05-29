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
import json
import os
import sys
from pathlib import Path

import pytest
import tritonserver
from fastapi.testclient import TestClient

sys.path.append(
    os.path.join(str(Path(__file__).resolve().parent.parent), "openai_frontend")
)

from frontend.fastapi.middleware.request_size import RequestSizeLimitMiddleware
from tests.utils import setup_fastapi_app, setup_server
from utils.utils import HTTP_DEFAULT_MAX_INPUT_SIZE

_MODEL = "mock_llm"

# All POST endpoints.
_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    f"/v1/models/{_MODEL}/load",
    f"/v1/models/{_MODEL}/unload",
)


@pytest.fixture(scope="module")
def client():
    """FastApiFrontend backed by a real Triton server with mock_llm loaded."""
    model_repository = str(Path(__file__).parent / "test_models")
    server = setup_server(
        model_repository,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
        load_models=[_MODEL],
    )
    try:
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
    finally:
        server.stop()


def _assert_content_too_large(response, actual_bytes: int) -> None:
    assert response.status_code == 413
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "content_too_large"
    assert error["message"] == RequestSizeLimitMiddleware._oversized_request_message(
        actual_bytes, HTTP_DEFAULT_MAX_INPUT_SIZE
    )


class TestRequestSizeLimitMiddleware:
    @pytest.mark.parametrize("endpoint", _ENDPOINTS)
    def test_body_at_limit_is_not_rejected(self, client, endpoint):
        response = client.post(endpoint, content=b"x" * HTTP_DEFAULT_MAX_INPUT_SIZE)
        assert response.status_code != 413

    @pytest.mark.parametrize("endpoint", _ENDPOINTS)
    def test_body_over_limit_is_rejected(self, client, endpoint):
        over = HTTP_DEFAULT_MAX_INPUT_SIZE + 1
        response = client.post(endpoint, content=b"x" * over)
        _assert_content_too_large(response, over)

    @pytest.mark.parametrize("endpoint", _ENDPOINTS)
    def test_chunked_body_over_limit_is_rejected(self, client, endpoint):
        over = HTTP_DEFAULT_MAX_INPUT_SIZE + 1

        # httpx switches to chunked transfer when content is an Iterable[bytes].
        def chunks():
            yield b"x" * HTTP_DEFAULT_MAX_INPUT_SIZE
            yield b"x"

        response = client.post(endpoint, content=chunks())
        _assert_content_too_large(response, over)

    def test_get_without_body_is_unaffected(self, client):
        response = client.get(_ENDPOINTS[0])
        assert response.status_code == 405


class TestContentLengthValidation:
    """Stage 1 rejects malformed Content-Length with 400."""

    def _run_with_content_length(self, raw_value: bytes) -> tuple[int, dict]:
        captured: dict = {"status": None, "body": b""}

        async def app(scope, receive, send):
            raise AssertionError("app must not be reached for invalid Content-Length")

        async def receive():
            raise AssertionError("receive() must not be called when Stage 1 rejects")

        async def send(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
            elif message["type"] == "http.response.body":
                captured["body"] += message.get("body", b"")

        middleware = RequestSizeLimitMiddleware(
            app=app, http_max_input_size=HTTP_DEFAULT_MAX_INPUT_SIZE
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", raw_value)],
        }
        asyncio.run(middleware(scope, receive, send))

        assert captured["status"] is not None, "no response was sent"
        return captured["status"], json.loads(captured["body"])

    def _assert_invalid_content_length(self, status: int, body: dict):
        assert status == 400
        assert set(body) == {"error"}
        error = body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_content_length"

    @pytest.mark.parametrize("raw", [b"not-a-number", b""])
    def test_non_integer_content_length_rejected_with_400(self, raw):
        status, body = self._run_with_content_length(raw)
        self._assert_invalid_content_length(status, body)
        assert "not an integer" in body["error"]["message"]

    @pytest.mark.parametrize("raw", [b"-1", b"-1024"])
    def test_negative_content_length_rejected_with_400(self, raw):
        status, body = self._run_with_content_length(raw)
        self._assert_invalid_content_length(status, body)
        assert "non-negative" in body["error"]["message"]
