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

import pytest
from frontend.fastapi.middleware.api_restriction import RestrictedFeatures
from frontend.fastapi.middleware.request_size import RequestSizeLimitMiddleware
from utils.utils import HTTP_DEFAULT_MAX_INPUT_SIZE


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


def test_restricted_feature_summary_omits_header_values():
    restricted = RestrictedFeatures(
        [["inference", "synthetic-header", "synthetic-secret-value"]]
    )

    assert restricted.RestrictionSummary() == {"inference": "synthetic-header"}
    assert "synthetic-secret-value" not in str(restricted.RestrictionSummary())
