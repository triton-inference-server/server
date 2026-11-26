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

import httpx
import pytest
from fastapi import FastAPI
from frontend.fastapi.middleware.api_restriction import (
    APIRestrictionMiddleware,
    RestrictedFeatures,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_value, header_value, status_code",
    [
        ("token", b"token", 200),
        ("token", None, 401),
        ("token", b"", 401),
        ("token", b"wrong", 401),
        ("token", b"caf\xe9", 401),
        ("caf\xe9", b"caf\xe9", 200),
        ("caf\xe9", b"cafe", 401),
        ("caf\udce9", b"caf\xe9", 401),
    ],
)
async def test_restriction_header_value(expected_value, header_value, status_code):
    app = FastAPI()
    restrictions = RestrictedFeatures([["health", "test-key", expected_value]])
    app.add_middleware(APIRestrictionMiddleware, restricted_apis=restrictions)

    @app.get("/health/ready")
    async def ready():
        return {"ready": True}

    headers = [] if header_value is None else [(b"test-key", header_value)]
    # ASGITransport preserves the original header bytes in the request scope.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/health/ready", headers=headers)

    assert response.status_code == status_code
    if status_code == 401:
        assert response.json()["error"]["code"] == "invalid_auth"
