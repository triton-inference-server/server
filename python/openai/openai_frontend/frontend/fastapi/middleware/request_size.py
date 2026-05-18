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

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.utils import StatusCode, validate_positive_int


async def _disconnect_receive() -> Message:
    return {"type": "http.disconnect"}


class RequestSizeLimitMiddleware:
    """
    Reject HTTP requests whose body exceeds ``http_max_input_size`` bytes.
    First validation rejects on the Content-Length header before any body bytes are
    read. Second validation streams the body chunks, counting bytes as they arrive,
    and rejects as soon as the running total crosses the limit. Driving
    receive() from the middleware protects every endpoint, including
    handlers that never read the body. The buffered body is released the
    moment the application consumes it, so the middleware contributes no
    sustained memory overhead.
    """

    def __init__(self, app: ASGIApp, http_max_input_size: int) -> None:
        self.app = app
        self.http_max_input_size = validate_positive_int(http_max_input_size)
        self._content_too_large_message = (
            "Request content size exceeds the maximum allowed input size of "
            f"{self.http_max_input_size} bytes. "
            "Use --http-max-input-size to increase the limit."
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Stage 1: reject on Content-Length before reading any body bytes.
        for name, value in scope["headers"]:
            if name != b"content-length":
                continue
            try:
                content_length = int(value)
            except ValueError:
                await self._send_error(
                    scope,
                    send,
                    StatusCode.CLIENT_ERROR,
                    "invalid_content_length",
                    "Invalid Content-Length header: not an integer.",
                )
                return
            if content_length < 0:
                await self._send_error(
                    scope,
                    send,
                    StatusCode.CLIENT_ERROR,
                    "invalid_content_length",
                    "Invalid Content-Length header: must be non-negative.",
                )
                return
            if content_length > self.http_max_input_size:
                await self._send_error(
                    scope,
                    send,
                    StatusCode.CONTENT_TOO_LARGE,
                    "content_too_large",
                    self._content_too_large_message,
                )
                return
            break

        # Stage 2: count chunks as they arrive, reject if total exceeds limit.
        body_chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.http_max_input_size:
                await self._send_error(
                    scope,
                    send,
                    StatusCode.CONTENT_TOO_LARGE,
                    "content_too_large",
                    self._content_too_large_message,
                )
                return
            body_chunks.append(chunk)
            if not message.get("more_body", False):
                break

        # Assemble body (single-chunk fast path avoids a copy) and replay to app.
        body_message: Message = {
            "type": "http.request",
            "body": body_chunks[0] if len(body_chunks) == 1 else b"".join(body_chunks),
            "more_body": False,
        }
        del body_chunks

        async def replay_receive() -> Message:
            nonlocal body_message
            if body_message is not None:
                # Drop the reference on hand-off so the body is freed while
                # the app processes it, instead of being held by this closure.
                message, body_message = body_message, None
                return message
            # Body already delivered — delegate to the original receive() so
            # streaming responses can wait for the real client disconnect.
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _send_error(
        self,
        scope: Scope,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            },
        )
        await response(scope, _disconnect_receive, send)
