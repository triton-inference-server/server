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

import pytest

from openai_frontend.engine.utils.triton import _create_trtllm_generate_request
from openai_frontend.schemas.openai import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
)
from utils.utils import ClientError


class _UnexpectedCreateRequestModel:
    def create_request(self, inputs):
        raise AssertionError("create_request should not be called")


@pytest.mark.parametrize(
    "request",
    [
        CreateCompletionRequest(model="test-model", prompt="hello", seed=-1),
        CreateChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            seed=-1,
        ),
    ],
)
def test_trtllm_generate_request_rejects_negative_seed(request):
    with pytest.raises(ClientError, match="seed must be non-negative"):
        _create_trtllm_generate_request(
            _UnexpectedCreateRequestModel(),
            "hello",
            request,
            lora_config=None,
            echo_tensor_name=None,
            default_max_tokens=16,
        )
