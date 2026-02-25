# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Re-runs the OpenAI client test suites against a model loaded via explicit
model control mode.

This MUST be in a separate module from test_openai_client.py so that the
module-scoped server fixtures do not overlap on the GPU.  The NONE-mode
server from test_openai_client.py is torn down (subprocess killed, GPU freed)
before this module's server_explicit fixture starts."""

import pytest
from tests.test_openai_client import TestAsyncOpenAIClient, TestOpenAIClient


@pytest.mark.openai
class TestOpenAIClientExplicitMode(TestOpenAIClient):
    @pytest.fixture(scope="class")
    def client(self, server_explicit):
        return server_explicit.get_client()


@pytest.mark.openai
class TestAsyncOpenAIClientExplicitMode(TestAsyncOpenAIClient):
    @pytest.fixture(scope="class")
    def client(self, server_explicit):
        return server_explicit.get_async_client()
