# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.utils import OpenAIServer, setup_fastapi_app

### TEST ENVIRONMENT SETUP ###
TEST_BACKEND = ""
TEST_MODEL = ""
TEST_PROMPT = "What is machine learning?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]
TEST_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Infer the test environment for simplicity in local dev/testing.
try:
    import vllm as _

    TEST_BACKEND = "vllm"
    TEST_MODEL = "llama-3.1-8b-instruct"
except ImportError:
    print("No vllm installation found.")

try:
    import tensorrt_llm as _

    TEST_BACKEND = "tensorrtllm"
    TEST_MODEL = "tensorrt_llm_bls"
except ImportError:
    print("No tensorrt_llm installation found.")

if not TEST_BACKEND or not TEST_MODEL:
    raise Exception("Unknown test environment")
###


# NOTE: OpenAI client requires actual server running, and won't work
# with the FastAPI TestClient. Run the server at module scope to run
# only once for all the tests below.
@pytest.fixture(scope="module")
def server():
    model_repository = Path(__file__).parent / f"{TEST_BACKEND}_models"
    args = ["--model-repository", model_repository, "--tokenizer", TEST_TOKENIZER]

    with OpenAIServer(args) as openai_server:
        yield openai_server


# NOTE: The FastAPI TestClient acts like a server and triggers the FastAPI app
# lifespan startup/shutdown, but does not actually expose the network port to interact
# with arbitrary clients - you must use the TestClient returned to interact with
# the "server" when "starting the server" via TestClient.
@pytest.fixture(scope="class")
def fastapi_client_class_scope():
    model_repository = str(Path(__file__).parent / f"{TEST_BACKEND}_models")
    app = setup_fastapi_app(tokenizer=TEST_TOKENIZER, model_repository=model_repository)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def model():
    return TEST_MODEL


@pytest.fixture
def backend():
    return TEST_BACKEND


@pytest.fixture
def prompt():
    return TEST_PROMPT


@pytest.fixture
def messages():
    return TEST_MESSAGES
