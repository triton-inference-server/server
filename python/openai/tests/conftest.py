# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.utils import OpenAIServer, setup_fastapi_app, setup_server


### TEST ENVIRONMENT SETUP ###
def infer_test_environment(tool_call_parser):
    # Infer the test environment for simplicity in local dev/testing.
    try:
        import vllm as _

        backend = "vllm"
        if tool_call_parser == "mistral":
            model = "mistral-nemo-instruct-2407"
        else:
            model = "llama-3.1-8b-instruct"
        return backend, model
    except ImportError:
        print("No vllm installation found.")

    try:
        import tensorrt_llm as _

        backend = "tensorrtllm"
        model = "tensorrt_llm_bls"
        return backend, model
    except ImportError:
        print("No tensorrt_llm installation found.")

    raise Exception("Unknown test environment")


def infer_test_model_repository(backend, tool_call_parser):
    if tool_call_parser == "mistral":
        model_repository = str(Path(__file__).parent / f"{backend}_mistral_models")
    else:
        model_repository = str(Path(__file__).parent / f"{backend}_models")
    return model_repository


# TODO: Refactor away from global variables
TEST_MODEL = os.environ.get("TEST_MODEL")
TEST_BACKEND = os.environ.get("TEST_BACKEND")
TEST_MODEL_REPOSITORY = os.environ.get("TEST_MODEL_REPOSITORY")

TEST_TOKENIZER = os.environ.get(
    "TEST_TOKENIZER", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
TEST_TOOL_CALL_PARSER = os.environ.get("TEST_TOOL_CALL_PARSER", "llama3")
TEST_PROMPT = "What is machine learning?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]

if not TEST_BACKEND or not TEST_MODEL:
    TEST_BACKEND, TEST_MODEL = infer_test_environment(TEST_TOOL_CALL_PARSER)

if not TEST_MODEL_REPOSITORY:
    TEST_MODEL_REPOSITORY = infer_test_model_repository(
        TEST_BACKEND, TEST_TOOL_CALL_PARSER
    )


# NOTE: OpenAI client requires actual server running, and won't work
# with the FastAPI TestClient. Run the server at module scope to run
# only once for all the tests below.
@pytest.fixture(scope="module")
def server():
    args = [
        "--model-repository",
        TEST_MODEL_REPOSITORY,
        "--tokenizer",
        TEST_TOKENIZER,
        "--backend",
        TEST_BACKEND,
        "--tool-call-parser",
        TEST_TOOL_CALL_PARSER,
    ]
    # TODO: Incorporate kserve frontend binding smoke tests to catch any
    # breakage with default values or slight cli arg variations
    extra_args = ["--enable-kserve-frontends"]
    args += extra_args

    with OpenAIServer(args) as openai_server:
        yield openai_server


# NOTE: The FastAPI TestClient acts like a server and triggers the FastAPI app
# lifespan startup/shutdown, but does not actually expose the network port to interact
# with arbitrary clients - you must use the TestClient returned to interact with
# the "server" when "starting the server" via TestClient.
@pytest.fixture(scope="class")
def fastapi_client_class_scope():
    server = setup_server(model_repository=TEST_MODEL_REPOSITORY)
    app = setup_fastapi_app(
        tokenizer=TEST_TOKENIZER, server=server, backend=TEST_BACKEND
    )
    with TestClient(app) as test_client:
        yield test_client

    server.stop()


@pytest.fixture(scope="module")
def model_repository():
    return TEST_MODEL_REPOSITORY


@pytest.fixture(scope="module")
def model():
    return TEST_MODEL


@pytest.fixture(scope="module")
def backend():
    return TEST_BACKEND


@pytest.fixture(scope="module")
def tokenizer_model():
    return TEST_TOKENIZER


@pytest.fixture(scope="module")
def prompt():
    return TEST_PROMPT


@pytest.fixture(scope="module")
def messages():
    return TEST_MESSAGES


# FIXME: In TRTLLM tests, the in-process Triton server for the FastAPI app
# does not automatically release GPU memory, even after calling stop().
# The memory is only released when the entire pytest process exits.
#
# As a result, when the OpenAI server starts another Triton server as a subprocess,
# there may not be enough GPU memory available to launch a new model instance.
#
# This is a workaround to ensure that tests using the OpenAI server run first.
# Once the OpenAI server subprocess is terminated, tests using the FastAPI app can safely run.
def pytest_collection_modifyitems(session, config, items):
    def get_priority(item):
        cls = item.cls
        if cls:
            if getattr(cls, "pytestmark", None):
                for mark in cls.pytestmark:
                    if mark.name == "openai":
                        return 0
                    elif mark.name == "fastapi":
                        return 1
        return 2  # unmarked tests last

    items.sort(key=get_priority)
