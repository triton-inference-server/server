# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "openai: mark test to run with OpenAI server (subprocess)"
    )
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")


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


### FIXTURES - Refactored from global variables ###


@pytest.fixture(scope="session")
def tool_call_parser():
    return os.environ.get("TEST_TOOL_CALL_PARSER", "llama3")


@pytest.fixture(scope="session")
def backend(tool_call_parser):
    env_backend = os.environ.get("TEST_BACKEND")
    env_model = os.environ.get("TEST_MODEL")

    if not env_backend or not env_model:
        inferred_backend, _ = infer_test_environment(tool_call_parser)
        return inferred_backend
    return env_backend


@pytest.fixture(scope="session")
def model(tool_call_parser):
    env_model = os.environ.get("TEST_MODEL")

    if not env_model:
        _, inferred_model = infer_test_environment(tool_call_parser)
        return inferred_model
    return env_model


@pytest.fixture(scope="session")
def model_repository(backend, tool_call_parser):
    env_repo = os.environ.get("TEST_MODEL_REPOSITORY")

    if env_repo:
        return env_repo
    return infer_test_model_repository(backend, tool_call_parser)


@pytest.fixture(scope="session")
def tokenizer_model():
    return os.environ.get("TEST_TOKENIZER", "meta-llama/Meta-Llama-3.1-8B-Instruct")


@pytest.fixture(scope="session")
def prompt():
    return "What is machine learning?"


@pytest.fixture(scope="session")
def messages(prompt):
    return [{"role": "user", "content": prompt}]


@pytest.fixture(scope="session")
def input(prompt):
    return prompt


# NOTE: OpenAI client requires actual server running, and won't work
# with the FastAPI TestClient. Run the server at module scope to run
# only once for all the tests below.
@pytest.fixture(scope="module")
def server(
    model_repository: str, tokenizer_model: str, backend: str, tool_call_parser: str
):
    args = [
        "--model-repository",
        model_repository,
        "--tokenizer",
        tokenizer_model,
        "--backend",
        backend,
        "--tool-call-parser",
        tool_call_parser,
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
def fastapi_client_class_scope(
    model_repository: str, tokenizer_model: str, backend: str
):
    server = setup_server(model_repository=model_repository)
    app = setup_fastapi_app(tokenizer=tokenizer_model, server=server, backend=backend)
    with TestClient(app) as test_client:
        yield test_client

    server.stop()


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
