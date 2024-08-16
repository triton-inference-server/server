from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.tests.utils import OpenAIServer, setup_fastapi_app

### TEST ENVIRONMENT SETUP ###
TEST_BACKEND = ""
TEST_MODEL = ""
TEST_PROMPT = "What is machine learning?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]
TEST_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
try:
    import vllm as _

    TEST_BACKEND = "vllm"
    TEST_MODEL = "llama-3.1-8b-instruct"
except ImportError:
    pass

try:
    import tensorrt_llm as _

    TEST_BACKEND = "tensorrtllm"
    TEST_MODEL = "tensorrt_llm_bls"
except ImportError:
    pass

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
