import copy
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import init_app

### TEST ENVIRONMENT SETUP ###
TEST_BACKEND = ""
TEST_MODEL = ""
TEST_PROMPT = "Machine learning is"
try:
    import vllm as _

    TEST_BACKEND = "vllm"
    TEST_MODEL = "llama-3-8b-instruct"
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


class TestCompletions:
    # TODO: Consider module/package scope, or join ChatCompletions tests into same file
    # to run server only once for both sets of tests for faster iteration.
    @pytest.fixture(scope="class")
    def client(self):
        model_repository = Path(__file__).parent / f"{TEST_BACKEND}_models"
        os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)
        app = init_app()
        with TestClient(app) as test_client:
            yield test_client

    def test_completions_defaults(self, client):
        response = client.post(
            "/v1/completions",
            json={"model": TEST_MODEL, "prompt": TEST_PROMPT},
        )

        print("Response:", response.json())
        assert response.status_code == 200
        # NOTE: Could be improved to look for certain quality of response,
        #       or tested with dummy identity model.
        assert response.json()["choices"][0]["text"].strip()
        # "usage" currently not supported
        assert response.json()["usage"] == None

    @pytest.mark.parametrize(
        "sampling_parameter, value",
        [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
            # logprobs is an integer for completions
            ("logprobs", 5),
            ("logit_bias", {"0": 0}),
        ],
    )
    def test_completions_sampling_parameters(self, client, sampling_parameter, value):
        response = client.post(
            "/v1/completions",
            json={
                "model": TEST_MODEL,
                "prompt": TEST_PROMPT,
                sampling_parameter: value,
            },
        )
        print("Response:", response.json())

        # TODO: Add support and remove this check
        unsupported_parameters = ["logprobs", "logit_bias"]
        if sampling_parameter in unsupported_parameters:
            assert response.status_code == 400
            assert response.json()["detail"] == "logit bias and log probs not supported"
            return

        assert response.status_code == 200
        assert response.json()["choices"][0]["text"].strip()

    # Simple tests to verify max_tokens roughly behaves as expected
    def test_completions_max_tokens(self, client):
        responses = []
        payload = {"model": TEST_MODEL, "prompt": TEST_PROMPT, "max_tokens": 1}

        # Send two requests with max_tokens = 1 to check their similarity
        payload["max_tokens"] = 1
        responses.append(
            client.post(
                "/v1/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/completions",
                json=payload,
            )
        )
        # Send one requests with larger max_tokens to check its dis-similarity
        payload["max_tokens"] = 100
        responses.append(
            client.post(
                "/v1/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = responses[0].json()["choices"][0]["text"].strip().split()
        response2_text = responses[1].json()["choices"][0]["text"].strip().split()
        response3_text = responses[2].json()["choices"][0]["text"].strip().split()
        # Simplification: One token shouldn't be more than one space-delimited word
        assert len(response1_text) == len(response2_text) == 1
        assert len(response3_text) > len(response1_text)

    @pytest.mark.skipif(TEST_BACKEND != "vllm", reason="Only used to test vLLM backend")
    @pytest.mark.parametrize(
        "temperature",
        [0.0, 1.0],
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_completions_temperature_vllm(self, client, temperature):
        responses = []
        payload = {
            "model": TEST_MODEL,
            "prompt": TEST_PROMPT,
            "temperature": temperature,
        }

        responses.append(
            client.post(
                "/v1/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = responses[0].json()["choices"][0]["text"].strip().split()
        response2_text = responses[1].json()["choices"][0]["text"].strip().split()

        # Temperature of 0.0 indicates greedy sampling, so check
        # that two equivalent requests produce the same response.
        if temperature == 0.0:
            # NOTE: This check may be ambitious to get an exact match in all
            # frameworks depending on how other parameter defaults are set, so
            # it can probably be removed if it introduces flakiness.
            print(f"Comparing '{response1_text}' == '{response2_text}'")
            assert response1_text == response2_text
        # Temperature of 1.0 indicates maximum randomness, so check
        # that two equivalent requests produce different responses.
        elif temperature == 1.0:
            print(f"Comparing '{response1_text}' != '{response2_text}'")
            assert response1_text != response2_text
        # Don't bother checking values other than the extremes
        else:
            raise ValueError(f"Unexpected {temperature=} for this test.")

    # Remove xfail when fix is released and this test returns xpass status
    @pytest.mark.xfail(
        reason="TRT-LLM BLS model will ignore temperature until a later release"
    )
    @pytest.mark.skipif(
        TEST_BACKEND != "tensorrtllm", reason="Only used to test TRT-LLM backend"
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_completions_temperature_tensorrtllm(self, client):
        responses = []
        payload1 = {
            "model": TEST_MODEL,
            "prompt": TEST_PROMPT,
            "temperature": 0.0,
            # TRT-LLM requires certain settings of `top_k` / `top_p` to
            # respect changes in `temperature`
            "top_p": 0.5,
        }
        payload2 = copy.deepcopy(payload1)
        payload2["temperature"] = 1.0

        # First 2 responses should be the same in TRT-LLM with identical payload
        responses.append(
            client.post(
                "/v1/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/completions",
                json=payload1,
            )
        )
        # Third response should differ with different temperature in payload
        responses.append(
            client.post(
                "/v1/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = responses[0].json()["choices"][0]["text"].strip().split()
        response2_text = responses[1].json()["choices"][0]["text"].strip().split()
        response3_text = responses[2].json()["choices"][0]["text"].strip().split()

        assert response1_text == response2_text
        assert response1_text != response3_text

    # Simple tests to verify seed roughly behaves as expected
    def test_completions_seed(self, client):
        responses = []
        payload1 = {"model": TEST_MODEL, "prompt": TEST_PROMPT, "seed": 1}
        payload2 = copy.deepcopy(payload1)
        payload2["seed"] = 2

        # First 2 responses should be the same in TRT-LLM with identical payload
        responses.append(
            client.post(
                "/v1/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/completions",
                json=payload1,
            )
        )
        # Third response should differ with different temperature in payload
        responses.append(
            client.post(
                "/v1/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = responses[0].json()["choices"][0]["text"].strip().split()
        response2_text = responses[1].json()["choices"][0]["text"].strip().split()
        response3_text = responses[2].json()["choices"][0]["text"].strip().split()

        assert response1_text == response2_text
        assert response1_text != response3_text

    @pytest.mark.parametrize(
        "sampling_parameter, value",
        [
            ("temperature", 2.1),
            ("temperature", -0.1),
            ("max_tokens", -1),
            ("top_p", 1.1),
            ("frequency_penalty", 3),
            ("frequency_penalty", -3),
            ("presence_penalty", 2.1),
            ("presence_penalty", -2.1),
        ],
    )
    def test_completions_invalid_sampling_parameters(
        self, client, sampling_parameter, value
    ):
        response = client.post(
            "/v1/completions",
            json={
                "model": TEST_MODEL,
                "prompt": TEST_PROMPT,
                sampling_parameter: value,
            },
        )

        print("Response:", response.json())
        assert response.status_code == 422

    def test_completions_empty_request(self, client):
        response = client.post("/v1/completions", json={})
        assert response.status_code == 422

    def test_completions_no_model(self, client):
        response = client.post("/v1/completions", json={"prompt": TEST_PROMPT})
        assert response.status_code == 422

    def test_completions_no_prompt(self, client):
        response = client.post("/v1/completions", json={"model": TEST_MODEL})
        assert response.status_code == 422

    def test_completions_empty_prompt(self, client):
        response = client.post(
            "/v1/completions", json={"model": TEST_MODEL, "prompt": ""}
        )

        # NOTE: Should this be validated in schema instead?
        # 400 Error returned in route handler
        assert response.status_code == 400

    def test_no_prompt(self, client):
        response = client.post("/v1/completions", json={"model": TEST_MODEL})

        # 422 Error returned by schema validation
        assert response.status_code == 422

    def test_completions_multiple_choices(self, client):
        response = client.post(
            "/v1/completions", json={"model": TEST_MODEL, "prompt": TEST_PROMPT, "n": 2}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Only single choice is supported"

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_multi_lora(self):
        pass

    # TODO: Do we want to support "usage" field for token counts in response?
    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_usage_response(self):
        pass
