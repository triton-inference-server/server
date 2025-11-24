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

import copy

import pytest


class TestCompletions:
    @pytest.fixture(scope="class")
    def client(self, fastapi_client_class_scope):
        yield fastapi_client_class_scope

    def test_completions_defaults(self, client, model: str, prompt: str):
        response = client.post(
            "/v1/completions",
            json={"model": model, "prompt": prompt},
        )

        print("Response:", response.json())
        assert response.status_code == 200
        # NOTE: Could be improved to look for certain quality of response,
        #       or tested with dummy identity model.
        assert response.json()["choices"][0]["text"].strip()

        usage = response.json().get("usage")
        assert usage is not None

    @pytest.mark.parametrize(
        "sampling_parameter, value",
        [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
            ("best_of", 1),
            ("n", 1),
            # logprobs is an integer for completions
            ("logprobs", 5),
            ("logit_bias", {"0": 0}),
            # NOTE: Extensions to the spec
            ("min_tokens", 16),
            ("ignore_eos", True),
        ],
    )
    def test_completions_sampling_parameters(
        self, client, sampling_parameter, value, model: str, prompt: str
    ):
        response = client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                sampling_parameter: value,
            },
        )
        print("Response:", response.json())

        # FIXME: Add support and remove this check
        unsupported_parameters = ["logit_bias"]
        if sampling_parameter in unsupported_parameters:
            assert response.status_code == 400
            assert response.json()["detail"] == "logit bias is not supported"
            return

        assert response.status_code == 200
        assert response.json()["choices"][0]["text"].strip()

    # Simple tests to verify max_tokens roughly behaves as expected
    def test_completions_max_tokens(self, client, model: str, prompt: str):
        responses = []
        payload = {"model": model, "prompt": prompt, "max_tokens": 1}

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

    @pytest.mark.parametrize(
        "temperature",
        [0.0, 1.0],
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_completions_temperature_vllm(
        self, client, temperature, backend: str, model: str, prompt: str
    ):
        if backend != "vllm":
            pytest.skip(reason="Only used to test vLLM-specific temperature behavior")

        responses = []
        payload = {
            "model": model,
            "prompt": prompt,
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
    # Simple tests to verify temperature roughly behaves as expected
    def test_completions_temperature_tensorrtllm(
        self, client, backend: str, model: str, prompt: str
    ):
        if backend != "tensorrtllm":
            pytest.skip(reason="Only used to test vLLM-specific temperature behavior")

        responses = []
        payload1 = {
            "model": model,
            "prompt": prompt,
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
    def test_completions_seed(self, client, model: str, prompt: str):
        responses = []
        payload1 = {"model": model, "prompt": prompt, "seed": 1}
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
            # NOTE: Extensions to the spec
            ("min_tokens", -1),
            ("ignore_eos", 123),
        ],
    )
    def test_completions_invalid_sampling_parameters(
        self, client, sampling_parameter, value, model: str, prompt: str
    ):
        response = client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                sampling_parameter: value,
            },
        )

        print("Response:", response.json())
        assert response.status_code == 422

    def test_completions_empty_request(self, client):
        response = client.post("/v1/completions", json={})
        assert response.status_code == 422

    def test_completions_no_model(self, client, prompt: str):
        response = client.post("/v1/completions", json={"prompt": prompt})
        assert response.status_code == 422

    def test_completions_no_prompt(self, client, model: str):
        response = client.post("/v1/completions", json={"model": model})
        assert response.status_code == 422

    def test_completions_empty_prompt(self, client, model: str):
        response = client.post("/v1/completions", json={"model": model, "prompt": ""})

        # NOTE: Should this be validated in schema instead?
        # 400 Error returned in route handler
        assert response.status_code == 400

    def test_no_prompt(self, client, model: str):
        response = client.post("/v1/completions", json={"model": model})

        # 422 Error returned by schema validation
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "sampling_parameter_dict",
        [
            # Each individual parameter should fail for > 1 for now
            {"n": 2},
            {"best_of": 2},
            {"n": 2, "best_of": 2},
            # When individual params > 1 are supported, best_of < n should fail
            {"n": 2, "best_of": 1},
        ],
    )
    def test_completions_multiple_choices(
        self, client, sampling_parameter_dict: dict, model: str, prompt: str
    ):
        response = client.post(
            "/v1/completions",
            json={"model": model, "prompt": prompt, **sampling_parameter_dict},
        )
        print("Response:", response.json())

        # FIXME: Add support and test for success
        # Expected to fail when n or best_of > 1, only single choice supported for now
        assert response.status_code == 400
        assert "only single choice" in response.json()["detail"]

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_multi_lora(self):
        pass

    def test_usage_response(self, client, model: str, prompt: str):
        response = client.post(
            "/v1/completions",
            json={"model": model, "prompt": prompt},
        )

        assert response.status_code == 200
        usage = response.json().get("usage")
        assert usage is not None
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_completions_logprobs(self, client, backend: str, model: str, prompt: str):
        """Test logprobs parameter for completions."""
        response = client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "logprobs": 3,
                "max_tokens": 10,
            },
        )

        # TRT-LLM should raise an error
        if backend == "tensorrtllm":
            assert response.status_code == 400
            assert "logprobs are not currently supported for TensorRT-LLM backend" in response.json()["detail"]
            return

        assert response.status_code == 200
        response_json = response.json()
        
        # Check that logprobs are present in the response
        choice = response_json["choices"][0]
        assert "logprobs" in choice
        logprobs = choice["logprobs"]
        
        if logprobs is not None:
            assert "text_offset" in logprobs
            assert "token_logprobs" in logprobs
            assert "tokens" in logprobs
            assert "top_logprobs" in logprobs
            
            assert isinstance(logprobs["text_offset"], list)
            assert isinstance(logprobs["token_logprobs"], list)
            assert isinstance(logprobs["tokens"], list)
            assert isinstance(logprobs["top_logprobs"], list)
            
            # All lists should have the same length
            num_tokens = len(logprobs["tokens"])
            assert len(logprobs["text_offset"]) == num_tokens
            assert len(logprobs["token_logprobs"]) == num_tokens
            assert len(logprobs["top_logprobs"]) == num_tokens
            
            # Validate each token
            for i in range(num_tokens):
                assert isinstance(logprobs["tokens"][i], str)
                assert isinstance(logprobs["token_logprobs"][i], (int, float))
                assert isinstance(logprobs["text_offset"][i], int)
                assert isinstance(logprobs["top_logprobs"][i], dict)
                
                # Validate top_logprobs dict contains token -> logprob mappings
                for token, logprob in logprobs["top_logprobs"][i].items():
                    assert isinstance(token, str)
                    assert isinstance(logprob, (int, float))

        # Test that logprobs=0 returns no logprobs
        response = client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "logprobs": 0,
                "max_tokens": 10,
            },
        )

        assert response.status_code == 200
        response_json = response.json()
        
        # logprobs should be None when logprobs=0
        choice = response_json["choices"][0]
        assert choice.get("logprobs") is None
