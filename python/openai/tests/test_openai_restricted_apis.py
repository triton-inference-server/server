#!/usr/bin/env python3

# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict, List, Optional

import pytest
import requests
from tests.utils import OpenAIServer


def make_get_request(
    base_url: str,
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
):
    """Make a GET request to the specified endpoint."""
    url = f"{base_url}{endpoint}"
    response = requests.get(url, headers=headers, timeout=timeout)
    return response


def make_chat_completion_request(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    **kwargs,
):
    """Make a POST request to the chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 10),
        **{k: v for k, v in kwargs.items() if k != "max_tokens"},
    }
    response = requests.post(url, json=data, headers=headers, timeout=timeout)
    return response


def make_completion_request(
    base_url: str,
    model: str,
    prompt: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    **kwargs,
):
    """Make a POST request to the completions endpoint."""
    url = f"{base_url}/v1/completions"
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": kwargs.get("max_tokens", 10),
        **{k: v for k, v in kwargs.items() if k != "max_tokens"},
    }
    response = requests.post(url, json=data, headers=headers, timeout=timeout)
    return response


def assert_response_success(
    response: requests.Response, expected_status: int = 200, description: str = ""
):
    """Assert that a response was successful."""
    assert (
        response.status_code == expected_status
    ), f"{description} should return {expected_status}, got {response.status_code}"


def assert_response_unauthorized(
    response: requests.Response, expected_status: int = 401, description: str = ""
):
    """Assert that a response was unauthorized."""
    assert (
        response.status_code == expected_status
    ), f"{description} should be unauthorized with {expected_status}, got {response.status_code}"


@pytest.mark.openai
class TestRestrictedAPIInvalidArguments:
    """Test cases for malformed --openai-restricted-api arguments."""

    def _test_server_startup_failure(
        self,
        malformed_api_arg,
        expected_error_pattern=None,
    ):
        """Helper method to test that server fails to start with malformed arguments."""
        args = [
            "--model-repository",
            str(
                Path(__file__).parent / f"test_models"
            ),  # Hardcode to simple models to speed up tests
        ]
        if type(malformed_api_arg[0]) == list:
            for api_arg in malformed_api_arg:
                args.append("--openai-restricted-api")
                args.extend(api_arg)
        else:
            args.append("--openai-restricted-api")
            args.extend(malformed_api_arg)

        # Server should fail to start with malformed arguments
        with pytest.raises((ValueError, Exception)) as exc_info:
            with OpenAIServer(args) as openai_server:
                pass  # Should not reach here

        if expected_error_pattern:
            assert expected_error_pattern in str(
                exc_info.value
            ), f"Expected error pattern '{expected_error_pattern}' not found in: {exc_info.value}"

    @pytest.mark.parametrize(
        "malformed_arg",
        [
            ["unknown-endpoint", "auth-key", "auth-value"],
            ["invalid,inference", "auth-key", "auth-value"],  # Mix of invalid and valid
            ["inference,unknown", "auth-key", "auth-value"],  # Mix of valid and invalid
        ],
    )
    def test_unknown_endpoint_names(self, malformed_arg):
        """Test that server handles unknown endpoint names gracefully."""
        self._test_server_startup_failure(
            malformed_arg,
            expected_error_pattern="Unknown API",
        )

    @pytest.mark.parametrize(
        "malformed_arg",
        [
            ["inference,inference", "auth-key", "auth-value"],
        ],
    )
    def test_duplicate_apis(self, malformed_arg):
        """Test that server handles duplicate APIs gracefully."""
        self._test_server_startup_failure(
            malformed_arg,
            expected_error_pattern="restricted api 'inference' can not be specified in multiple config groups",
        )

    @pytest.mark.parametrize(
        "malformed_arg",
        [
            # API with different auth specs
            [
                ["inference", "auth-key1", "value1"],
                ["inference", "auth-key2", "value2"],
            ],
            # API with same auth specs
            [["inference", "auth-key", "value"], ["inference", "auth-key", "value"]],
            # Multiple APIs with one duplicate
            [
                ["inference", "auth-key1", "value1"],
                ["model-repository", "auth-key2", "value2"],
                ["inference", "auth-key3", "value3"],
            ],
            # All APIs duplicated
            [
                ["inference", "auth-key1", "value1"],
                ["model-repository", "auth-key2", "value2"],
                ["inference", "auth-key3", "value3"],
                ["model-repository", "auth-key4", "value4"],
            ],
        ],
    )
    def test_conflict_configs(self, malformed_arg):
        """Test that server fails when duplicate APIs are specified in multiple arguments."""
        # Test cases where the same API name appears in multiple --openai-restricted-api arguments
        self._test_server_startup_failure(
            malformed_arg,
            expected_error_pattern="restricted api 'inference' can not be specified in multiple config groups",
        )


@pytest.mark.openai
class TestOpenAIServerRestrictedAPIs:
    """Test cases for OpenAI server with restricted APIs functionality."""

    @pytest.fixture(scope="class")
    def server_with_restrictions(self, model_repository, tokenizer_model, backend):
        """Start server with restricted APIs enabled."""
        args = [
            "--model-repository",
            model_repository,
            "--tokenizer",
            tokenizer_model,
            "--backend",
            backend,
            "--openai-restricted-api",
            "inference,model-repository",
            "admin-key",
            "admin-value",
        ]

        with OpenAIServer(args) as openai_server:
            yield openai_server

    def _test_restricted_endpoints(
        self, base_url, model, headers, expected_success=True, description_prefix=""
    ):
        """Helper method to test all restricted endpoints (models + inference)."""
        messages = [{"role": "user", "content": "Hello"}]

        # Test models endpoint
        response = make_get_request(base_url, "/v1/models", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Models endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Models endpoint"
            )

        # Test specific model endpoint
        response = make_get_request(base_url, f"/v1/models/{model}", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Specific model endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Specific model endpoint"
            )

        # Test chat completions endpoint
        response = make_chat_completion_request(
            base_url, model, messages, headers=headers
        )
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Chat completions endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Chat completions endpoint"
            )

        # Test completions endpoint
        response = make_completion_request(base_url, model, "Hello", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Completions endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Completions endpoint"
            )

    @pytest.mark.parametrize(
        "headers, should_succeed, description",
        [
            (None, False, "No auth"),
            ({"admin-key": "admin-value"}, True, "Valid auth"),
            ({"admin-key": "wrong-value"}, False, "Invalid auth value"),
            ({"wrong-key": "admin-value"}, False, "Invalid auth key"),
        ],
    )
    def test_restricted_endpoints_with_auth(
        self, server_with_restrictions, model, headers, should_succeed, description
    ):
        """Test restricted endpoints with different authentication scenarios."""
        base_url = server_with_restrictions.url_root

        self._test_restricted_endpoints(
            base_url,
            model,
            headers,
            expected_success=should_succeed,
            description_prefix=description,
        )

    def test_unrestricted_endpoints(self, server_with_restrictions):
        """Test that unrestricted endpoints work without authentication."""
        base_url = server_with_restrictions.url_root

        # Test metrics endpoint
        response = make_get_request(base_url, "/metrics")
        assert_response_success(response, description="Unrestricted Metrics endpoint")

        # Test health endpoint
        response = make_get_request(base_url, "/health/ready")
        assert_response_success(response, description="Unrestricted Health endpoint")


@pytest.mark.openai
class TestOpenAIServerMultipleRestrictions:
    """Test cases for OpenAI server with multiple restriction groups."""

    @pytest.fixture(scope="class")
    def server_multiple_restrictions(self, model_repository, tokenizer_model, backend):
        """Start server with multiple restriction groups."""
        args = [
            "--model-repository",
            model_repository,
            "--tokenizer",
            tokenizer_model,
            "--backend",
            backend,
            "--openai-restricted-api",
            "model-repository",
            "model-key",
            "model-value",
            "--openai-restricted-api",
            "inference",
            "infer-key",
            "infer-value",
        ]

        with OpenAIServer(args) as openai_server:
            yield openai_server

    def _test_model_repository_endpoints(
        self, base_url, model, headers, expected_success=True, description_prefix=""
    ):
        """Helper method to test model repository endpoints."""
        # Test models endpoint
        response = make_get_request(base_url, "/v1/models", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Models endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Models endpoint"
            )

        # Test specific model endpoint
        response = make_get_request(base_url, f"/v1/models/{model}", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Specific model endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Specific model endpoint"
            )

    def _test_inference_endpoints(
        self, base_url, model, headers, expected_success=True, description_prefix=""
    ):
        """Helper method to test inference endpoints."""
        messages = [{"role": "user", "content": "Hello"}]

        # Test chat completions endpoint
        response = make_chat_completion_request(
            base_url, model, messages, headers=headers
        )
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Chat completions endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Chat completions endpoint"
            )

        # Test completions endpoint
        response = make_completion_request(base_url, model, "Hello", headers=headers)
        if expected_success:
            assert_response_success(
                response, description=f"{description_prefix} Completions endpoint"
            )
        else:
            assert_response_unauthorized(
                response, description=f"{description_prefix} Completions endpoint"
            )

    def test_endpoint_groups_with_correct_auth(
        self, server_multiple_restrictions, model
    ):
        """Test that endpoint groups work with their specific authentication keys."""
        base_url = server_multiple_restrictions.url_root

        # Test model repository endpoints with model key
        model_headers = {"model-key": "model-value"}
        self._test_model_repository_endpoints(
            base_url,
            model,
            model_headers,
            expected_success=True,
            description_prefix="Model key",
        )

        # Test inference endpoints with inference key
        infer_headers = {"infer-key": "infer-value"}
        self._test_inference_endpoints(
            base_url,
            model,
            infer_headers,
            expected_success=True,
            description_prefix="Inference key",
        )

    @pytest.mark.parametrize(
        "model_headers, model_description, infer_headers, infer_description",
        [
            (None, "No auth", None, "No auth"),
            (
                {"infer-key": "infer-value"},
                "Model key for inference endpoints",
                {"model-key": "model-value"},
                "Inference key for model endpoints",
            ),
            (
                {"wrong-key": "wrong-value"},
                "Completely wrong key",
                {"wrong-key": "wrong-value"},
                "Completely wrong key",
            ),
        ],
    )
    def test_endpoint_groups_with_wrong_auth(
        self,
        server_multiple_restrictions,
        model,
        model_headers,
        model_description,
        infer_headers,
        infer_description,
    ):
        """Test that endpoint groups are blocked with wrong authentication keys."""
        base_url = server_multiple_restrictions.url_root

        # Test scenarios where wrong auth keys are used
        self._test_model_repository_endpoints(
            base_url,
            model,
            model_headers,
            expected_success=False,
            description_prefix=model_description,
        )
        self._test_inference_endpoints(
            base_url,
            model,
            infer_headers,
            expected_success=False,
            description_prefix=infer_description,
        )

    def test_unrestricted_endpoints(self, server_multiple_restrictions):
        """Test that unrestricted endpoints work without authentication."""
        base_url = server_multiple_restrictions.url_root

        # Test metrics endpoint
        response = make_get_request(base_url, "/metrics")
        assert_response_success(response, description="Unrestricted Metrics endpoint")

        # Test health endpoint
        response = make_get_request(base_url, "/health/ready")
        assert_response_success(response, description="Unrestricted Health endpoint")
