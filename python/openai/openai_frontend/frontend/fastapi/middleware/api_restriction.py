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

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from utils.utils import StatusCode

# Mapping of API to their corresponding HTTP endpoints
ENDPOINT_MAPPING = {
    "inference": [
        "POST /v1/chat/completions",
        "POST /v1/completions",
        "POST /v1/embeddings",
    ],
    "model-repository": ["GET /v1/models"],
    "metrics": ["GET /metrics"],
    "health": ["GET /health/ready"],
}


class RestrictedFeatures:
    """
    Manages API endpoint restrictions and their authentication requirements.

    This class parses command-line arguments for restricted API configurations
    and provides methods to check if specific APIs are restricted
    and what authentication is required.
    """

    def __init__(self, args: list[str]):
        """
        Initialize the RestrictedFeatures with command-line arguments.

        Args:
            args: List of --openai-restricted-api argument strings
                 (e.g., [["inference", "infer-key", "infer-value"],
                         ["model-repository", "model-key", "model-value"]])
        """
        self._restrictions = {}
        self.ParseRestrictedFeatureOption(args)

    def ParseRestrictedFeatureOption(self, args):
        """
        Parse command-line arguments to extract API restrictions.

        Args:
            args: List of restriction configuration strings

        Raises:
            ValueError: If unknown API is specified or duplicate API configs are found
        """
        for apis, key, value in args:
            api_list = apis.split(",")
            for api in api_list:
                # Validate that the API is valid
                if api not in ENDPOINT_MAPPING:
                    raise ValueError(
                        f"Unknown API '{api}'. Available APIs: {list(ENDPOINT_MAPPING.keys())}"
                    )

                # Check for duplicate APIs across different arguments
                if self.IsRestricted(api):
                    raise ValueError(
                        f"restricted api '{api}' can not be specified in multiple config groups"
                    )

                self.Insert(api, (key, value))

    def RestrictionDict(self) -> dict[str, tuple[str, str]]:
        """
        Get a copy of the restrictions dictionary.

        Returns:
            dict: Copy of the restrictions mapping API names to (header_key, header_value) tuples
        """
        return self._restrictions.copy()

    def Insert(self, api: str, restriction: tuple[str, str]):
        """
        Add a restriction for a specific API.

        Args:
            api: The API name (e.g., "inference", "model-repository")
            restriction: Tuple of (header_key, header_value) for authentication
        """
        self._restrictions[api] = restriction

    def IsRestricted(self, api: str) -> bool:
        """
        Check if a specific API is restricted.

        Args:
            api: The API name to check

        Returns:
            bool: True if the API is restricted, False otherwise
        """
        return api in self._restrictions


class APIRestrictionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to restrict API endpoint access based on allowed APIs configuration.

    This middleware intercepts HTTP requests and checks if they match any restricted
    API endpoints. If a request matches a restricted endpoint, it validates the
    authentication headers before allowing the request to proceed.

    Similar to Triton Server's endpoint access control feature.
    """

    def __init__(self, app, restricted_apis: RestrictedFeatures):
        """
        Initialize the API restriction middleware.

        Args:
            app: The FastAPI application instance
            restricted_apis: RestrictedFeatures instance containing the restriction configuration
        """
        super().__init__(app)
        self.restricted_apis = restricted_apis

    def _get_auth_header(self, request: Request) -> tuple[str, str] | None:
        request_method = request.method
        request_path = request.url.path

        # Check each restricted API to see if the request matches
        for (
            restricted_api,
            auth_spec,
        ) in self.restricted_apis.RestrictionDict().items():
            # Check each endpoint in the API
            for restricted_endpoint in ENDPOINT_MAPPING[restricted_api]:
                restricted_method, restricted_path = restricted_endpoint.split(" ")

                # Match both HTTP method and path prefix
                if request_method == restricted_method and request_path.startswith(
                    restricted_path
                ):
                    return auth_spec
        return None

    async def dispatch(self, request: Request, call_next):
        """
        Main middleware dispatch method that processes each incoming request.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            Response: Either the next handler's response or a 401 authentication error
        """
        # Check if the request matches any restricted patterns
        auth_header = self._get_auth_header(request)

        # If request not restricted, proceed with the request
        if not auth_header:
            return await call_next(request)

        # Check authentication for the matching restricted endpoint
        auth_result = self._check_authentication(request, auth_header)
        if auth_result["valid"]:
            # Authentication passed, allow request to proceed
            return await call_next(request)
        else:
            # Authentication failed, return 401 error
            return JSONResponse(
                status_code=StatusCode.AUTHORIZATION_ERROR,
                content={
                    "error": {
                        "message": auth_result["message"],
                        "type": "authentication_error",
                        "code": "invalid_auth",
                    }
                },
            )

    def _check_authentication(self, request: Request, auth_header: tuple[str, str]):
        """
        Check if the request contains valid authentication headers.

        Args:
            request: The incoming HTTP request
            auth_header: Tuple of (expected_header_key, expected_header_value)

        Returns:
            dict: {"valid": bool, "message": str} - Authentication result and error message if invalid
        """
        expected_key, expected_value = auth_header

        # Get the actual header value from the request
        actual_value = request.headers.get(expected_key)

        # Validate the header value matches the expected value
        if not actual_value or actual_value != expected_value:
            return {
                "valid": False,
                "message": f"This API is restricted, expecting header '{expected_key}' with valid value",
            }

        return {"valid": True}
