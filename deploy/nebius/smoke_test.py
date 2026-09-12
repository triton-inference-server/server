#!/usr/bin/env python3
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Check an existing authenticated Triton HTTP endpoint; creates no resources."""

import argparse
import http.client
import json
import math
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODEL = "densenet_onnx"
MODEL_READY = f"/v2/models/{MODEL}/versions/1/ready"
INFER = f"/v2/models/{MODEL}/versions/1/infer"
TRANSIENT = {404, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 1024 * 1024


class CheckError(Exception):
    """Safe-to-display error that never contains tokens or response bodies."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def inference_payload():
    return {
        "inputs": [
            {
                "name": "data_0",
                "shape": [3, 224, 224],
                "datatype": "FP32",
                "data": [0.0] * (3 * 224 * 224),
            }
        ],
        "outputs": [{"name": "fc6_1"}],
    }


class Client:
    def __init__(self, url, token, allow_local_http=False):
        parsed = urllib.parse.urlsplit(url)
        local_http = (
            allow_local_http
            and parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if (
            (parsed.scheme != "https" and not local_http)
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in url)
        ):
            raise CheckError(
                "Use the returned HTTPS URL without credentials or query parameters"
            )
        if not token or any(c.isspace() for c in token):
            raise CheckError("Set a nonempty ENDPOINT_AUTH_TOKEN without whitespace")
        self.url = url.rstrip("/")
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, path, timeout, payload=None, token=None):
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise CheckError("Response exceeds the smoke test's 1 MiB limit")
                return response.status, data
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            return code, b""
        except (
            urllib.error.URLError,
            OSError,
            ValueError,
            http.client.HTTPException,
        ) as exc:
            # Keep response bodies, supplied URLs and authorization out of errors.
            raise CheckError(
                "HTTP transport failed; check URL, TLS and connectivity"
            ) from exc

    def wait_ready(self, wait_seconds, request_timeout):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            ready = True
            for path in ("/v2/health/ready", MODEL_READY):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                status, _ = self.request(
                    path, min(request_timeout, remaining), token=self.token
                )
                if status in {401, 403}:
                    raise CheckError("Readiness authentication failed (401/403)")
                if status != 200:
                    ready = False
                    if status not in TRANSIENT:
                        raise CheckError(f"Unexpected readiness HTTP status: {status}")
                    break
            else:
                if ready:
                    return
            time.sleep(max(0, min(2, deadline - time.monotonic())))
        raise CheckError(
            "Model readiness deadline exceeded; inspect Endpoint status and logs"
        )

    def check_auth(self, request_timeout):
        wrong_token = secrets.token_hex(32)
        while wrong_token == self.token:
            wrong_token = secrets.token_hex(32)
        for path, payload in (("/v2/health/ready", None), (INFER, inference_payload())):
            for token in (None, wrong_token):
                status, _ = self.request(path, request_timeout, payload, token)
                if status not in {401, 403}:
                    raise CheckError(
                        "Missing/wrong token was not rejected with 401 or 403"
                    )

    def infer(self, request_timeout):
        status, data = self.request(
            INFER, request_timeout, inference_payload(), self.token
        )
        if status != 200:
            raise CheckError(f"Inference failed with HTTP status {status}")
        try:
            response = json.loads(data)
            return validate_output(response)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise CheckError("Inference returned an invalid model response") from exc


def validate_values(values):
    if not isinstance(values, list) or len(values) != 1000:
        raise ValueError("Expected 1000 values")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError("Expected finite numbers")
    return values


def validate_output(response):
    if response["model_name"] != MODEL or response["model_version"] != "1":
        raise ValueError("Unexpected model or version")
    outputs = response["outputs"]
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Expected one output")
    output = outputs[0]
    if (
        output["name"] != "fc6_1"
        or output["datatype"] != "FP32"
        or output["shape"] != [1000]
    ):
        raise ValueError("Unexpected tensor metadata")
    return validate_values(output["data"])


def compare_reference(values, reference):
    validate_values(reference)
    if any(abs(a - b) > 1e-4 + 1e-3 * abs(b) for a, b in zip(values, reference)):
        raise CheckError(
            "Inference differs from the CPU reference (rtol=1e-3, atol=1e-4)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("ENDPOINT_URL", ""))
    parser.add_argument("--wait-seconds", type=positive_seconds, default=600)
    parser.add_argument("--request-timeout", type=positive_seconds, default=15)
    parser.add_argument(
        "--reference", type=Path, help="Optional 1000-value zero-input JSON array"
    )
    parser.add_argument(
        "--allow-local-http", action="store_true", help="Allow loopback HTTP testing"
    )
    args = parser.parse_args()
    try:
        reference = None
        if args.reference:
            reference = validate_values(
                json.loads(args.reference.read_text(encoding="utf-8"))
            )
        client = Client(
            args.url, os.environ.get("ENDPOINT_AUTH_TOKEN", ""), args.allow_local_http
        )
        client.wait_ready(args.wait_seconds, args.request_timeout)
        client.check_auth(args.request_timeout)
        values = client.infer(args.request_timeout)
        if reference is not None:
            compare_reference(values, reference)
    except CheckError as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Check failed: invalid URL or reference file", file=sys.stderr)
        return 1
    print("PASS: model ready; missing/wrong tokens rejected; 1000 finite FP32 outputs")
    if reference is not None:
        print("PASS: numerical CPU reference comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
