#!/usr/bin/python
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Local test modules are imported after adjusting sys.path for ../common.
# flake8: noqa: E402
import sys

sys.path.append("../common")

import concurrent.futures
import unittest

import requests
from test_util import MIB, TestResultCollector

# Must match --http-parse-memory-budget in test.sh.
PARSE_MEMORY_BUDGET = 256 * MIB

# At its peak, parsing a request takes about 2 to 2.5 times the memory the
# parsed request keeps. A request with 4M numbers takes about 144 MiB, so one
# such request fits in the budget but two at the same time do not.
FITS_ELEMENTS = 4 * 1000 * 1000
# A request with 16M numbers takes about 525 MiB, more than the whole budget.
TOO_LARGE_ELEMENTS = 16 * 1000 * 1000

CONCURRENT_REQUESTS = 8
MAX_ROUNDS = 10

BUSY_ERROR = "the HTTP parse memory budget is in use by other requests"
TOO_LARGE_ERROR = (
    "bytes needs more memory to parse than the HTTP parse memory budget. "
    "Use --http-parse-memory-budget to increase the budget."
)

URL = "http://localhost:8000"
MODEL = "onnx_zero_1_float32"


def numbers(elements):
    return ",".join(["0"] * elements)


class ParseMemoryBudgetTest(TestResultCollector):
    def _infer_body(self, elements):
        # Ask for binary output, so the server does not build a JSON response.
        return (
            '{"inputs":[{"name":"INPUT0","datatype":"FP32","shape":[1,%d],'
            '"data":[%s]}],"outputs":[{"name":"OUTPUT0",'
            '"parameters":{"binary_data":true}}]}' % (elements, numbers(elements))
        )

    def _post(self, path, body):
        return requests.post(
            URL + path, headers={"Content-Type": "application/json"}, data=body
        )

    def _infer(self, elements):
        return self._post(
            "/v2/models/{}/infer".format(MODEL), self._infer_body(elements)
        )

    def _assert_status(self, response, status):
        if response.status_code != status:
            self.fail(
                "expected {}, got {}: {}".format(
                    status, response.status_code, response.content[:500]
                )
            )

    def _assert_budget_free(self):
        # A request that needs more than half of the budget still fits, so
        # earlier requests did not leave their memory counted.
        response = requests.get(URL + "/v2/health/ready")
        self._assert_status(response, 200)
        self._assert_status(self._infer(FITS_ELEMENTS), 200)

    def test_request_within_budget(self):
        self._assert_status(self._infer(FITS_ELEMENTS), 200)
        self._assert_status(self._infer(16), 200)

    def test_request_too_large_for_budget(self):
        too_large = numbers(TOO_LARGE_ELEMENTS)
        # All HTTP endpoints that take JSON share the budget.
        for path, body in [
            ("/v2/models/{}/infer".format(MODEL), self._infer_body(TOO_LARGE_ELEMENTS)),
            ("/v2/models/{}/generate".format(MODEL), '{"INPUT0":[%s]}' % too_large),
            (
                "/v2/models/{}/generate_stream".format(MODEL),
                '{"INPUT0":[%s]}' % too_large,
            ),
            ("/v2/repository/index", '{"ready":[%s]}' % too_large),
        ]:
            response = self._post(path, body)
            self._assert_status(response, 400)
            self.assertIn(TOO_LARGE_ERROR, response.text, path)
        self._assert_budget_free()

    def test_concurrent_requests_over_budget(self):
        # Requests that fit in the budget one at a time but not all together
        # either succeed or get a retryable 503.
        body = self._infer_body(FITS_ELEMENTS)
        path = "/v2/models/{}/infer".format(MODEL)
        status_codes = []
        busy_errors = []
        with concurrent.futures.ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
            for _ in range(MAX_ROUNDS):
                for response in pool.map(
                    lambda _: self._post(path, body), range(CONCURRENT_REQUESTS)
                ):
                    status_codes.append(response.status_code)
                    if response.status_code == 503:
                        busy_errors.append(response.text)
                if busy_errors:
                    break

        self.assertTrue(set(status_codes) <= {200, 503}, status_codes)
        self.assertIn(200, status_codes)
        self.assertTrue(busy_errors, "no request got 503: {}".format(status_codes))
        self.assertIn(BUSY_ERROR, busy_errors[0])
        self._assert_budget_free()

    def test_large_request_without_budget(self):
        # Run against a server started with --http-parse-memory-budget=0.
        self._assert_status(self._infer(TOO_LARGE_ELEMENTS), 200)


if __name__ == "__main__":
    unittest.main()
