#!/usr/bin/python3
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

import json
import sys
import unittest

import requests

sys.path.append("../common")

import test_util as tu  # noqa: E402


class GenerateErrorStatusTest(tu.TestResultCollector):
    def _post(self, route, inputs):
        return requests.post(
            f"http://localhost:8000/v2/models/mock_llm/{route}",
            data=json.dumps(inputs),
            headers={"Accept": "text/event-stream"},
            timeout=30,
        )

    def test_model_error_status(self):
        for error_code, expected_status in ((None, 500), ("UNAVAILABLE", 503)):
            inputs = {
                "PROMPT": ["hello world"],
                "STREAM": True,
                "REPETITION": 0,
                "FAIL_LAST": True,
            }
            if error_code is not None:
                inputs["ERROR_CODE"] = error_code

            for route in ("generate", "generate_stream"):
                with self.subTest(route=route, error_code=error_code):
                    response = self._post(route, inputs)
                    self.assertEqual(response.status_code, expected_status)
                    if route == "generate_stream":
                        payload = json.loads(response.text.removeprefix("data: "))
                    else:
                        payload = response.json()
                    self.assertEqual(payload, {"error": "An Error Occurred"})


if __name__ == "__main__":
    unittest.main()
