#!/usr/bin/env python3
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

"""Two model instances on one GPU (instance_group count 2, gpu_device_ids "0;0")."""

import concurrent.futures
import unittest

import llmapi_test_util as util
import requests

PROMPT = "The capital of France is Paris. The capital of Germany is"


class LlmapiMultiInstanceTest(unittest.TestCase):
    def test_both_instances_are_loaded(self):
        config = requests.get(
            f"{util.HTTP_URL}/v2/models/{util.MODEL_NAME}/config", timeout=10
        ).json()
        self.assertEqual(config["instance_group"][0]["count"], 2)
        log = util.server_log_text()
        for index in (0, 1):
            self.assertIn(
                f"TRITONBACKEND_ModelInstanceInitialize: {util.MODEL_NAME}_0_{index}",
                log,
            )

    def test_concurrent_requests_succeed(self):
        def generate(_):
            result = util.grpc_infer(
                PROMPT, sampling_param_max_tokens=16, sampling_param_temperature=0.0
            )
            return util.output_text(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(generate, range(16)))
        self.assertEqual(len(outputs), 16)
        self.assertTrue(all("Berlin" in text for text in outputs), outputs)


if __name__ == "__main__":
    unittest.main()
