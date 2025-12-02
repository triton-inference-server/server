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

import unittest

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class ModelReadyTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "identity_fp32"
        self.url_http = "localhost:8000"
        self.url_grpc = "localhost:8001"
        self.client_http = httpclient.InferenceServerClient(url=self.url_http)
        self.client_grpc = grpcclient.InferenceServerClient(url=self.url_grpc)

    def test_model_ready(self):
        print(f"\nTesting if model '{self.model_name}' is READY ...")

        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertTrue(
                is_ready, f"[HTTP] Model {self.model_name} should be READY but is NOT"
            )
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertTrue(
                is_ready, f"[gRPC] Model {self.model_name} should be READY but is NOT"
            )
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")

    def test_model_not_ready(self):
        print(f"\nTesting if model '{self.model_name}' is NOT READY ...")

        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertFalse(
                is_ready,
                f"[HTTP] Model {self.model_name} should be NOT READY but is READY.",
            )
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertFalse(
                is_ready,
                f"[gRPC] Model {self.model_name} should be NOT READY but is READY",
            )
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
