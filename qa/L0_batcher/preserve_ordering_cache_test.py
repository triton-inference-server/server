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

import os
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

_MODEL_NAME = "preserve_ordering_cache"
_CLIENT_TIMEOUT_S = 10


class PreserveOrderingCacheTest(unittest.TestCase):
    def setUp(self):
        self._client = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")

    def _inputs(self, values):
        data = np.asarray(values, dtype=np.float32).reshape(1, 4)
        inputs = [grpcclient.InferInput("INPUT0", data.shape, "FP32")]
        inputs[0].set_data_from_numpy(data)
        return inputs, data

    def _infer(self, values):
        inputs, data = self._inputs(values)
        try:
            result = self._client.infer(
                _MODEL_NAME, inputs, client_timeout=_CLIENT_TIMEOUT_S
            )
        except InferenceServerException as ex:
            self.fail(f"request did not receive a response: {ex}")
        output = result.as_numpy("OUTPUT0")
        self.assertTrue(
            np.allclose(output, data),
            f"expected {data}, got {output}",
        )

    def _cache_hit_count(self):
        stats = self._client.get_inference_statistics(_MODEL_NAME, as_json=True)
        for model_stat in stats["model_stats"]:
            if model_stat["name"] == _MODEL_NAME:
                # gRPC JSON omits a statistic count when it is zero.
                cache_hit = model_stat["inference_stats"].get("cache_hit", {})
                return int(cache_hit.get("count", 0))
        self.fail(f"no statistics for model {_MODEL_NAME}")
        return 0

    def test_request_after_cache_hit_receives_response(self):
        # The repeated request is a cache hit. The following request has
        # different inputs, so it is a miss and must still return.
        repeated = [1.0, 2.0, 3.0, 4.0]
        different = [5.0, 6.0, 7.0, 8.0]

        self._infer(repeated)
        hits_before = self._cache_hit_count()
        self._infer(repeated)
        hits_after = self._cache_hit_count()
        self.assertGreater(
            hits_after,
            hits_before,
            "repeated request was not a cache hit",
        )
        self._infer(different)


if __name__ == "__main__":
    unittest.main()
