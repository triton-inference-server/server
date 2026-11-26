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
import subprocess
import sys
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient

sys.path.append("../common")
import test_util as tu  # noqa: E402

MODEL = "identity_cache"


class RedisCacheEntryTest(tu.TestResultCollector):
    def setUp(self):
        self.client = grpcclient.InferenceServerClient("localhost:8001")
        self.addCleanup(self.client.close)
        # Each case expects the only key in Redis to be the one it creates.
        self._redis("FLUSHALL")

    def _redis(self, *args):
        return subprocess.run(
            [
                "redis-cli",
                "-h",
                os.environ["TRITON_REDIS_HOST"],
                "-p",
                os.environ["TRITON_REDIS_PORT"],
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    def _counts(self):
        stats = self.client.get_inference_statistics(MODEL, client_timeout=10)
        stats = stats.model_stats[0].inference_stats
        return stats.cache_hit.count, stats.cache_miss.count

    def _wait_counts(self, expected):
        deadline = time.monotonic() + 5
        while True:
            actual = self._counts()
            if actual == expected or time.monotonic() >= deadline:
                self.assertEqual(actual, expected)
                return
            time.sleep(0.01)

    def _infer(self, data):
        tensor = grpcclient.InferInput("INPUT0", data.shape, "FP32")
        tensor.set_data_from_numpy(data)
        result = self.client.infer(MODEL, [tensor], client_timeout=10)
        np.testing.assert_array_equal(result.as_numpy("OUTPUT0"), data)

    def _check_byte_size_mismatch(self, byte_size_offset):
        data = np.random.rand(16).astype(np.float32)
        hits, misses = self._counts()
        self._infer(data)
        self._wait_counts((hits, misses + 1))

        keys = self._redis("--scan").split()
        self.assertEqual(len(keys), 1, keys)
        key = keys[0]
        byte_size = int(self._redis("HSTRLEN", key, "0:b"))
        self.assertEqual(self._redis("HGET", key, "0:s"), str(byte_size))
        self._redis("HSET", key, "0:s", str(byte_size + byte_size_offset))

        # The inconsistent entry must be rejected as a miss, and the model
        # result inserted in its place must then be served from the cache.
        self._infer(data)
        self._wait_counts((hits, misses + 2))
        self._infer(data)
        self._wait_counts((hits + 1, misses + 2))

    def test_byte_size_larger_than_buffer(self):
        self._check_byte_size_mismatch(100000)

    def test_byte_size_smaller_than_buffer(self):
        self._check_byte_size_mismatch(-1)


if __name__ == "__main__":
    unittest.main()
