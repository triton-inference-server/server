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
import re
import sys
import time
import unittest
import urllib.request
import uuid

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

sys.path.append("../common")
import test_util as tu  # noqa: E402


class CacheKeyTest(tu.TestResultCollector):
    def setUp(self):
        self.http = httpclient.InferenceServerClient(
            "localhost:8000", connection_timeout=10, network_timeout=10
        )
        self.grpc = grpcclient.InferenceServerClient("localhost:8001")
        self.addCleanup(self.http.close)
        self.addCleanup(self.grpc.close)

    def _data(self):
        # Keep cases independent even when Redis contains earlier test entries.
        return np.frombuffer(uuid.uuid4().bytes, dtype=np.int32) & 0xFFFFF

    def _counts(self, model, namespace=None):
        if namespace is None:
            stats = self.grpc.get_inference_statistics(model, client_timeout=10)
            stats = stats.model_stats[0].inference_stats
            return stats.cache_hit.count, stats.cache_miss.count

        # The statistics endpoint cannot disambiguate models named X. Metrics
        # include the repository namespace, so check each composing model there.
        namespace = os.path.join(os.environ["CACHE_KEY_REPOS"], namespace)
        with urllib.request.urlopen("http://localhost:8002/metrics", timeout=10) as r:
            metrics = r.read().decode("utf-8")
        names = ("nv_cache_num_hits_per_model", "nv_cache_num_misses_per_model")
        counts = {}
        for line in metrics.splitlines():
            sample = re.fullmatch(
                r"(nv_cache_num_(?:hits|misses)_per_model)\{(.*)\} (\S+)", line
            )
            if sample is None:
                continue
            labels = dict(re.findall(r'(\w+)="([^"]*)"', sample[2]))
            if (
                labels.get("model"),
                labels.get("namespace"),
                labels.get("version"),
            ) == (model, namespace, "1"):
                self.assertNotIn(sample[1], counts)
                counts[sample[1]] = int(float(sample[3]))
        self.assertEqual(
            set(counts), set(names), f"Missing cache metrics for {namespace}/{model}"
        )
        return tuple(counts[name] for name in names)

    def _wait_counts(self, model, expected, namespace=None):
        deadline = time.monotonic() + 5
        while True:
            actual = self._counts(model, namespace)
            if actual == expected or time.monotonic() >= deadline:
                self.assertEqual(actual, expected, (namespace, model))
                return
            time.sleep(0.01)

    def _infer(self, api, model, data, outputs=None, parameters=None, maxima=False):
        client = self.http if api is httpclient else self.grpc
        tensor = api.InferInput("INPUT", data.shape, "INT32")
        tensor.set_data_from_numpy(data)
        kwargs = {}
        if api is grpcclient:
            kwargs["client_timeout"] = 10
        result = client.infer(
            model,
            [tensor],
            outputs=None
            if outputs is None
            else [api.InferRequestedOutput(o) for o in outputs],
            parameters=parameters,
            **kwargs,
        )
        response = (
            result.get_response()
            if api is httpclient
            else result.get_response(as_json=True)
        )
        self.assertEqual(
            {o["name"] for o in response["outputs"]},
            set(outputs) if outputs is not None else {"SUM", "MAX"},
        )
        batched = model == "cache_key_batched"
        maximum = data.max(axis=-1, keepdims=batched)
        total = (
            maximum.astype(np.int64)
            if maxima
            else data.sum(axis=-1, dtype=np.int64, keepdims=batched)
        )
        total = total * int((parameters or {}).get("scale", 1))
        expected = {"SUM": total, "MAX": maximum}
        for name in set(expected).intersection(outputs or expected):
            actual = result.as_numpy(name)
            self.assertEqual(actual.shape, expected[name].shape, name)
            self.assertEqual(actual.dtype, expected[name].dtype, name)
            np.testing.assert_array_equal(actual, expected[name])
        return result

    def _miss_then_hit(self, api, model, data, **kwargs):
        hits, misses = self._counts(model)
        self._infer(api, model, data, **kwargs)
        self._wait_counts(model, (hits, misses + 1))
        self._infer(api, model, data, **kwargs)
        self._wait_counts(model, (hits + 1, misses + 1))

    def test_input_shapes(self):
        for api in (httpclient, grpcclient):
            for model in ("cache_key", "cache_key_batched", "cache_key_ensemble"):
                for shapes in (((2, 2), (1, 4), (4, 1)), ((4, 1), (1, 4), (2, 2))):
                    data = self._data()
                    for shape in shapes:
                        with self.subTest(
                            protocol=api.__name__, model=model, shape=shape
                        ):
                            self._miss_then_hit(api, model, data.reshape(shape))

    def test_parameters(self):
        for api in (httpclient, grpcclient):
            for model in ("cache_key", "cache_key_ensemble"):
                data = self._data().reshape(2, 2)
                cases = [
                    None,
                    {"scale": 1},
                    {"scale": 1000},
                    {"tag": 1},
                    {"tag": True},
                    {"tag": "1"},
                    {"tag": 1.0},
                    {"scale": 2, "tag": "test"},
                ]
                for parameters in cases:
                    with self.subTest(
                        protocol=api.__name__, model=model, parameters=parameters
                    ):
                        self._miss_then_hit(api, model, data, parameters=parameters)
                counts = self._counts(model)
                self._infer(api, model, data, parameters={"tag": "test", "scale": 2})
                self._wait_counts(model, (counts[0] + 1, counts[1]))

    def test_requested_outputs(self):
        for api in (httpclient, grpcclient):
            for model in ("cache_key", "cache_key_ensemble"):
                # Cover both default-all and explicit-all cache priming through
                # production normalization, including output-order independence.
                for all_outputs in (None, ["MAX", "SUM"]):
                    data = self._data().reshape(2, 2)
                    for outputs in (["MAX"], ["SUM"], all_outputs):
                        with self.subTest(
                            protocol=api.__name__, model=model, outputs=outputs
                        ):
                            self._miss_then_hit(api, model, data, outputs=outputs)
                    for outputs in (None, ["SUM", "MAX"], ["MAX", "SUM"]):
                        counts = self._counts(model)
                        self._infer(api, model, data, outputs=outputs)
                        self._wait_counts(model, (counts[0] + 1, counts[1]))

    def test_model_namespaces(self):
        for api in (httpclient, grpcclient):
            data = self._data().reshape(2, 2)
            counts = {repo: self._counts("X", repo) for repo in ("repoA", "repoB")}
            for repeat in range(2):
                for repo, model, maxima in (
                    ("repoA", "ensA", False),
                    ("repoB", "ensB", True),
                ):
                    with self.subTest(
                        protocol=api.__name__, namespace=repo, repeat=repeat
                    ):
                        self._infer(api, model, data, maxima=maxima)
                        hits, misses = counts[repo]
                        self._wait_counts("X", (hits + repeat, misses + 1), repo)

    def test_cache_disabled(self):
        for api in (httpclient, grpcclient):
            data = self._data()
            for shape in ((2, 2), (1, 4), (4, 1)):
                self._infer(api, "cache_key_uncached", data.reshape(shape))
            self.assertEqual(self._counts("cache_key_uncached"), (0, 0))


if __name__ == "__main__":
    unittest.main()
