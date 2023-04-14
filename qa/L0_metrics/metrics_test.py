#!/usr/bin/python
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
sys.path.append("../common")

import requests
import unittest
import test_util as tu

INF_COUNTER_PATTERNS = [
  'nv_inference_request_duration', 
  'nv_inference_queue_duration',
  'nv_inference_compute_input_duration',
  'nv_inference_compute_infer_duration',
  'nv_inference_compute_output_duration'
]
INF_SUMMARY_PATTERNS = [
  'nv_inference_request_summary',
  'nv_inference_queue_summary',
  'nv_inference_compute_input_summary',
  'nv_inference_compute_infer_summary',
  'nv_inference_compute_output_summary'
]
CACHE_COUNTER_PATTERNS = [
  'nv_cache_num_hits_per_model',
  'nv_cache_num_misses_per_model',
  'nv_cache_hit_duration_per_model',
  'nv_cache_miss_duration_per_model'
]
CACHE_SUMMARY_PATTERNS = [
  'nv_cache_hit_summary',
  'nv_cache_miss_summary'
]

class MetricsTest(tu.TestResultCollector):
    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url)
        r.raise_for_status()
        return r.text

    # Counters
    def test_inf_counters_exist(self):
        metrics = self._get_metrics()
        for metric in INF_COUNTER_PATTERNS:
            self.assertIn(metric, metrics)

    def test_inf_counters_missing(self):
        metrics = self._get_metrics()
        for metric in INF_COUNTER_PATTERNS:
            self.assertNotIn(metric, metrics)

    def test_cache_counters_exist(self):
        metrics = self._get_metrics()
        for metric in CACHE_COUNTER_PATTERNS:
            self.assertIn(metric, metrics)

    def test_cache_counters_missing(self):
        metrics = self._get_metrics()
        for metric in CACHE_COUNTER_PATTERNS:
            self.assertNotIn(metric, metrics)

    # Summaries
    def test_inf_summaries_exist(self):
        metrics = self._get_metrics()
        for metric in INF_SUMMARY_PATTERNS:
            self.assertIn(metric, metrics)

    def test_inf_summaries_missing(self):
        metrics = self._get_metrics()
        for metric in INF_SUMMARY_PATTERNS:
            self.assertNotIn(metric, metrics)

    def test_cache_summaries_exist(self):
        metrics = self._get_metrics()
        for metric in CACHE_SUMMARY_PATTERNS:
            self.assertIn(metric, metrics)

    def test_cache_summaries_missing(self):
        metrics = self._get_metrics()
        for metric in CACHE_SUMMARY_PATTERNS:
            self.assertNotIn(metric, metrics)

    def test_summaries_custom_quantiles(self):
        metrics = self._get_metrics()
        # This env var should be set by test.sh or caller
        quantile_pairs = os.environ.get("SUMMARY_QUANTILES", None)
        self.assertIsNotNone(quantile_pairs)

        quantiles = [pair.split(":")[0] for pair in quantile_pairs.split(",")]
        print(metrics)
        for quantile in quantiles:
            print(quantile)
            self.assertIn(f"quantile=\"{quantile}\"", metrics)

    # DLIS-4762: Disable request summary when caching enabled for now
    def test_inf_summaries_exist_with_cache(self):
        metrics = self._get_metrics()
        bad_patterns = ["nv_inference_request_summary"]
        ok_patterns = list(set(INF_SUMMARY_PATTERNS) - set(bad_patterns))
        for metric in ok_patterns:
            self.assertIn(metric, metrics)
        for metric in bad_patterns:
            self.assertNotIn(metric, metrics)

if __name__ == '__main__':
    unittest.main()
