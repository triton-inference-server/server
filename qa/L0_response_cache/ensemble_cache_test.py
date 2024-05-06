#!/usr/bin/env python3
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append("../common")
sys.path.append("../clients")
import logging
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

RESPONSE_CACHE_PATTERN = "response_cache"
RESPONSE_CACHE_CONFIG = "response_cache {\n  enable:true\n}\n"


class EnsembleCacheTest(tu.TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        self.ensemble_model = "simple_graphdef_float32_float32_float32"
        self.composing_model = "graphdef_float32_float32_float32"
        self.model_directory = os.path.join(os.getcwd(), "models", "ensemble_models")
        self.ensemble_config_file = os.path.join(
            self.model_directory, self.ensemble_model, "config.pbtxt"
        )
        self.composing_config_file = os.path.join(
            self.model_directory, self.composing_model, "config.pbtxt"
        )
        input0_data = np.ones((1, 16), dtype=np.float32)
        input1_data = np.ones((1, 16), dtype=np.float32)
        self.input_tensors = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            grpcclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
        ]
        self.input_tensors[0].set_data_from_numpy(input0_data)
        self.input_tensors[1].set_data_from_numpy(input1_data)

    def _update_config(self, config_file, config_pattern, config_to_add):
        # Utility function to update config files as per testcase
        with open(config_file, "r") as f:
            config_data = f.read()
            if config_pattern not in config_data:
                with open(config_file, "w") as f:
                    config_data += config_to_add
                    f.write(config_data)

    def _remove_config(self, config_file, config_to_remove):
        # Utility function to remove extra added config from the config files
        with open(config_file, "r") as f:
            config_data = f.read()
        updated_config_data = re.sub(config_to_remove, "", config_data)
        with open(config_file, "w") as f:
            f.write(updated_config_data)

    def _reset_config_files(self):
        # Utility function to reset all config files to original
        self._remove_config(self.ensemble_config_file, RESPONSE_CACHE_CONFIG)
        self._remove_config(self.composing_config_file, RESPONSE_CACHE_CONFIG)

    def _run_ensemble(self):
        # Run the ensemble pipeline and validate output
        output = self.triton_client.infer(
            model_name=self.ensemble_model, inputs=self.input_tensors
        )
        self.assertIsNotNone(
            output,
            f"Unexpected error: Inference result is None for model '{self.ensemble_model}'. Expected non-null output.",
        )
        output0 = output.as_numpy("OUTPUT0")
        output1 = output.as_numpy("OUTPUT1")
        outputs = [output0, output1]
        return outputs

    def _get_model_statistics(self, model):
        # Get the stats for the requested model
        model_stats = self.triton_client.get_inference_statistics(
            model_name=model, as_json=True
        )

        """
        The models used have two versions, version 1 and version 3.
        Since, model_version is set to -1 in config.pbtxt, the highest version is loaded
        which is version 3.
        model_stats has inference stats for version 1 at index 0 and inference stats for version 3 at index 1.
        """
        return model_stats["model_stats"][1]["inference_stats"]

    def _run_inference_and_validate(self, model):
        """
        Helper function that takes model as a parameter to verify the corresponding model's stats
        The passed model is composing model for test case `test_ensemble_composing_model_cache_enabled`
        For other testcases, the top-level ensemble model stats are verified.
            * loads the simple_graphdef_float32_float32_float32 and graphdef_float32_float32_float32
              and verifies if they are loaded properly.
            * Checks the initial statistics of the model passed in the parameter
              Expected - baseline statistics to be all empty metrics since
            * Calls the run_ensemble function to run the ensemble pipeline.
            * Verifies the stats after first inference. Expected single cache miss.
            * Calls the run_ensemble function to run the ensemble pipeline again.
            * Checks if returned output is equal to th output of first inference.
        """
        self.triton_client.load_model(self.ensemble_model)
        self.assertTrue(
            self.triton_client.is_model_ready(self.ensemble_model),
            f"Failed to load ensemble model '{self.ensemble_model}'",
        )
        self.triton_client.load_model(self.composing_model)
        self.assertTrue(
            self.triton_client.is_model_ready(self.composing_model),
            f"Failed to load composing model '{self.composing_model}'",
        )

        model_stats_initial = self._get_model_statistics(model)
        self.assertNotIn(
            "count",
            model_stats_initial["success"],
            f"No inference stats expected initially for model '{model}'",
        )

        inference_output = self._run_ensemble()
        model_stats = self._get_model_statistics(model)
        self.assertIn(
            "count", model_stats["success"], f"Failed inference for model '{model}'"
        )
        self.assertIn(
            "count",
            model_stats["cache_miss"],
            f"No cache miss recorded for model '{model}', expected exactly one cache miss",
        )
        self.assertEqual(
            model_stats["cache_miss"]["count"],
            "1",
            f"Expected exactly one cache miss in model '{model}', found {model_stats['cache_miss']['count']}",
        )

        cached_output = self._run_ensemble()
        self.assertTrue(
            np.array_equal(inference_output, cached_output),
            f"Cache response does not match actual inference output for model '{model}'",
        )

    def test_ensemble_top_level_response_cache(self):
        """
        Test top level response caching when response cache enabled only in
        ensemble model's config file.
        Expected result: One cache hit in ensemble model stats. No cache related metric counts in
        composing model stats.
        """
        self._update_config(
            self.ensemble_config_file, RESPONSE_CACHE_PATTERN, RESPONSE_CACHE_CONFIG
        )
        self._run_inference_and_validate(self.ensemble_model)
        ensemble_model_stats = self._get_model_statistics(self.ensemble_model)
        expected_cache_hit_count = "1"
        actual_cache_hit_count = ensemble_model_stats["cache_hit"]["count"]
        self.assertIn(
            "count",
            ensemble_model_stats["success"],
            f"Failed inference recorded for ensemble model '{self.ensemble_model}'. Expected successful inference.",
        )
        self.assertIn(
            "count",
            ensemble_model_stats["cache_hit"],
            f"No cache hit recorded for ensemble model '{self.ensemble_model}'. Expected exactly one cache hit.",
        )
        self.assertEqual(
            actual_cache_hit_count,
            expected_cache_hit_count,
            f"Unexpected number of cache hits recorded for ensemble model '{self.ensemble_model}'. Expected exactly one cache hit.",
        )

    def test_ensemble_all_models_cache_enabled(self):
        """
        Test top level response caching when response cache enabled in
        all the models.
        Expected result: One cache hit in ensemble model stats. No cache hit in composing model stats.
        """
        self._update_config(
            self.ensemble_config_file, RESPONSE_CACHE_PATTERN, RESPONSE_CACHE_CONFIG
        )
        self._update_config(
            self.composing_config_file, RESPONSE_CACHE_PATTERN, RESPONSE_CACHE_CONFIG
        )
        self._run_inference_and_validate(self.ensemble_model)
        ensemble_model_stats = self._get_model_statistics(self.ensemble_model)
        composing_model_stats = self._get_model_statistics(self.composing_model)
        expected_cache_hit_count = "1"
        actual_cache_hit_count = ensemble_model_stats["cache_hit"]["count"]
        self.assertIn(
            "count",
            ensemble_model_stats["success"],
            f"Failed inference recorded for ensemble model '{self.ensemble_model}'. Expected successful inference.",
        )
        self.assertIn(
            "count",
            ensemble_model_stats["cache_hit"],
            f"No cache hit recorded for ensemble model '{self.ensemble_model}'. Expected exactly one cache hit.",
        )
        self.assertNotIn(
            "count",
            composing_model_stats["cache_hit"],
            f"Unexpected cache hit recorded for composing model '{self.composing_model}'. Expected top-level response in cache for ensemble model '{self.ensemble_model}'.",
        )
        self.assertEqual(
            actual_cache_hit_count,
            expected_cache_hit_count,
            f"Unexpected number of cache hits recorded for ensemble model '{self.ensemble_model}'. Expected exactly one cache hit.",
        )

    def test_ensemble_composing_model_cache_enabled(self):
        """
        Test caching behavior when response cache enabled only in
        composing model's config file.
        Expected result: One cache hit in composing model stats. No cache related metric counts in
        ensemble model stats.
        """
        self._update_config(
            self.composing_config_file, RESPONSE_CACHE_PATTERN, RESPONSE_CACHE_CONFIG
        )
        self._run_inference_and_validate(self.composing_model)
        ensemble_model_stats = self._get_model_statistics(self.ensemble_model)
        composing_model_stats = self._get_model_statistics(self.composing_model)
        self.assertIn(
            "count",
            composing_model_stats["success"],
            f"Failed inference recorded for ensemble model '{self.composing_model}'. Expected successful inference.",
        )
        self.assertIn(
            "count",
            composing_model_stats["cache_hit"],
            f"No cache hit recorded for ensemble model '{self.composing_model}'. Expected exactly one cache hit.",
        )
        self.assertNotIn(
            "count",
            ensemble_model_stats["cache_hit"],
            f"Unexpected number of cache hits recorded for ensemble model '{self.ensemble_model}'. Expected empty cache metrics",
        )

    def test_ensemble_cache_insertion_failure(self):
        """
        Test cache insertion failure with cache enabled in
        ensemble model's config file.
        Expected result: Two cache miss in ensemble model stats indicating request/response not inserted into cache
        Reason: The data (input tensors, output tensors and other model information) to be inserted in cache is bigger cache size.
        """
        self._update_config(
            self.ensemble_config_file, RESPONSE_CACHE_PATTERN, RESPONSE_CACHE_CONFIG
        )
        self._run_inference_and_validate(self.ensemble_model)
        ensemble_model_stats = self._get_model_statistics(self.ensemble_model)
        expected_cache_miss_count = "2"
        actual_cache_miss_count = ensemble_model_stats["cache_miss"]["count"]
        self.assertIn(
            "count",
            ensemble_model_stats["success"],
            f"Failed inference recorded for ensemble model '{self.ensemble_model}'. Expected successful inference.",
        )
        self.assertNotIn(
            "count",
            ensemble_model_stats["cache_hit"],
            f"No cache hit recorded for ensemble model '{self.ensemble_model}'. Expected exactly one cache hit.",
        )
        self.assertIn(
            "count",
            ensemble_model_stats["cache_miss"],
            f"No cache miss recorded in ensemble model '{self.ensemble_model}'. Expected cache miss.",
        )
        self.assertEqual(
            actual_cache_miss_count,
            expected_cache_miss_count,
            f"Unexpected number of cache misses recorded in ensemble model '{self.ensemble_model}'. Expected exactly {expected_cache_miss_count} cache misses for two inference requests, but found {actual_cache_miss_count}.",
        )

    def tearDown(self):
        self._reset_config_files()
        self.triton_client.close()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
