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


class ModelConfigManager:
    def __init__(self):
        self.ensemble_model_name = "simple_graphdef_float32_float32_float32"
        self.composing_model_name = "graphdef_float32_float32_float32"
        self.model_directory = os.path.join(os.getcwd(), "models", "ensemble_models")
        self.ensemble_config_file = os.path.join(
            self.model_directory, self.ensemble_model_name, "config.pbtxt"
        )
        self.composing_config_file = os.path.join(
            self.model_directory, self.composing_model_name, "config.pbtxt"
        )
        self.response_cache_pattern = "response_cache"
        self.response_cache_config = "response_cache {\n  enable:true\n}\n"
        self.decoupled_pattern = "decoupled:true"
        self.decoupled_config = "model_transaction_policy {\n decoupled:true\n}\n"

    def _update_config(self, config_file, config_pattern, config_to_add):
        with open(config_file, "r") as f:
            config_data = f.read()
            if config_pattern not in config_data:
                with open(config_file, "w") as f:
                    config_data = config_data + config_to_add
                    f.write(config_data)

    def _remove_config(self, config_file, config_to_remove):
        with open(config_file, "r") as f:
            config_data = f.read()
        updated_config_data = re.sub(config_to_remove, "", config_data)
        with open(config_file, "w") as f:
            f.write(updated_config_data)

    def _get_all_config_files(self):
        config_files = []
        contents = os.listdir(self.model_directory)
        folders = [
            folder
            for folder in contents
            if os.path.isdir(os.path.join(self.model_directory, folder))
        ]
        for model_dir in folders:
            config_file_path = os.path.join(
                self.model_directory, str(model_dir), "config.pbtxt"
            )
            config_files.append(config_file_path)
        return config_files

    def get_ensemble_model(self):
        return self.ensemble_model_name

    def get_composing_model(self):
        return self.composing_model_name

    def enable_response_cache_ensemble_model(self):
        config_file = self.ensemble_config_file
        self._update_config(
            self.ensemble_config_file,
            self.response_cache_pattern,
            self.response_cache_config,
        )

    def enable_response_cache_all_models(self):
        config_files = self._get_all_config_files()
        for config_file in config_files:
            self._update_config(
                config_file, self.response_cache_pattern, self.response_cache_config
            )

    def enable_response_cache_composing_model(self):
        config_file = self.composing_config_file
        self._update_config(
            self.composing_config_file,
            self.response_cache_pattern,
            self.response_cache_config,
        )

    def reset_config_files(self):
        config_files = self._get_all_config_files()
        for config_file in config_files:
            self._remove_config(config_file, self.response_cache_config)
            self._remove_config(config_file, self.decoupled_config)

    def enable_decoupled_ensemble_model(self):
        self._update_config(
            self.ensemble_config_file,
            self.decoupled_pattern,
            self.decoupled_config,
        )

    def enable_decoupled_composing_model(self):
        self._update_config(
            self.composing_config_file, self.decoupled_pattern, self.decoupled_config
        )


class InferenceHandler:
    def __init__(self):
        self.config_manager = ModelConfigManager()
        self.input_tensors = self._get_input_tensors()
        self.triton_client = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )

    def _get_input_tensors(self):
        input0_data = np.ones((1, 16), dtype=np.float32)
        input1_data = np.ones((1, 16), dtype=np.float32)
        input_tensors = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            grpcclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
        ]
        input_tensors[0].set_data_from_numpy(input0_data)
        input_tensors[1].set_data_from_numpy(input1_data)
        return input_tensors

    def get_inference_statistics(self, model):
        stats = self.triton_client.get_inference_statistics(
            model_name=model, as_json=True
        )
        return stats["model_stats"][1]["inference_stats"]

    def infer(self):
        model = self.config_manager.get_ensemble_model()
        output = self.triton_client.infer(model_name=model, inputs=self.input_tensors)
        output0 = output.as_numpy("OUTPUT0")
        output1 = output.as_numpy("OUTPUT1")
        outputs = [output0, output1]
        return outputs


class InferenceValidator:
    def __init__(self):
        self.inference_handler = InferenceHandler()

    def _check_valid_output(self, output):
        if output is None:
            self.assertTrue(False, "unexpected error in inference")

    def _check_valid_output_inference(self, inference_outputs, cached_outputs):
        if not np.array_equal(inference_outputs, cached_outputs):
            self.assertTrue(False, "mismtached outputs")

    def _check_zero_stats_baseline(self, model_inference_stats):
        if "count" in model_inference_stats["success"]:
            self.assertTrue(False, "unexpected non-zero inference statistics")

    def _check_valid_stats(self, model_inference_stats):
        if "count" not in model_inference_stats["success"]:
            self.assertTrue(False, "unexpected error while retrieving statistics")

    def _check_single_cache_miss_success_inference(self, model_inference_stats):
        if (
            "count" not in model_inference_stats["cache_miss"]
            and "count" not in model_inference_stats["cache_hit"]
        ):
            self.assertTrue(False, "unexpected error with response cache")
        if (
            "count" in model_inference_stats["cache_hit"]
            and int(model_inference_stats["cache_hit"]["count"]) > 0
        ):
            self.assertTrue(False, "unexpected cache hit")
        if int(model_inference_stats["cache_miss"]["count"]) > 1:
            self.assertTrue(False, "unexpected multiple cache miss")

    def run_ensemble(self, model):
        model_inference_stats = self.inference_handler.get_inference_statistics(model)
        self._check_zero_stats_baseline(model_inference_stats)
        inference_outputs = self.inference_handler.infer()
        self._check_valid_output(inference_outputs)
        model_inference_stats = self.inference_handler.get_inference_statistics(model)
        self._check_valid_stats(model_inference_stats)
        self._check_single_cache_miss_success_inference(model_inference_stats)
        cached_outputs = self.inference_handler.infer()
        self._check_valid_output(cached_outputs)
        self._check_valid_output_inference(inference_outputs, cached_outputs)
        model_inference_stats = self.inference_handler.get_inference_statistics(model)
        self._check_valid_stats(model_inference_stats)
        return model_inference_stats


class EnsembleCacheTest(tu.TestResultCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_manager = ModelConfigManager()
        self.inference_validator = InferenceValidator()
        self.inference_handler = InferenceHandler()
        self.ensemble_model = self.config_manager.get_ensemble_model()
        self.composing_model = self.config_manager.get_composing_model()

    def setup_cache_ensemble_model(self):
        self.config_manager.enable_response_cache_ensemble_model()

    def setup_decoupled_ensemble_model(self):
        self.config_manager.enable_decoupled_ensemble_model()

    def setup_decoupled_composing_model(self):
        self.config_manager.enable_decoupled_composing_model()

    def setup_cache_composing_model(self):
        self.config_manager.enable_response_cache_composing_model()

    def setup_cache_all_models(self):
        self.config_manager.enable_response_cache_all_models()

    def reset_config(self):
        self.config_manager.reset_config_files()

    def test_ensemble_top_level_cache(self):
        model_inference_stats = self.inference_validator.run_ensemble(
            self.ensemble_model
        )
        if (
            "count" not in model_inference_stats["cache_hit"]
            or model_inference_stats["cache_hit"]["count"] != "0"
        ):
            self.assertFalse(
                False, "unexpected error in top-level ensemble response caching"
            )
        if int(model_inference_stats["cache_hit"]["count"]) > 1:
            self.assertTrue(False, "unexpected multiple cache hits")
        self.config_manager.reset_config_files()

    def test_all_models_with_cache_enabled(self):
        ensemble_model_stats = self.inference_validator.run_ensemble(
            self.ensemble_model
        )
        composing_model_stats = self.inference_handler.get_inference_statistics(
            self.composing_model
        )
        if (
            "count" not in ensemble_model_stats["cache_hit"]
            or int(ensemble_model_stats["cache_hit"]["count"]) == 0
        ):
            self.assertTrue(
                False, "unexpected error in top-level ensemble request caching"
            )
        if "count" in composing_model_stats["cache_hit"]:
            self.assertTrue(False, "unexpected composing model cache hits")
        self.config_manager.reset_config_files()

    def test_composing_model_cache_enabled(self):
        composing_model_stats = self.inference_validator.run_ensemble(
            self.composing_model
        )
        ensemble_model_stats = self.inference_handler.get_inference_statistics(
            self.ensemble_model
        )
        if "count" not in composing_model_stats["cache_miss"]:
            self.assertTrue(
                False, "unexpected error in caching composing model response"
            )
        if "count" in ensemble_model_stats["cache_hit"]:
            self.assertTrue(False, "unexpected top-level response caching")
        self.config_manager.reset_config_files()

    def test_cache_insertion_failure(self):
        ensemble_model_stats = self.inference_validator.run_ensemble(
            self.ensemble_model
        )
        if "count" in ensemble_model_stats["cache_hit"]:
            self.assertTrue(False, "unexpected successful cache insert")
        if (
            ensemble_model_stats["cache_miss"]["count"]
            != ensemble_model_stats["success"]["count"]
        ):
            self.assertTrue(False, "unexpected error with response cache")
        self.config_manager.reset_config_files()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
