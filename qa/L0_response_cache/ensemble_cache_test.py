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

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient


class EnsembleCacheTest(tu.TestResultCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_model_name = "simple_graphdef_float32_float32_float32"
        self.composing_model_name = "graphdef_float32_float32_float32"
        self.model_directory = os.path.join(os.getcwd(), "models", "ensemble_models")
        input0_data = np.ones((1, 16), dtype=np.float32)
        input1_data = np.ones((1, 16), dtype=np.float32)
        self.input_tensors = [
            grpcclient.InferInput("INPUT0", [1, 16], "FP32"),
            grpcclient.InferInput("INPUT1", [1, 16], "FP32"),
        ]
        self.input_tensors[0].set_data_from_numpy(input0_data)
        self.input_tensors[1].set_data_from_numpy(input1_data)

    def _get_infer_result(self):
        triton_client = grpcclient.InferenceServerClient("localhost:8001", verbose=True)
        results = triton_client.infer(
            model_name=self.ensemble_model_name, inputs=self.input_tensors
        )
        stats = triton_client.get_inference_statistics(
            model_name=self.ensemble_model_name, as_json=True
        )
        return stats

    def _run_ensemble(self):
        stats = self._get_infer_result()
        stats = self._get_infer_result()
        return stats["model_stats"][1]["inference_stats"]

    def _update_config(self, config_file_path, config_pattern, config_to_add):
        with open(config_file_path, "r") as f:
            config_data = f.read()
            if config_pattern not in config_data:
                with open(config_file_path, "a") as f:
                    f.write(config_to_add)

    def _enable_response_cache_for_all_models(self):
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
            config_pattern = "response_cache"
            config_to_add = "response_cache {\n  enable:true\n}\n"
            self._update_config(config_file_path, config_pattern, config_to_add)

    def _remove_extra_config(self, config_file_path, config_to_remove):
        with open(config_file_path, "r") as f:
            config_data = f.read()
        updated_config_data = re.sub(config_to_remove, "", config_data)
        with open(config_file_path, "w") as f:
            f.write(updated_config_data)

    def test_ensemble_top_level_cache(self):
        model_inference_stats = self._run_ensemble()
        if (
            "count" not in model_inference_stats["cache_hit"]
            or int(model_inference_stats["cache_hit"]["count"]) == 0
        ):
            self.assertFalse(
                False, "unexpected error in top-level ensemble request caching"
            )

    def test_all_models_with_cache_enabled(self):
        self._enable_response_cache_for_all_models()
        model_inference_stats = self._run_ensemble()
        print(model_inference_stats)
        if (
            "count" not in model_inference_stats["cache_hit"]
            or int(model_inference_stats["cache_hit"]["count"]) == 0
        ):
            self.assertTrue(
                False, "unexpected error in top-level ensemble request caching"
            )
        if int(model_inference_stats["cache_hit"]["count"]) > int(
            model_inference_stats["success"]["count"]
        ):
            self.assertTrue(False, "unexpected composing model cache hits")

    def enable_cache_and_decoupled_ensemble_model(self):
        config_file_path = os.path.join(
            self.model_directory, self.ensemble_model_name, "config.pbtxt"
        )
        config_pattern = "decoupled:true"
        config_to_add = "model_transaction_policy {\n decoupled:true\n}\n"
        self._update_config(config_file_path, config_pattern, config_to_add)
        config_pattern = "response_cache"
        config_to_add = "response_cache {\n  enable:true\n}\n"
        self._update_config(config_file_path, config_pattern, config_to_add)

    def enable_composing_model_decoupled(self):
        config_file_path = os.path.join(
            self.model_directory, self.ensemble_model_name, "config.pbtxt"
        )
        config_to_remove = (
            r"model_transaction_policy\s*\{[^}]*decoupled\s*:\s*true[^}]*\}\n*"
        )
        self._remove_extra_config(config_file_path, config_to_remove)
        config_file_path = os.path.join(
            self.model_directory, self.composing_model_name, "config.pbtxt"
        )
        config_pattern = "decoupled:true"
        config_to_add = "model_transaction_policy {\n decoupled:true\n}\n"
        self._update_config(config_file_path, config_pattern, config_to_add)

    def remove_decoupled_config(self):
        config_file_path = os.path.join(
            self.model_directory, self.composing_model_name, "config.pbtxt"
        )
        config_to_remove = (
            r"model_transaction_policy\s*\{[^}]*decoupled\s*:\s*true[^}]*\}\n*"
        )
        self._remove_extra_config(config_file_path, config_to_remove)

    def reset_config_files(self):
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
            config_to_remove = r"response_cache\s*\{[^}]*\}\n*"
            self._remove_extra_config(config_file_path, config_to_remove)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
