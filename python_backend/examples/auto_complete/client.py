# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

nobatch_model_name = "nobatch_auto_complete"
batch_model_name = "batch_auto_complete"


def validate_ios(config, expected_ios, model_name):
    for io in config:
        for expected_io in expected_ios:
            if io["name"] == expected_io["name"]:
                if io["data_type"] != expected_io["data_type"]:
                    print("model '" + model_name + "' has unexpected data_type")
                    sys.exit(1)
                elif io["dims"] != expected_io["dims"]:
                    print("model '" + model_name + "' has unexpected dims")
                    sys.exit(1)


if __name__ == "__main__":
    with httpclient.InferenceServerClient("localhost:8000") as client:
        expected_max_batch_size = {
            "nobatch_auto_complete": 0,
            "batch_auto_complete": 4,
        }
        expected_inputs = [
            {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]},
            {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]},
        ]
        expected_outputs = [
            {"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [4]},
            {"name": "OUTPUT1", "data_type": "TYPE_FP32", "dims": [4]},
        ]

        models = [nobatch_model_name, batch_model_name]

        for model_name in models:
            # Validate the auto-complete model configuration
            model_config = client.get_model_config(model_name)
            if model_config["max_batch_size"] != expected_max_batch_size[model_name]:
                print("model '" + model_name + "' has unexpected max_batch_size")
                sys.exit(1)
            validate_ios(model_config["input"], expected_inputs, model_name)
            validate_ios(model_config["output"], expected_outputs, model_name)
            print(
                "'"
                + model_name
                + "' configuration matches the expected "
                + "auto complete configuration\n"
            )

    print("PASS: auto_complete")

    sys.exit(0)
