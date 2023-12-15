#!/usr/bin/env python3

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

import json
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class TestBlsParameters(unittest.TestCase):
    def test_bls_parameters(self):
        model_name = "bls_parameters"
        shape = [1]
        num_params = 3

        # Based on the num_params specified, the model will generate a JSON response
        # containing all the supported parameter types for num_params times recursively.
        # Make sure the model has at least num_params + 1 instances.
        expected_params = {}
        for i in range(1, num_params + 1):
            expected_params["bool_" + str(i)] = bool(i)
            expected_params["int_" + str(i)] = i
            expected_params["str_" + str(i)] = str(i)

        with grpcclient.InferenceServerClient("localhost:8001") as client:
            input_data = np.array([num_params], dtype=np.ubyte)
            inputs = [
                grpcclient.InferInput(
                    "NUMBER_PARAMETERS", shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput("PARAMETERS_AGGREGATED")]
            result = client.infer(model_name, inputs, outputs=outputs)
            params_json = str(
                result.as_numpy("PARAMETERS_AGGREGATED")[0], encoding="utf-8"
            )

        params = json.loads(params_json)
        self.assertEqual(params, expected_params)


if __name__ == "__main__":
    unittest.main()
