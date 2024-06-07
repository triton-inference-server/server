#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append("../common")

import unittest

import infer_util as iu
import numpy as np
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


class InputValTest(unittest.TestCase):
    def test_input_validation_required_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 0 inputs for model 'input_all_required'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 0 inputs for model 'input_optional'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_required_missing(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 1 inputs for model 'input_all_required'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))
        # Option Input is added, 2 required are missing

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 1 inputs for model 'input_optional'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_all_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        result = triton_client.infer(
            model_name="input_all_optional",
            inputs=inputs,
        )
        response = result.get_response()
        self.assertIn(str(response.outputs[0].name), "OUTPUT0")


class InputShapeTest(unittest.TestCase):
    def test_input_shape_validation(self):
        input_size = 8
        model_name = "pt_identity"
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

        # Pass
        input_data = np.arange(input_size)[None].astype(np.float32)
        inputs = [
            tritongrpcclient.InferInput(
                "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        triton_client.infer(model_name=model_name, inputs=inputs)

        # Larger input byte size than expected
        input_data = np.arange(input_size + 2)[None].astype(np.float32)
        inputs = [
            tritongrpcclient.InferInput(
                "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        # Compromised input shape
        inputs[0].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name=model_name,
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "input byte size mismatch for input 'INPUT0' for model 'pt_identity'. Expected 32, got 40",
            err_str,
        )

    def test_input_string_shape_validation(self):
        input_size = 16
        model_name = "graphdef_object_int32_int32"
        np_dtype_string = np.dtype(object)
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

        def get_input_array(input_size, np_dtype):
            rinput_dtype = iu._range_repr_dtype(np_dtype)
            input_array = np.random.randint(
                low=0, high=127, size=(1, input_size), dtype=rinput_dtype
            )

            # Convert to string type
            inn = np.array(
                [str(x) for x in input_array.reshape(input_array.size)], dtype=object
            )
            input_array = inn.reshape(input_array.shape)

            inputs = []
            inputs.append(
                tritongrpcclient.InferInput(
                    "INPUT0", input_array.shape, np_to_triton_dtype(np_dtype)
                )
            )
            inputs.append(
                tritongrpcclient.InferInput(
                    "INPUT1", input_array.shape, np_to_triton_dtype(np_dtype)
                )
            )

            inputs[0].set_data_from_numpy(input_array)
            inputs[1].set_data_from_numpy(input_array)
            return inputs

        # Input size is less than expected
        inputs = get_input_array(input_size - 2, np_dtype_string)
        # Compromised input shape
        inputs[0].set_shape((1, input_size))
        inputs[1].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name=model_name, inputs=inputs)
        err_str = str(e.exception)
        self.assertIn(
            f"expected {input_size} string elements for inference input 'INPUT1', got {input_size-2}",
            err_str,
        )

        # Input size is greater than expected
        inputs = get_input_array(input_size + 2, np_dtype_string)
        # Compromised input shape
        inputs[0].set_shape((1, input_size))
        inputs[1].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name=model_name, inputs=inputs)
        err_str = str(e.exception)
        self.assertIn(
            f"expected {input_size} string elements for inference input 'INPUT1', got {input_size+2}",
            err_str,
        )


if __name__ == "__main__":
    unittest.main()
