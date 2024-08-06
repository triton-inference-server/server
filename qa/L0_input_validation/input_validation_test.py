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
import tritonclient.http as tritonhttpclient
import tritonclient.utils as utils
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import InferenceServerException


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
    def test_client_input_shape_validation(self):
        model_name = "simple"

        for client_type in ["http", "grpc"]:
            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

            # Infer
            inputs = []
            if client_type == "http":
                inputs.append(tritonhttpclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritonhttpclient.InferInput("INPUT1", [1, 16], "INT32"))
            else:
                inputs.append(tritongrpcclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritongrpcclient.InferInput("INPUT1", [1, 16], "INT32"))

            # Create the data for the two input tensors. Initialize the first
            # to unique integers and the second to all ones.
            input0_data = np.arange(start=0, stop=16, dtype=np.int32)
            input0_data = np.expand_dims(input0_data, axis=0)
            input1_data = np.ones(shape=(1, 16), dtype=np.int32)

            # Initialize the data
            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            # 1. Test wrong shapes with correct element counts
            # Compromised input shapes
            inputs[0].set_shape([2, 8])
            inputs[1].set_shape([2, 8])

            # If element count is correct but shape is wrong, core will return an error.
            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"unexpected shape for input 'INPUT1' for model 'simple'. Expected [-1,16], got [2,8]",
                err_str,
            )

            # 2. Test wrong shapes with wrong element counts
            # Compromised input shapes
            inputs[0].set_shape([1, 8])
            inputs[1].set_shape([1, 8])

            # If element count is wrong, client returns an error.
            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"input 'INPUT0' got unexpected elements count 16, expected 8",
                err_str,
            )

    def test_client_input_string_shape_validation(self):
        for client_type in ["http", "grpc"]:

            def identity_inference(triton_client, np_array, binary_data):
                model_name = "simple_identity"

                # Total elements no change
                inputs = []
                if client_type == "http":
                    inputs.append(
                        tritonhttpclient.InferInput("INPUT0", np_array.shape, "BYTES")
                    )
                    inputs[0].set_data_from_numpy(np_array, binary_data=binary_data)
                    inputs[0].set_shape([2, 8])
                else:
                    inputs.append(
                        tritongrpcclient.InferInput("INPUT0", np_array.shape, "BYTES")
                    )
                    inputs[0].set_data_from_numpy(np_array)
                    inputs[0].set_shape([2, 8])
                triton_client.infer(model_name=model_name, inputs=inputs)

                # Compromised input shape
                inputs[0].set_shape([1, 8])

                with self.assertRaises(InferenceServerException) as e:
                    triton_client.infer(model_name=model_name, inputs=inputs)
                err_str = str(e.exception)
                self.assertIn(
                    f"input 'INPUT0' got unexpected elements count 16, expected 8",
                    err_str,
                )

            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

            # Example using BYTES input tensor with 16 elements, where each
            # element is a 4-byte binary blob with value 0x00010203. Can use
            # dtype=np.bytes_ in this case.
            bytes_data = [b"\x00\x01\x02\x03" for i in range(16)]
            np_bytes_data = np.array(bytes_data, dtype=np.bytes_)
            np_bytes_data = np_bytes_data.reshape([1, 16])
            identity_inference(triton_client, np_bytes_data, True)  # Using binary data
            identity_inference(triton_client, np_bytes_data, False)  # Using JSON data

    def test_wrong_input_shape_tensor_size(self):
        def inference_helper(model_name, batch_size=1):
            triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
            if batch_size > 1:
                dummy_input_data = np.random.rand(batch_size, 32, 32).astype(np.float32)
            else:
                dummy_input_data = np.random.rand(32, 32).astype(np.float32)
            shape_tensor_data = np.asarray([4, 4], dtype=np.int32)

            # Pass incorrect input byte size date for shape tensor
            # Use shared memory to bypass the shape check in client library
            input_byte_size = (shape_tensor_data.size - 1) * np.dtype(np.int32).itemsize

            input_shm_handle = shm.create_shared_memory_region(
                "INPUT0_SHM",
                "/INPUT0_SHM",
                input_byte_size,
            )
            shm.set_shared_memory_region(
                input_shm_handle,
                [
                    shape_tensor_data,
                ],
            )
            triton_client.register_system_shared_memory(
                "INPUT0_SHM",
                "/INPUT0_SHM",
                input_byte_size,
            )

            inputs = [
                tritongrpcclient.InferInput(
                    "DUMMY_INPUT0",
                    dummy_input_data.shape,
                    utils.np_to_triton_dtype(np.float32),
                ),
                tritongrpcclient.InferInput(
                    "INPUT0",
                    shape_tensor_data.shape,
                    utils.np_to_triton_dtype(np.int32),
                ),
            ]
            inputs[0].set_data_from_numpy(dummy_input_data)
            inputs[1].set_shared_memory("INPUT0_SHM", input_byte_size)

            outputs = [
                tritongrpcclient.InferRequestedOutput("DUMMY_OUTPUT0"),
                tritongrpcclient.InferRequestedOutput("OUTPUT0"),
            ]

            try:
                # Perform inference
                with self.assertRaises(InferenceServerException) as e:
                    triton_client.infer(
                        model_name=model_name, inputs=inputs, outputs=outputs
                    )
                err_str = str(e.exception)
                correct_input_byte_size = (
                    shape_tensor_data.size * np.dtype(np.int32).itemsize
                )
                self.assertIn(
                    f"input byte size mismatch for input 'INPUT0' for model '{model_name}'. Expected {correct_input_byte_size}, got {input_byte_size}",
                    err_str,
                )
            finally:
                shm.destroy_shared_memory_region(input_shm_handle)
                triton_client.unregister_system_shared_memory("INPUT0_SHM")

        inference_helper(model_name="plan_nobatch_zero_1_float32_int32")
        inference_helper(model_name="plan_zero_1_float32_int32", batch_size=8)


if __name__ == "__main__":
    unittest.main()
