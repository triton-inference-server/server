#!/usr/bin/python
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

import sys

sys.path.append("../common")

import unittest

import test_util as tu
import torch
import tritonclient.http as http


class TorchAotiTest(tu.TestResultCollector):
    def _get_complex_input_shape(self):
        return (1, 16)

    def _get_complex_output_shape(self):
        return (1, 16)

    def _get_complex_input_data(self, shape):
        return [
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
        ]

    def _get_simple_input_data(self, shape, io_type):
        if io_type in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return torch.randint(low=0, high=127, size=shape, dtype=io_type).numpy()
        elif io_type in [torch.float16, torch.float32, torch.float64]:
            return torch.randn(size=shape, dtype=io_type).numpy()
        else:
            raise ValueError(f"Unsupported data type: {io_type}")

    def _get_torchvision_input_data(self, shape):
        return torch.randn(size=shape, dtype=torch.float32).numpy()

    def _dtype_to_triton_dtype(self, dtype):
        if dtype == torch.int8:
            return "INT8"
        elif dtype == torch.int16:
            return "INT16"
        elif dtype == torch.int32:
            return "INT32"
        elif dtype == torch.int64:
            return "INT64"
        elif dtype == torch.float16:
            return "FP16"
        elif dtype == torch.float32:
            return "FP32"
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def _get_simple_model_name(self, io_type):
        if io_type == torch.int8:
            return "torch_aoti_int8_int8"
        elif io_type == torch.int16:
            return "torch_aoti_int16_int16"
        elif io_type == torch.int32:
            return "torch_aoti_int32_int32"
        elif io_type == torch.int64:
            return "torch_aoti_int64_int64"
        elif io_type == torch.float16:
            return "torch_aoti_float16_float16"
        elif io_type == torch.float32:
            return "torch_aoti_float32_float32"
        else:
            raise ValueError(f"Unsupported data type: {io_type}")

    def test_complex_index(self):
        MODEL_NAME = "torch_aoti_complex_index"
        INPUT_SHAPE = self._get_complex_input_shape()
        OUTPUT_SHAPE = self._get_complex_output_shape()

        input_data = self._get_complex_input_data(INPUT_SHAPE)

        with http.InferenceServerClient("localhost:8000") as client:
            inputs = [
                http.InferInput("INPUT__0", input_data[0].shape, "INT8"),
                http.InferInput("INPUT__1", input_data[1].shape, "INT8"),
                http.InferInput("INPUT__2", input_data[2].shape, "INT8"),
                http.InferInput("INPUT__3", input_data[3].shape, "INT8"),
            ]

            inputs[0].set_data_from_numpy(input_data[0], binary_data=True)
            inputs[1].set_data_from_numpy(input_data[1], binary_data=True)
            inputs[2].set_data_from_numpy(input_data[2], binary_data=True)
            inputs[3].set_data_from_numpy(input_data[3], binary_data=True)

            output_names = [
                "OUTPUT__0",
                "OUTPUT__1",
                "OUTPUT__2",
                "OUTPUT__3",
                "OUTPUT__4",
                "OUTPUT__5",
            ]

            outputs = []
            for output_name in output_names:
                outputs.append(http.InferRequestedOutput(output_name, binary_data=True))

            output_data = []
            results = client.infer(MODEL_NAME, inputs, outputs=outputs)

            for output_name in output_names:
                output_data.append(results.as_numpy(output_name))

            self.assertEqual(len(outputs), len(output_data))
            for data in output_data:
                self.assertEqual(data.shape, OUTPUT_SHAPE)

            self.assertTrue((output_data[0] == (input_data[0] + input_data[1])).all())
            self.assertTrue((output_data[1] == input_data[0] - input_data[1]).all())
            self.assertTrue((output_data[2] == input_data[0]).all())
            self.assertTrue((output_data[3] == input_data[1]).all())
            self.assertTrue((output_data[4] == input_data[2]).all())
            self.assertTrue((output_data[5] == input_data[3]).all())

    def test_complex_named(self):
        MODEL_NAME = "torch_aoti_complex_named"
        INPUT_SHAPE = self._get_complex_input_shape()
        OUTPUT_SHAPE = self._get_complex_output_shape()

        input_data = self._get_complex_input_data(INPUT_SHAPE)

        with http.InferenceServerClient("localhost:8000") as client:
            inputs = [
                http.InferInput("ARGS[0]", input_data[0].shape, "INT8"),
                http.InferInput("ARGS[1]", input_data[1].shape, "INT8"),
                http.InferInput("ARGS[2][option1]", input_data[2].shape, "INT8"),
                http.InferInput("ARGS[2][option2]", input_data[3].shape, "INT8"),
            ]

            inputs[0].set_data_from_numpy(input_data[0], binary_data=True)
            inputs[1].set_data_from_numpy(input_data[1], binary_data=True)
            inputs[2].set_data_from_numpy(input_data[2], binary_data=True)
            inputs[3].set_data_from_numpy(input_data[3], binary_data=True)

            output_names = [
                "RESULT[AAA]",
                "RESULT[BBB][0]",
                "RESULT[BBB][1]",
                "RESULT[CCC][option1]",
                "RESULT[CCC][option2]",
                "RESULT[ZZZ]",
            ]

            outputs = []
            for output_name in output_names:
                outputs.append(http.InferRequestedOutput(output_name, binary_data=True))

            output_data = []
            results = client.infer(MODEL_NAME, inputs, outputs=outputs)

            for output_name in output_names:
                output_data.append(results.as_numpy(output_name))

            self.assertEqual(len(outputs), len(output_data))
            for data in output_data:
                self.assertEqual(data.shape, OUTPUT_SHAPE)

            self.assertTrue((output_data[0] == (input_data[0] + input_data[1])).all())
            self.assertTrue((output_data[1] == input_data[0]).all())
            self.assertTrue((output_data[2] == input_data[1]).all())
            self.assertTrue((output_data[3] == input_data[2]).all())
            self.assertTrue((output_data[4] == input_data[3]).all())
            self.assertTrue((output_data[5] == (input_data[0] - input_data[1])).all())

    def test_simple_model(self):
        io_types = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
        ]
        for io_type in io_types:
            MODEL_NAME = self._get_simple_model_name(io_type)
            INPUT_SHAPE = (16,)
            OUTPUT_SHAPE = (16,)
            TRITON_IO_TYPE = self._dtype_to_triton_dtype(io_type)

            input_data = (
                self._get_simple_input_data(INPUT_SHAPE, io_type),
                self._get_simple_input_data(INPUT_SHAPE, io_type),
            )

            with http.InferenceServerClient("localhost:8000") as client:
                inputs = [
                    http.InferInput("ARGS[0]", input_data[0].shape, TRITON_IO_TYPE),
                    http.InferInput("ARGS[1]", input_data[1].shape, TRITON_IO_TYPE),
                ]

                inputs[0].set_data_from_numpy(input_data[0], binary_data=True)
                inputs[1].set_data_from_numpy(input_data[1], binary_data=True)

                output_names = [
                    "RESULT",
                ]

                outputs = []
                for output_name in output_names:
                    outputs.append(
                        http.InferRequestedOutput(output_name, binary_data=True)
                    )

                output_data = []
                results = client.infer(MODEL_NAME, inputs, outputs=outputs)

                for output_name in output_names:
                    output_data.append(results.as_numpy(output_name))

                self.assertEqual(len(outputs), len(output_data))
                for data in output_data:
                    self.assertEqual(data.shape, OUTPUT_SHAPE)
                    self.assertTrue((data == input_data[0] + input_data[1]).all())

    def test_torchvision(self):
        MODEL_NAME = "torchvision_aoti"
        INPUT_SHAPE = (1, 3, 224, 224)
        OUTPUT_SHAPE = (1, 1000)

        input_data = self._get_torchvision_input_data(INPUT_SHAPE)
        input_data[0][0] = 1.0

        with http.InferenceServerClient("localhost:8000") as client:
            inputs = [
                http.InferInput("ARGS[0]", input_data.shape, "FP32"),
            ]

            inputs[0].set_data_from_numpy(input_data, binary_data=True)

            output_names = [
                "RESULT",
            ]

            outputs = []
            for output_name in output_names:
                outputs.append(http.InferRequestedOutput(output_name, binary_data=True))

            output_data = []
            results = client.infer(MODEL_NAME, inputs, outputs=outputs)

            for output_name in output_names:
                output_data.append(results.as_numpy(output_name))

            self.assertEqual(len(outputs), len(output_data))
            for data in output_data:
                self.assertEqual(data.shape, OUTPUT_SHAPE)
                output_tensor = torch.from_numpy(data)
                self.assertTrue(torch.isfinite(output_tensor).all().item())


if __name__ == "__main__":
    unittest.main()
