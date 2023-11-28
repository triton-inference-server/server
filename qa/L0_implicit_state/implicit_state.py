#!/usr/bin/env python
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest
from builtins import range

import numpy as np
import test_util as tu
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException

BACKENDS = os.environ.get("BACKENDS", "onnx plan libtorch")


class ImplicitStateTest(tu.TestResultCollector):
    def test_no_implicit_state(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
        inputs.append(tritonhttpclient.InferInput("TEST_CASE", [1], "INT32"))
        inputs[0].set_data_from_numpy(np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([0], dtype=np.int32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="no_implicit_state",
                inputs=inputs,
                sequence_id=1,
                sequence_start=True,
            )

        err_str = str(e.exception).lower()
        self.assertIn("unable to add state 'undefined_state'", err_str)
        self.assertIn(
            "state configuration is missing for model 'no_implicit_state'", err_str
        )

    def test_wrong_implicit_state_name(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
        inputs.append(tritonhttpclient.InferInput("TEST_CASE", [1], "INT32"))
        inputs[0].set_data_from_numpy(np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([0], dtype=np.int32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="wrong_internal_state",
                inputs=inputs,
                sequence_id=2,
                sequence_start=True,
            )

        err_str = str(e.exception).lower()
        self.assertIn("state 'undefined_state' is not a valid state name", err_str)

    def test_implicit_state_single_buffer(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
        inputs.append(tritonhttpclient.InferInput("TEST_CASE", [1], "INT32"))
        inputs[0].set_data_from_numpy(np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([2], dtype=np.int32))

        triton_client.infer(
            model_name="single_state_buffer",
            inputs=inputs,
            sequence_id=2,
            sequence_start=True,
            sequence_end=False,
        )

        triton_client.infer(
            model_name="single_state_buffer",
            inputs=inputs,
            sequence_id=2,
            sequence_start=False,
            sequence_end=True,
        )

    def test_implicit_state_growable_memory(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
        inputs.append(tritonhttpclient.InferInput("TEST_CASE", [1], "INT32"))
        inputs[0].set_data_from_numpy(np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([3], dtype=np.int32))

        output = triton_client.infer(
            model_name="growable_memory",
            inputs=inputs,
            sequence_id=2,
            sequence_start=True,
            sequence_end=False,
        )
        output_state = output.as_numpy("OUTPUT_STATE")
        expected_output_state = np.zeros(output_state.shape, dtype=np.int8)
        np.testing.assert_equal(output_state, expected_output_state)

        output = triton_client.infer(
            model_name="growable_memory",
            inputs=inputs,
            sequence_id=2,
            sequence_start=False,
            sequence_end=False,
        )
        output_state = output.as_numpy("OUTPUT_STATE")
        expected_output_state = np.concatenate(
            [expected_output_state, np.ones(expected_output_state.shape, dtype=np.int8)]
        )
        np.testing.assert_equal(output_state, expected_output_state)

        output = triton_client.infer(
            model_name="growable_memory",
            inputs=inputs,
            sequence_id=2,
            sequence_start=False,
            sequence_end=False,
        )
        output_state = output.as_numpy("OUTPUT_STATE")
        expected_output_state = np.concatenate(
            [
                expected_output_state,
                np.full(
                    (expected_output_state.shape[0] // 2,), dtype=np.int8, fill_value=2
                ),
            ]
        )
        np.testing.assert_equal(output_state, expected_output_state)

    def test_no_update(self):
        # Test implicit state without updating any state
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
        inputs.append(tritonhttpclient.InferInput("TEST_CASE", [1], "INT32"))
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([1], dtype=np.int32))
        correlation_id = 3

        # Make sure the state is never updated.
        result_start = triton_client.infer(
            model_name="no_state_update",
            inputs=inputs,
            sequence_id=correlation_id,
            sequence_start=True,
        )
        self.assertEqual(result_start.as_numpy("OUTPUT")[0], 1)
        for _ in range(10):
            result = triton_client.infer(
                model_name="no_state_update", inputs=inputs, sequence_id=correlation_id
            )
            self.assertEqual(result.as_numpy("OUTPUT")[0], 1)

        _ = triton_client.infer(
            model_name="no_state_update",
            inputs=inputs,
            sequence_id=correlation_id,
            sequence_end=True,
        )
        self.assertEqual(result.as_numpy("OUTPUT")[0], 1)

    def test_request_output_not_allowed(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")

        for backend in BACKENDS.split(" "):
            inputs = []
            if backend.strip() == "libtorch":
                inputs.append(tritonhttpclient.InferInput("INPUT__0", [1], "INT32"))
            else:
                inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
            inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))

            outputs = []
            if backend.strip() == "libtorch":
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT_STATE__1"))
            else:
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT_STATE"))

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(
                    model_name=f"{backend}_nobatch_sequence_int32",
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=1,
                    sequence_start=True,
                    sequence_end=True,
                )
            if backend.strip() == "libtorch":
                self.assertIn(
                    "unexpected inference output 'OUTPUT_STATE__1' for model",
                    str(e.exception),
                )
            else:
                self.assertIn(
                    "unexpected inference output 'OUTPUT_STATE' for model",
                    str(e.exception),
                )

    def test_request_output(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        for backend in BACKENDS.split(" "):
            inputs = []
            if backend.strip() == "libtorch":
                inputs.append(tritonhttpclient.InferInput("INPUT__0", [1], "INT32"))
            else:
                inputs.append(tritonhttpclient.InferInput("INPUT", [1], "INT32"))
            inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))

            outputs = []
            if backend.strip() == "libtorch":
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT_STATE__1"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT__0"))
            else:
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT_STATE"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT"))

            result = triton_client.infer(
                model_name=f"{backend}_nobatch_sequence_int32_output",
                inputs=inputs,
                outputs=outputs,
                sequence_id=1,
                sequence_start=True,
                sequence_end=True,
            )
            if backend.strip() == "libtorch":
                self.assertTrue(result.as_numpy("OUTPUT_STATE__1")[0], 1)
                self.assertTrue(result.as_numpy("OUTPUT__0")[0], 1)
            else:
                self.assertTrue(result.as_numpy("OUTPUT_STATE")[0], 1)
                self.assertTrue(result.as_numpy("OUTPUT")[0], 1)


if __name__ == "__main__":
    unittest.main()
