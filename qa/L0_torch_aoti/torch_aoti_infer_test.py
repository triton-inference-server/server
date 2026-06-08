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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import test_util as tu
import torch
import tritonclient.http as http
from tritonclient.utils import InferenceServerException


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
        # The simple AOTI add/sub model is compiled with a dynamic batch
        # dimension and configured with max_batch_size: 8. Exercise a range of
        # batch sizes (including 1) so we validate that batched inputs are
        # assembled and batched outputs are scattered back per-row correctly.
        batch_sizes = [1, 4, 8]
        for io_type in io_types:
            MODEL_NAME = self._get_simple_model_name(io_type)
            TRITON_IO_TYPE = self._dtype_to_triton_dtype(io_type)

            for batch_size in batch_sizes:
                INPUT_SHAPE = (batch_size, 16)
                OUTPUT_SHAPE = (batch_size, 16)

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
        # torchvision_aoti is exported with a dynamic batch dim (max_batch_size
        # 8), so exercise batching of a real, higher-rank [N,3,224,224] model.
        MODEL_NAME = "torchvision_aoti"
        with http.InferenceServerClient("localhost:8000") as client:
            for batch_size in (1, 4, 8):
                input_data = self._get_torchvision_input_data((batch_size, 3, 224, 224))
                inputs = [http.InferInput("ARGS[0]", input_data.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_data, binary_data=True)
                outputs = [http.InferRequestedOutput("RESULT", binary_data=True)]
                results = client.infer(MODEL_NAME, inputs, outputs=outputs)
                data = results.as_numpy("RESULT")
                self.assertEqual(data.shape, (batch_size, 1000))
                output_tensor = torch.from_numpy(data)
                self.assertTrue(torch.isfinite(output_tensor).all().item())

    def test_batch_size_limit(self):
        # A request whose batch exceeds max_batch_size (8) must be rejected;
        # exactly max_batch_size must succeed.
        MODEL_NAME = "torch_aoti_float32_float32"
        with http.InferenceServerClient("localhost:8000") as client:
            ok = self._get_simple_input_data((8, 16), torch.float32)
            inputs = [
                http.InferInput("ARGS[0]", ok.shape, "FP32"),
                http.InferInput("ARGS[1]", ok.shape, "FP32"),
            ]
            inputs[0].set_data_from_numpy(ok, binary_data=True)
            inputs[1].set_data_from_numpy(ok, binary_data=True)
            outputs = [http.InferRequestedOutput("RESULT", binary_data=True)]
            client.infer(MODEL_NAME, inputs, outputs=outputs)  # batch == max OK

            too_big = self._get_simple_input_data((16, 16), torch.float32)
            big_inputs = [
                http.InferInput("ARGS[0]", too_big.shape, "FP32"),
                http.InferInput("ARGS[1]", too_big.shape, "FP32"),
            ]
            big_inputs[0].set_data_from_numpy(too_big, binary_data=True)
            big_inputs[1].set_data_from_numpy(too_big, binary_data=True)
            with self.assertRaises(InferenceServerException):
                client.infer(MODEL_NAME, big_inputs, outputs=outputs)

    def _infer_add(self, model_name, a, b, triton_type="FP32"):
        # Run the two-input add model and return its RESULT output.
        with http.InferenceServerClient("localhost:8000") as client:
            inputs = [
                http.InferInput("ARGS[0]", a.shape, triton_type),
                http.InferInput("ARGS[1]", b.shape, triton_type),
            ]
            inputs[0].set_data_from_numpy(a, binary_data=True)
            inputs[1].set_data_from_numpy(b, binary_data=True)
            outputs = [http.InferRequestedOutput("RESULT", binary_data=True)]
            return client.infer(model_name, inputs, outputs=outputs).as_numpy("RESULT")

    def _execution_count(self, model_name):
        with http.InferenceServerClient("localhost:8000") as client:
            stats = client.get_inference_statistics(model_name=model_name)
            return int(stats["model_stats"][0]["execution_count"])

    def _infer_one_row(self, model_name):
        a = self._get_simple_input_data((1, 16), torch.float32)
        b = self._get_simple_input_data((1, 16), torch.float32)
        out = self._infer_add(model_name, a, b)
        self.assertTrue((out == a + b).all())

    def test_dynamic_batching_coalescing(self):
        # Fire many concurrent single-row requests and confirm the dynamic
        # batcher coalesced them into far fewer backend executions than requests.
        MODEL_NAME = "torch_aoti_float32_float32"
        num_requests = 200
        before = self._execution_count(MODEL_NAME)
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(self._infer_one_row, MODEL_NAME)
                for _ in range(num_requests)
            ]
            for future in futures:
                future.result()
        executions = self._execution_count(MODEL_NAME) - before
        self.assertGreater(executions, 0)
        self.assertLess(executions, num_requests)

    def test_multi_instance(self):
        # Concurrent requests against a 2-instance model must all be correct.
        MODEL_NAME = "torch_aoti_multi_instance_float32"
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(self._infer_one_row, MODEL_NAME) for _ in range(64)]
            for future in futures:
                future.result()

    def test_variable_shape_batching(self):
        # The variable model has dims [-1]; exercise different feature lengths
        # across batch sizes.
        MODEL_NAME = "torch_aoti_variable_float32"
        for batch_size in (1, 4, 8):
            for feature in (1, 16, 64):
                a = np.random.randn(batch_size, feature).astype(np.float32)
                b = np.random.randn(batch_size, feature).astype(np.float32)
                out = self._infer_add(MODEL_NAME, a, b)
                self.assertEqual(out.shape, (batch_size, feature))
                self.assertTrue((out == a + b).all())


class TorchAotiSequenceTest(tu.TestResultCollector):
    # The AOTI sequence model (see gen_qa_implicit_models.py) is a running
    # accumulator that resets on the sequence start and adds the correlation id
    # to each emitted output:
    #   new_state = INPUT0 + INPUT_STATE * (1 - START)
    #   OUTPUT0   = (new_state + CORRID) * READY
    # Triton's sequence scheduler synthesizes the START / READY / CORRID control
    # tensors and manages the implicit state, so the client only sends INPUT__0
    # (the correlation id is supplied via sequence_id).
    MODEL_NAME = "torch_aoti_sequence_float32"

    def _infer_step(
        self,
        client,
        seq_id,
        value,
        start,
        end,
        model=None,
        in_name="INPUT__0",
        out_name="OUTPUT__0",
    ):
        data = np.full((1, 1), value, dtype=np.float32)
        inputs = [http.InferInput(in_name, data.shape, "FP32")]
        inputs[0].set_data_from_numpy(data, binary_data=True)
        outputs = [http.InferRequestedOutput(out_name, binary_data=True)]
        result = client.infer(
            model or self.MODEL_NAME,
            inputs,
            outputs=outputs,
            sequence_id=seq_id,
            sequence_start=start,
            sequence_end=end,
        )
        return result.as_numpy(out_name)

    def test_single_sequence(self):
        seq_id = 100
        steps = [2.0, 3.0, 4.0, 5.0]
        # Output is the running sum plus the correlation id.
        expected = np.cumsum(steps) + seq_id
        with http.InferenceServerClient("localhost:8000") as client:
            for i, value in enumerate(steps):
                out = self._infer_step(
                    client,
                    seq_id=seq_id,
                    value=value,
                    start=(i == 0),
                    end=(i == len(steps) - 1),
                )
                self.assertEqual(out.shape, (1, 1))
                self.assertAlmostEqual(float(out[0, 0]), float(expected[i]), places=3)

    def test_interleaved_sequences(self):
        # Two concurrent sequences must keep independent state. Each output is
        # the per-sequence running sum plus that sequence's correlation id.
        seqs = {
            201: {"steps": [1.0, 1.0, 1.0, 1.0], "sum": 0.0},
            202: {"steps": [10.0, 20.0, 30.0, 40.0], "sum": 0.0},
        }
        with http.InferenceServerClient("localhost:8000") as client:
            num_steps = len(next(iter(seqs.values()))["steps"])
            for i in range(num_steps):
                for seq_id, st in seqs.items():
                    value = st["steps"][i]
                    st["sum"] += value
                    out = self._infer_step(
                        client,
                        seq_id=seq_id,
                        value=value,
                        start=(i == 0),
                        end=(i == num_steps - 1),
                    )
                    self.assertAlmostEqual(
                        float(out[0, 0]), st["sum"] + seq_id, places=3
                    )

    def test_many_concurrent_sequences(self):
        # Fill every batch slot with a distinct live sequence (max_batch_size is
        # 8); each must keep independent state (+ its own correlation id) as they
        # are stepped in lockstep.
        seq_ids = list(range(300, 308))  # 8 concurrent sequences == slots
        sums = {s: 0.0 for s in seq_ids}
        num_steps = 5
        with http.InferenceServerClient("localhost:8000") as client:
            for i in range(num_steps):
                for s in seq_ids:
                    value = float(s % 7 + 1)
                    sums[s] += value
                    out = self._infer_step(
                        client,
                        seq_id=s,
                        value=value,
                        start=(i == 0),
                        end=(i == num_steps - 1),
                    )
                    self.assertAlmostEqual(float(out[0, 0]), sums[s] + s, places=3)

    def test_staggered_sequences(self):
        # Sequences of different lengths that start/end at different steps, so
        # batch slots are freed and reused. Each sequence's state is independent
        # and resets on its own START.
        plans = {
            400: [1.0, 2.0],  # short
            401: [3.0, 4.0, 5.0, 6.0],  # long
            402: [10.0, 10.0, 10.0],
        }
        with http.InferenceServerClient("localhost:8000") as client:
            for seq_id, steps in plans.items():
                running = 0.0
                for i, value in enumerate(steps):
                    running += value
                    out = self._infer_step(
                        client,
                        seq_id=seq_id,
                        value=value,
                        start=(i == 0),
                        end=(i == len(steps) - 1),
                    )
                    self.assertAlmostEqual(float(out[0, 0]), running + seq_id, places=3)

    def test_initial_state_sequence(self):
        # Model relies on a declared zero initial_state (no START reset). Output
        # is the running sum (no correlation id for this variant).
        model = "torch_aoti_sequence_initstate_float32"
        steps = [2.0, 3.0, 4.0, 5.0]
        expected = np.cumsum(steps)
        with http.InferenceServerClient("localhost:8000") as client:
            for i, value in enumerate(steps):
                out = self._infer_step(
                    client,
                    seq_id=500,
                    value=value,
                    start=(i == 0),
                    end=(i == len(steps) - 1),
                    model=model,
                )
                self.assertAlmostEqual(float(out[0, 0]), float(expected[i]), places=3)

    def test_forward_interface_sequence(self):
        # Same sequence artifact, but control/state addressed via the forward
        # interface (ARGS[...] / RESULT[...]). Behaviour must match the ordinal
        # model: running sum + correlation id.
        model = "torch_aoti_sequence_forward_float32"
        seq_id = 600
        steps = [2.0, 3.0, 4.0]
        expected = np.cumsum(steps) + seq_id
        with http.InferenceServerClient("localhost:8000") as client:
            for i, value in enumerate(steps):
                out = self._infer_step(
                    client,
                    seq_id=seq_id,
                    value=value,
                    start=(i == 0),
                    end=(i == len(steps) - 1),
                    model=model,
                    in_name="ARGS[0]",
                    out_name="RESULT[0]",
                )
                self.assertAlmostEqual(float(out[0, 0]), float(expected[i]), places=3)


if __name__ == "__main__":
    unittest.main()
