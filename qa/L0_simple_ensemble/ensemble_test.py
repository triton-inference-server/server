#!/usr/bin/env python3

# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient

sys.path.append("../common")
sys.path.append("../clients")

import logging
import unittest

import infer_util as iu
import numpy as np
import test_util as tu
import tritonhttpclient


# Utility function to Generate N requests with appropriate sequence flags
class RequestGenerator:
    def __init__(self, init_value, num_requests) -> None:
        self.count = 0
        self.init_value = init_value
        self.num_requests = num_requests

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        value = self.init_value + self.count
        if self.count == self.num_requests:
            raise StopIteration
        start = True if self.count == 0 else False
        end = True if self.count == self.num_requests - 1 else False
        self.count = self.count + 1
        return start, end, self.count - 1, value


class EnsembleTest(tu.TestResultCollector):
    def _get_infer_count_per_version(self, model_name):
        triton_client = tritonhttpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        stats = triton_client.get_inference_statistics(model_name)
        self.assertEqual(len(stats["model_stats"]), 2)
        infer_count = [0, 0]
        for model_stat in stats["model_stats"]:
            self.assertEqual(
                model_stat["name"], model_name, "expected stats for model " + model_name
            )
            model_version = model_stat["version"]
            if model_version == "1":
                infer_count[0] = model_stat["inference_stats"]["success"]["count"]
            elif model_version == "2":
                infer_count[1] = model_stat["inference_stats"]["success"]["count"]
            else:
                self.assertTrue(
                    False,
                    "unexpected version {} for model {}".format(
                        model_version, model_name
                    ),
                )
        return infer_count

    def test_ensemble_add_sub(self):
        for bs in (1, 8):
            iu.infer_exact(
                self, "ensemble_add_sub", (bs, 16), bs, np.int32, np.int32, np.int32
            )

        infer_count = self._get_infer_count_per_version("simple")
        # The two 'simple' versions should have the same infer count
        if infer_count[0] != infer_count[1]:
            self.assertTrue(
                False, "unexpeced different infer count for different 'simple' versions"
            )

    def test_ensemble_add_sub_one_output(self):
        for bs in (1, 8):
            iu.infer_exact(
                self,
                "ensemble_add_sub",
                (bs, 16),
                bs,
                np.int32,
                np.int32,
                np.int32,
                outputs=("OUTPUT0",),
            )

        infer_count = self._get_infer_count_per_version("simple")
        # Only 'simple' version 2 should have non-zero infer count
        # as it is in charge of producing OUTPUT0
        if infer_count[0] != 0:
            self.assertTrue(
                False, "unexpeced non-zero infer count for 'simple' version 1"
            )
        elif infer_count[1] == 0:
            self.assertTrue(False, "unexpeced zero infer count for 'simple' version 2")

    def test_ensemble_sequence_flags(self):
        request_generator = RequestGenerator(0, 3)
        # 3 request made expect the START of 1st req to be true and
        # END of last request to be true
        expected_flags = [[True, False], [False, False], [False, True]]
        response_flags = []

        def callback(start_time, result, error):
            response = result.get_response()
            arr = []
            arr.append(response.parameters["sequence_start"].bool_param)
            arr.append(response.parameters["sequence_end"].bool_param)
            response_flags.append(arr)

        start_time = time.time()
        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        triton_client.start_stream(callback=partial(callback, start_time))
        correlation_id = random.randint(1, 2**31 - 1)
        # create input tensors
        input0_data = np.random.randint(0, 100, size=(1, 16), dtype=np.int32)
        input1_data = np.random.randint(0, 100, size=(1, 16), dtype=np.int32)

        inputs = [
            grpcclient.InferInput("INPUT0", input0_data.shape, "INT32"),
            grpcclient.InferInput("INPUT1", input1_data.shape, "INT32"),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        # create output tensors
        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]
        for sequence_start, sequence_end, count, input_value in request_generator:
            triton_client.async_stream_infer(
                model_name="ensemble_add_sub_int32_int32_int32",
                inputs=inputs,
                outputs=outputs,
                request_id=f"{correlation_id}_{count}",
                sequence_id=correlation_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
        time.sleep(2)
        if expected_flags != response_flags:
            self.assertTrue(False, "unexpeced sequence flags mismatch error")

    def test_ensemble_partial_add_sub(self):
        # assert OUTPUT1 is not skipped by ensemble at this point
        output1_skipped_msg = "Composing models did not output tensor OUTPUT1"
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn(output1_skipped_msg, server_log, "test precondition not met")
        # inputs
        input0_np = np.random.randint(0, 100, size=(1, 16), dtype=np.int32)
        input1_np = np.random.randint(0, 100, size=(1, 16), dtype=np.int32)
        inputs = [
            grpcclient.InferInput("INPUT0", input0_np.shape, "INT32"),
            grpcclient.InferInput("INPUT1", input1_np.shape, "INT32"),
        ]
        inputs[0].set_data_from_numpy(input0_np)
        inputs[1].set_data_from_numpy(input1_np)
        # request all outputs
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
            grpcclient.InferRequestedOutput("OUTPUT1"),
        ]
        # infer
        model_name = "ensemble_partial_add_sub"
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            result = client.infer(model_name, inputs=inputs, outputs=outputs)
        # assert OUTPUT0 is in result
        intermediate_1_np = input1_np - input1_np
        expected_output0_np = input0_np + intermediate_1_np
        self.assertTrue(np.allclose(result.as_numpy("OUTPUT0"), expected_output0_np))
        # assert OUTPUT1 is not in result
        self.assertIsNone(result.as_numpy("OUTPUT1"))
        # assert OUTPUT1 is skipped by ensemble
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertIn(output1_skipped_msg, server_log)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
