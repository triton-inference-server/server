#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append('../common')

import argparse
import numpy as np
import os
from builtins import range
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException
import unittest
import test_util as tu

BACKENDS = os.environ.get('BACKENDS', "onnx plan")


class ImplicitStateTest(tu.TestResultCollector):

    def test_no_implicit_state(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], 'INT32'))
        inputs.append(tritonhttpclient.InferInput('TEST_CASE', [1], 'INT32'))
        inputs[0].set_data_from_numpy(
            np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([0], dtype=np.int32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name="no_implicit_state",
                                inputs=inputs,
                                sequence_id=1,
                                sequence_start=True)

        self.assertEqual(
            str(e.exception),
            "unable to add state 'undefined_state'. State configuration is missing for model 'no_implicit_state'."
        )

    def test_wrong_implicit_state_name(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], 'INT32'))
        inputs.append(tritonhttpclient.InferInput('TEST_CASE', [1], 'INT32'))
        inputs[0].set_data_from_numpy(
            np.random.randint(5, size=[1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([0], dtype=np.int32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name="wrong_internal_state",
                                inputs=inputs,
                                sequence_id=2,
                                sequence_start=True)

        self.assertEqual(str(e.exception),
                         "state 'undefined_state' is not a valid state name.")

    def test_no_update(self):
        # Test implicit state without updating any state
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], 'INT32'))
        inputs.append(tritonhttpclient.InferInput('TEST_CASE', [1], 'INT32'))
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.asarray([1], dtype=np.int32))
        correlation_id = 3

        # Make sure the state is never updated.
        result_start = triton_client.infer(model_name="no_state_update",
                                           inputs=inputs,
                                           sequence_id=correlation_id,
                                           sequence_start=True)
        self.assertEqual(result_start.as_numpy('OUTPUT')[0], 1)
        for _ in range(10):
            result = triton_client.infer(model_name="no_state_update",
                                         inputs=inputs,
                                         sequence_id=correlation_id)
            self.assertEqual(result.as_numpy('OUTPUT')[0], 1)

        result_start = triton_client.infer(model_name="no_state_update",
                                           inputs=inputs,
                                           sequence_id=correlation_id,
                                           sequence_end=True)
        self.assertEqual(result.as_numpy('OUTPUT')[0], 1)

    def test_request_output_not_allowed(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], 'INT32'))
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))

        outputs = []
        outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT_STATE'))

        for backend in BACKENDS.split(" "):
            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(
                    model_name=f"{backend}_nobatch_sequence_int32",
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=1,
                    sequence_start=True,
                    sequence_end=True)
            self.assertTrue(
                str(e.exception).startswith(
                    "unexpected inference output 'OUTPUT_STATE' for model"))

    def test_request_output(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], 'INT32'))
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.int32))

        outputs = []
        outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT_STATE'))
        outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT'))

        for backend in BACKENDS.split(" "):
            result = triton_client.infer(
                model_name=f"{backend}_nobatch_sequence_int32_output",
                inputs=inputs,
                outputs=outputs,
                sequence_id=1,
                sequence_start=True,
                sequence_end=True)
            self.assertTrue(result.as_numpy('OUTPUT_STATE')[0], 1)
            self.assertTrue(result.as_numpy('OUTPUT')[0], 1)


if __name__ == '__main__':
    unittest.main()
