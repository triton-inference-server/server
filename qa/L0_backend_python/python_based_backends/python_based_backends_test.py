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

import queue
import unittest
from functools import partial
from random import randint

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

import sys
sys.path.append("../../common")
from test_util import TestResultCollector


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def create_vllm_request(
    prompt,
    request_id,
    stream,
    sampling_parameters,
    model_name,
    send_parameters_as_tensor=True,
):
    inputs = []

    inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=np.object_))

    inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(np.array([stream], dtype=bool))

    if send_parameters_as_tensor:
        sampling_parameters_data = np.array(
            [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
        )
        inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(sampling_parameters_data)

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
        "parameters": sampling_parameters,
    }

class PythonBasedBackendsTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_1 = "vllm_opt_1"
        self.vllm_model_2 = "vllm_opt_2"
        self.add_sub_model_1 = "add"
        self.add_sub_model_2 = "sub"
        self.python_model = "add_sub"
        self.pytorch_model = "add_sub_pytorch"

    def test_vllm_models(self):
        self.triton_client.load_model(self.vllm_model_1)
        self.assertTrue(self.triton_client.is_model_ready(self.vllm_model_1))
        self._test_vllm_model(self.vllm_model_1)

        self.triton_client.load_model(self.vllm_model_2)
        self.assertTrue(self.triton_client.is_model_ready(self.vllm_model_2))
        self._test_vllm_model(self.vllm_model_2)

    def test_add_sub_models(self):
        self.triton_client.load_model(self.add_sub_model_1)
        self.assertTrue(self.triton_client.is_model_ready(self.add_sub_model_1))
        self._test_add_sub_model(model_name=self.add_sub_model_1, single_output=True)

        self.triton_client.load_model(self.add_sub_model_2)
        self.assertTrue(self.triton_client.is_model_ready(self.add_sub_model_2))
        self._test_add_sub_model(model_name=self.add_sub_model_2, single_output=True)

    def test_python_model(self):
        self.triton_client.load_model(self.python_model)
        self.assertTrue(self.triton_client.is_model_ready(self.python_model))
        self._test_add_sub_model(model_name=self.python_model, shape=[16])

    def test_pytorh_model(self):
        self.triton_client.load_model(self.pytorch_model)
        self.assertTrue(self.triton_client.is_model_ready(self.pytorch_model))
        self._test_add_sub_model(model_name=self.pytorch_model)

    def _test_vllm_model(self, model_name, send_parameters_as_tensor=True):
        user_data = UserData()
        stream = False
        prompts = [
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        number_of_vllm_reqs = len(prompts)
        sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            request_data = create_vllm_request(
                prompts[i],
                i,
                stream,
                sampling_parameters,
                model_name,
                send_parameters_as_tensor,
            )
            self.triton_client.async_stream_infer(
                model_name=model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=sampling_parameters,
            )

        for i in range(number_of_vllm_reqs):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException)

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output)

        self.triton_client.stop_stream()

    def _test_add_sub_model(self, model_name, shape=[4], single_output=False):
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)

        inputs = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            grpcclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        if single_output:
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

        else:
            outputs = [
                grpcclient.InferRequestedOutput("OUTPUT0"),
                grpcclient.InferRequestedOutput("OUTPUT1"),
            ]

        response = self.triton_client.infer(
            model_name, inputs, request_id=str(randint(10, 99)), outputs=outputs
        )

        if single_output:
            if model_name == "add":
                self.assertTrue(
                    np.allclose(input0_data + input1_data, response.as_numpy("OUTPUT"))
                )
            else:
                self.assertTrue(
                    np.allclose(input0_data - input1_data, response.as_numpy("OUTPUT"))
                )
        else:
            self.assertTrue(
                np.allclose(input0_data + input1_data, response.as_numpy("OUTPUT0"))
            )
            self.assertTrue(
                np.allclose(input0_data - input1_data, response.as_numpy("OUTPUT1"))
            )

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
