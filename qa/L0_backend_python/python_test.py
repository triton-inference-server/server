#!/usr/bin/python

# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
import infer_util as iu
import test_util as tu
import os
import requests as httpreq

from tritonclient.utils import *
import tritonclient.http as httpclient


class PythonTest(tu.TestResultCollector):

    def test_async_infer(self):
        model_name = "identity_uint8"
        request_parallelism = 4
        shape = [2, 2]
        with httpclient.InferenceServerClient(
                "localhost:8000", concurrency=request_parallelism) as client:
            input_datas = []
            requests = []
            for i in range(request_parallelism):
                input_data = (16384 * np.random.randn(*shape)).astype(np.uint8)
                input_datas.append(input_data)
                inputs = [
                    httpclient.InferInput("INPUT0", input_data.shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                requests.append(client.async_infer(model_name, inputs))

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                results = requests[i].get_result()
                print(results)

                output_data = results.as_numpy("OUTPUT0")
                self.assertIsNotNone(output_data, "error: expected 'OUTPUT0'")
                self.assertTrue(
                    np.array_equal(output_data, input_datas[i]),
                    "error: expected output {} to match input {}".format(
                        output_data, input_datas[i]))

            # Make sure the requests ran in parallel.
            stats = client.get_inference_statistics(model_name)
            test_cond = (len(stats['model_stats']) !=
                         1) or (stats['model_stats'][0]['name'] != model_name)
            self.assertFalse(
                test_cond,
                "error: expected statistics for {}".format(model_name))

            stat = stats['model_stats'][0]
            self.assertFalse((stat['inference_count'] != 8) or (
                stat['execution_count'] != 1
            ), "error: expected execution_count == 1 and inference_count == 8, got {} and {}"
                             .format(stat['execution_count'],
                                     stat['inference_count']))

            # Check metrics to make sure they are reported correctly
            metrics = httpreq.get('http://localhost:8002/metrics')
            print(metrics.text)

            success_str = 'nv_inference_request_success{model="identity_uint8",version="1"}'
            infer_count_str = 'nv_inference_count{model="identity_uint8",version="1"}'
            infer_exec_str = 'nv_inference_exec_count{model="identity_uint8",version="1"}'

            success_val = None
            infer_count_val = None
            infer_exec_val = None
            for line in metrics.text.splitlines():
                if line.startswith(success_str):
                    success_val = float(line[len(success_str):])
                if line.startswith(infer_count_str):
                    infer_count_val = float(line[len(infer_count_str):])
                if line.startswith(infer_exec_str):
                    infer_exec_val = float(line[len(infer_exec_str):])

            self.assertFalse(
                success_val != 4,
                "error: expected metric {} == 4, got {}".format(
                    success_str, success_val))
            self.assertFalse(
                infer_count_val != 8,
                "error: expected metric {} == 8, got {}".format(
                    infer_count_str, infer_count_val))
            self.assertFalse(
                infer_exec_val != 1,
                "error: expected metric {} == 1, got {}".format(
                    infer_exec_str, infer_exec_val))

    def test_bool(self):
        model_name = 'identity_bool'
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data = np.array([[True, False, True]], dtype=np.bool)
            inputs = [
                httpclient.InferInput("INPUT0", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            self.assertTrue(output0 is not None)
            self.assertTrue(np.all(output0 == input_data))

    def test_infer_pymodel_error(self):
        model_name = "wrong_model"
        shape = [2, 2]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data = (16384 * np.random.randn(*shape)).astype(np.uint32)
            inputs = [
                httpclient.InferInput("IN", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            try:
                client.infer(model_name, inputs)
            except InferenceServerException as e:
                self.assertTrue(
                    e.message().startswith("GRPC Execute Failed, message:"),
                    "Exception message is not correct")
            else:
                self.assertTrue(
                    False,
                    "Wrong exception raised or did not raise an exception")

    def test_infer_pytorch(self):
        model_name = "pytorch_fp32_fp32"
        shape = [1, 1, 28, 28]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data = np.zeros(shape, dtype=np.float32)
            inputs = [
                httpclient.InferInput("IN", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            result = client.infer(model_name, inputs)
            output_data = result.as_numpy('OUT')
            self.assertIsNotNone(output_data, "error: expected 'OUT'")

            # expected inference resposne from a zero tensor
            expected_result = [
                -2.2377274, -2.3976364, -2.2464046, -2.2790744, -2.3828976,
                -2.2940576, -2.2928185, -2.340665, -2.275219, -2.292135
            ]
            self.assertTrue(np.allclose(output_data[0], expected_result),
                            'Inference result is not correct')

    def test_infer_output_error(self):
        model_name = "execute_error"
        shape = [2, 2]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data = np.zeros(shape, dtype=np.float32)
            inputs = [
                httpclient.InferInput("IN", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            try:
                client.infer(model_name, inputs)
            except InferenceServerException as e:
                print(e)
                self.assertTrue(
                    e.message().startswith("An error occured during execution"),
                    "Exception message is not correct")
            else:
                self.assertTrue(
                    False,
                    "Wrong exception raised or did not raise an exception")

    def test_init_args(self):
        model_name = "init_args"
        shape = [2, 2]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data = np.zeros(shape, dtype=np.float32)
            inputs = [
                httpclient.InferInput("IN", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            result = client.infer(model_name, inputs)
            # output response in this model is the number of keys in the args
            self.assertTrue(
                result.as_numpy("OUT") == 7,
                "Number of keys in the init args is not correct")

    def test_ensemble(self):
        model_name = "ensemble"
        shape = [16]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data_0 = np.random.random(shape).astype(np.float32)
            input_data_1 = np.random.random(shape).astype(np.float32)
            inputs = [
                httpclient.InferInput("INPUT0", input_data_0.shape,
                                      np_to_triton_dtype(input_data_0.dtype)),
                httpclient.InferInput("INPUT1", input_data_1.shape,
                                      np_to_triton_dtype(input_data_1.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            inputs[1].set_data_from_numpy(input_data_1)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            output1 = result.as_numpy('OUTPUT1')
            self.assertIsNotNone(output0)
            self.assertIsNotNone(output1)

            self.assertTrue(np.allclose(output0, 2 * input_data_0))
            self.assertTrue(np.allclose(output1, 2 * input_data_1))

        model_name = "ensemble_gpu"
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data_0 = np.random.random(shape).astype(np.float32)
            input_data_1 = np.random.random(shape).astype(np.float32)
            inputs = [
                httpclient.InferInput("INPUT0", input_data_0.shape,
                                      np_to_triton_dtype(input_data_0.dtype)),
                httpclient.InferInput("INPUT1", input_data_1.shape,
                                      np_to_triton_dtype(input_data_1.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            inputs[1].set_data_from_numpy(input_data_1)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            output1 = result.as_numpy('OUTPUT1')
            self.assertIsNotNone(output0)
            self.assertIsNotNone(output1)

            self.assertTrue(np.allclose(output0, 2 * input_data_0))
            self.assertTrue(np.allclose(output1, 2 * input_data_1))

    def test_unicode(self):
        model_name = "string"
        shape = [1]

        for i in range(3):
            with httpclient.InferenceServerClient("localhost:8000") as client:
                utf8 = '😀'
                input_data = np.array([bytes(utf8, encoding='utf-8')],
                                      dtype=np.bytes_)
                inputs = [
                    httpclient.InferInput("INPUT0", shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy('OUTPUT0')
                self.assertTrue(output0 is not None)
                self.assertTrue(output0[0] == input_data)

    def test_string(self):
        model_name = "string_fixed"
        shape = [1]

        # Each time inference is performed with a new
        # API
        for i in range(3):
            with httpclient.InferenceServerClient("localhost:8000") as client:
                sample_input = '123456'
                input_data = np.array([sample_input], dtype=np.object_)
                inputs = [
                    httpclient.InferInput("INPUT0", shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy('OUTPUT0')
                self.assertTrue(output0 is not None)
                self.assertTrue(output0[0] == input_data.astype(np.bytes_))


if __name__ == '__main__':
    unittest.main()
