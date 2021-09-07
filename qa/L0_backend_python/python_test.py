#!/usr/bin/python

# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import test_util as tu
import os
import requests as httpreq

from tritonclient.utils import *
import tritonclient.http as httpclient


class PythonTest(tu.TestResultCollector):
    def _infer_help(self, model_name, shape, data_type):
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data_0 = np.array(np.random.randn(*shape), dtype=data_type)
            inputs = [
                httpclient.InferInput("INPUT0", shape,
                                      np_to_triton_dtype(input_data_0.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            self.assertTrue(np.all(input_data_0 == output0))

    def test_growth_error(self):
        # 2 MiBs
        total_byte_size = 2 * 1024 * 1024
        shape = [total_byte_size]
        model_name = 'identity_uint8_nobatch'
        dtype = np.uint8
        self._infer_help(model_name, shape, dtype)

        # 1 GiB payload leads to error in the main Python backned process.
        # Total shared memory available is 1GiB.
        total_byte_size = 1024 * 1024 * 1024
        shape = [total_byte_size]
        with self.assertRaises(InferenceServerException) as ex:
            self._infer_help(model_name, shape, dtype)
        self.assertIn("Failed to increase the shared memory pool size",
                      str(ex.exception))

        # 512 MiBs payload leads to error in the Python stub process.
        total_byte_size = 512 * 1024 * 1024
        shape = [total_byte_size]
        with self.assertRaises(InferenceServerException) as ex:
            self._infer_help(model_name, shape, dtype)
        self.assertIn("Failed to increase the shared memory pool size",
                      str(ex.exception))

        # 2 MiBs
        # Send a small paylaod to make sure it is still working properly
        total_byte_size = 2 * 1024 * 1024
        shape = [total_byte_size]
        self._infer_help(model_name, shape, dtype)

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
            batch_stat = stat['batch_stats'][0]
            self.assertFalse(
                batch_stat['batch_size'] != 8,
                f"error: expected batch_size == 8, got {batch_stat['batch_size']}"
            )
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
            input_data = np.array([[True, False, True]], dtype=bool)
            inputs = [
                httpclient.InferInput("INPUT0", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            self.assertTrue(output0 is not None)
            self.assertTrue(np.all(output0 == input_data))

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
            # output respone in this model is the number of keys in the args
            self.assertTrue(
                result.as_numpy("OUT") == 7,
                "Number of keys in the init args is not correct")

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

        for i in range(6):
            with httpclient.InferenceServerClient("localhost:8000") as client:
                input_data = np.array(['123456'], dtype=np.object_)
                inputs = [
                    httpclient.InferInput("INPUT0", shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy('OUTPUT0')
                self.assertTrue(output0 is not None)

                if i % 2 == 0:
                    self.assertTrue(output0[0] == input_data.astype(np.bytes_))
                else:
                    self.assertTrue(output0.size == 0)


    def test_non_contiguous(self):
        model_name = 'non_contiguous'
        shape = [2, 64, 84, 32, 55]
        new_shape = [64, 2, 32, 55, 84]
        shape_reorder = [1, 0, 4, 2, 3]
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_numpy = np.random.rand(*shape)
            input_numpy = input_numpy.astype(np.float32)
            inputs = [
                httpclient.InferInput("INPUT0", shape,
                                      np_to_triton_dtype(input_numpy.dtype))
            ]
            inputs[0].set_data_from_numpy(input_numpy)
            result = client.infer(model_name, inputs)
            output0 = input_numpy.reshape(new_shape)

            # Transpose the tensor to create a non-contiguous tensor.
            output1 = input_numpy.T
            output2 = np.transpose(input_numpy, shape_reorder)

            self.assertTrue(np.all(output0 == result.as_numpy('OUTPUT0')))
            self.assertTrue(np.all(output1 == result.as_numpy('OUTPUT1')))
            self.assertTrue(np.all(output2 == result.as_numpy('OUTPUT2')))


if __name__ == '__main__':
    unittest.main()
