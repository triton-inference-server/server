#!/usr/bin/python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

from tritonclientutils import *
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient

class PythonTest(tu.TestResultCollector):

    def test_async_infer(self):
        client_util = httpclient
        model_name = "identity_uint8"
        request_parallelism = 4
        shape = [2, 2]
        with client_util.InferenceServerClient("localhost:8000",
                                               concurrency=request_parallelism) as client:
            input_datas = []
            requests = []
            for i in range(request_parallelism):
                input_data = (16384 * np.random.randn(*shape)).astype(np.uint8)
                input_datas.append(input_data)
                inputs = [
                    client_util.InferInput("IN", input_data.shape,
                                           np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                requests.append(client.async_infer(model_name, inputs))

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                results = requests[i].get_result()
                print(results)

                output_data = results.as_numpy("OUT")
                self.assertIsNotNone(output_data, "error: expected 'OUT'")
                self.assertTrue(np.array_equal(output_data, input_datas[i]), "error: expected output {} to match input {}".format(output_data, input_datas[i]))

            # Make sure the requests ran in parallel.
            stats = client.get_inference_statistics(model_name)
            test_cond = (len(stats['model_stats']) != 1) or (stats['model_stats'][0]['name'] != model_name)
            self.assertFalse(test_cond, "error: expected statistics for {}".format(model_name))

            stat = stats['model_stats'][0]
            self.assertFalse((stat['inference_count'] != 8) or (stat['execution_count'] != 1), "error: expected execution_count == 1 and inference_count == 8, got {} and {}".format(stat['execution_count'], stat['inference_count']))

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

            self.assertFalse(success_val != 4, "error: expected metric {} == 4, got {}".format(
                    success_str, success_val))
            self.assertFalse(infer_count_val != 8, "error: expected metric {} == 8, got {}".format(
                    infer_count_str, infer_count_val))
            self.assertFalse(infer_exec_val != 1, "error: expected metric {} == 1, got {}".format(
                    infer_exec_str, infer_exec_val))

if __name__ == '__main__':
    unittest.main()
