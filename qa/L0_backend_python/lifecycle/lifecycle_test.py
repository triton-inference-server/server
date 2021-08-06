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
sys.path.append("../../common")

import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import *
import numpy as np
import unittest


class LifecycleTest(tu.TestResultCollector):
    def test_batch_error(self):
        # The execute_error model returns an error for the first request and
        # sucessfully processes the second request.  This is making sure that
        # an error in a single request does not completely fail the batch.
        model_name = "execute_error"
        shape = [2, 2]
        request_parallelism = 2

        with httpclient.InferenceServerClient(
                "localhost:8000", concurrency=request_parallelism) as client:
            input_datas = []
            requests = []
            for i in range(request_parallelism):
                input_data = np.random.randn(*shape).astype(np.float32)
                input_datas.append(input_data)
                inputs = [
                    httpclient.InferInput("IN", input_data.shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                requests.append(client.async_infer(model_name, inputs))

            for i in range(request_parallelism):
                results = None
                if i == 0:
                    with self.assertRaises(InferenceServerException):
                        results = requests[i].get_result()
                    continue
                else:
                    results = requests[i].get_result()

                print(results)
                output_data = results.as_numpy("OUT")
                self.assertIsNotNone(output_data, "error: expected 'OUT'")
                self.assertTrue(
                    np.array_equal(output_data, input_datas[i]),
                    "error: expected output {} to match input {}".format(
                        output_data, input_datas[i]))

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
                print(e.message())
                self.assertTrue(
                    e.message().startswith(
                        "Failed to process the request(s) for model instance"),
                    "Exception message is not correct")
            else:
                self.assertTrue(
                    False,
                    "Wrong exception raised or did not raise an exception")

if __name__ == '__main__':
    unittest.main()
