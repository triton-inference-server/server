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

sys.path.append("../../common")

import test_util as tu
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import numpy as np
import unittest
from functools import partial
import queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class DecoupledTest(tu.TestResultCollector):

    def test_decoupled_execute_error(self):
        # The decoupled_execute_error model returns an error for the first
        # request and sucessfully processes the second request. This is making
        # sure that an error in a single request does not completely fail the
        # batch.

        model_name = "decoupled_execute_error"
        shape = [2, 2]
        number_of_requests = 2
        user_data = UserData()
        with grpcclient.InferenceServerClient(
                "localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            input_datas = []
            for i in range(number_of_requests):
                input_data = np.random.randn(*shape).astype(np.float32)
                input_datas.append(input_data)
                inputs = [
                    grpcclient.InferInput("IN", input_data.shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                triton_client.async_stream_infer(model_name=model_name,
                                                 inputs=inputs)

            for i in range(number_of_requests):
                result = user_data._completed_requests.get()
                if i == 0:
                    self.assertIs(type(result), InferenceServerException)
                    continue

                print(result)
                output_data = result.as_numpy("OUT")
                self.assertIsNotNone(output_data, "error: expected 'OUT'")
                self.assertTrue(
                    np.array_equal(output_data, input_datas[i]),
                    "error: expected output {} to match input {}".format(
                        output_data, input_datas[i]))

    def test_decoupled_bls(self):
        # Test combinations of BLS and decoupled API in Python backend.
        model_name = "decoupled_bls"
        shape = [1, 2]
        user_data = UserData()
        with grpcclient.InferenceServerClient(
                "localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            input_datas = []
            input_data = np.random.randn(*shape).astype(np.float32)
            input_datas.append(input_data)
            inputs = [
                grpcclient.InferInput("IN", input_data.shape,
                                      np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            triton_client.async_stream_infer(model_name=model_name,
                                             inputs=inputs)

            # Check the results of the decoupled model using BLS
            def check_result(result):
                # Make sure the result is not an exception
                self.assertIsNot(type(result), InferenceServerException)

                output_data = result.as_numpy("OUT")
                self.assertIsNotNone(output_data, "error: expected 'OUT'")
                self.assertTrue(
                    np.array_equal(output_data, input_data),
                    "error: expected output {} to match input {}".format(
                        output_data, input_data))

            result = user_data._completed_requests.get()
            check_result(result)

    def test_decoupled_return_response_error(self):
        model_name = "decoupled_return_response_error"
        shape = [16]
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            client.start_stream(callback=partial(callback, user_data))
            input_data_0 = np.random.random(shape).astype(np.float32)
            input_data_1 = np.random.random(shape).astype(np.float32)
            inputs = [
                grpcclient.InferInput("INPUT0", input_data_0.shape,
                                      np_to_triton_dtype(input_data_0.dtype)),
                grpcclient.InferInput("INPUT1", input_data_1.shape,
                                      np_to_triton_dtype(input_data_1.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            inputs[1].set_data_from_numpy(input_data_1)
            client.async_stream_infer(model_name=model_name, inputs=inputs)
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                self.assertEqual(
                    data_item.message(),
                    "Python model 'decoupled_return_response_error_0' is using "
                    "the decoupled mode and the execute function must return "
                    "None.", "Exception message didn't match.")

    def test_decoupled_send_after_close_error(self):
        model_name = "decoupled_send_after_close_error"
        shape = [16]
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            client.start_stream(callback=partial(callback, user_data))
            input_data_0 = np.random.random(shape).astype(np.float32)
            input_data_1 = np.random.random(shape).astype(np.float32)
            inputs = [
                grpcclient.InferInput("INPUT0", input_data_0.shape,
                                      np_to_triton_dtype(input_data_0.dtype)),
                grpcclient.InferInput("INPUT1", input_data_1.shape,
                                      np_to_triton_dtype(input_data_1.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            inputs[1].set_data_from_numpy(input_data_1)
            client.async_stream_infer(model_name=model_name, inputs=inputs)

            # Because the model has closed the response sender there is no
            # way to deliver the error message to the client. The error
            # will be logged on the server side.
            time.sleep(4)
            self.assertEqual(user_data._completed_requests.qsize(), 0,
                             "The completed request size must be zero.")


if __name__ == '__main__':
    unittest.main()
