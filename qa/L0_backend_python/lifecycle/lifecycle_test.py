# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shm_util
from functools import partial
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import numpy as np
import unittest
import queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class LifecycleTest(tu.TestResultCollector):

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def test_batch_error(self):
        # The execute_error model returns an error for the first request and
        # sucessfully processes the second request. This is making sure that
        # an error in a single request does not completely fail the batch.
        model_name = "execute_error"
        shape = [2, 2]
        number_of_requests = 2
        user_data = UserData()
        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        triton_client.start_stream(callback=partial(callback, user_data))

        with self._shm_leak_detector.Probe() as shm_probe:
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

    def test_infer_pymodel_error(self):
        model_name = "wrong_model"
        shape = [2, 2]

        with self._shm_leak_detector.Probe() as shm_probe:
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
                            "Failed to process the request(s) for model instance"
                        ), "Exception message is not correct")
                else:
                    self.assertTrue(
                        False,
                        "Wrong exception raised or did not raise an exception")

    def test_incorrect_execute_return(self):
        model_name = 'execute_return_error'
        shape = [1, 1]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient("localhost:8000") as client:
                input_data = (5 * np.random.randn(*shape)).astype(np.float32)
                inputs = [
                    httpclient.InferInput("INPUT", input_data.shape,
                                          np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)

                # The first request to this model will return None.
                with self.assertRaises(InferenceServerException) as e:
                    client.infer(model_name, inputs)

                self.assertTrue(
                    str(e.exception).startswith(
                        "Failed to process the request(s) for model instance "
                        "'execute_return_error_0', message: Expected a list in the "
                        "execute return"), "Exception message is not correct.")

                # The second inference request will return a list of None object
                # instead of Python InferenceResponse objects.
                with self.assertRaises(InferenceServerException) as e:
                    client.infer(model_name, inputs)

                self.assertTrue(
                    str(e.exception).startswith(
                        "Failed to process the request(s) for model instance "
                        "'execute_return_error_0', message: Expected an "
                        "'InferenceResponse' object in the execute function return"
                        " list"), "Exception message is not correct.")


if __name__ == '__main__':
    unittest.main()
