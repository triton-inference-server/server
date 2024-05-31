#!/usr/bin/env python3

# Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

sys.path.append("../../common")

import queue
import time
import unittest
from functools import partial

import numpy as np
import shm_util
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import *

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def test_error_code(self):
        model_name = "error_code"
        shape = [1, 1]
        # [(Triton error, expected gRPC error message starting), ...]
        errors = [
            ("UNKNOWN", "[StatusCode.UNKNOWN]"),
            ("INTERNAL", "[StatusCode.INTERNAL]"),
            ("NOT_FOUND", "[StatusCode.NOT_FOUND]"),
            ("INVALID_ARG", "[StatusCode.INVALID_ARGUMENT]"),
            ("UNAVAILABLE", "[StatusCode.UNAVAILABLE]"),
            ("UNSUPPORTED", "[StatusCode.UNIMPLEMENTED]"),
            ("ALREADY_EXISTS", "[StatusCode.ALREADY_EXISTS]"),
            ("CANCELLED", "[StatusCode.CANCELLED]"),
            ("(default)", "[StatusCode.INTERNAL] unrecognized"),
        ]
        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8001"
            ) as client:
                for error, expected_grpc_error_start in errors:
                    input_data = np.array([[error]], dtype=np.object_)
                    inputs = [
                        grpcclient.InferInput(
                            "ERROR_CODE", shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    with self.assertRaises(InferenceServerException) as e:
                        client.infer(model_name, inputs)
                    # e.g. [StatusCode.UNKNOWN] error code: TRITONSERVER_ERROR_UNKNOWN
                    # e.g. [StatusCode.INTERNAL] unrecognized error code: (default)
                    self.assertEqual(
                        str(e.exception),
                        expected_grpc_error_start + " error code: " + error,
                    )

    def test_execute_cancel(self):
        model_name = "execute_cancel"
        log_path = "lifecycle_server.log"
        execute_delay = 4.0  # seconds
        shape = [1, 1]
        response = {"responded": False, "result": None, "error": None}

        def callback(result, error):
            response["responded"] = True
            response["result"] = result
            response["error"] = error

        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8001"
            ) as client:
                input_data = np.array([[execute_delay]], dtype=np.float32)
                inputs = [
                    grpcclient.InferInput(
                        "EXECUTE_DELAY", shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                exec_future = client.async_infer(model_name, inputs, callback)
                time.sleep(2)  # ensure the request is executing
                self.assertFalse(response["responded"])
                exec_future.cancel()
                time.sleep(2)  # ensure the cancellation is delivered
                self.assertTrue(response["responded"])

        self.assertEqual(response["result"], None)
        self.assertIsInstance(response["error"], InferenceServerException)
        self.assertEqual(response["error"].status(), "StatusCode.CANCELLED")
        with open(log_path, mode="r", encoding="utf-8", errors="strict") as f:
            log_text = f.read()
            self.assertIn("[execute_cancel] Request not cancelled at 1.0 s", log_text)
            self.assertIn("[execute_cancel] Request cancelled at ", log_text)

    def test_batch_error(self):
        # The execute_error model returns an error for the first and third
        # request and successfully processes the second request. This is making
        # sure that an error in a single request does not completely fail the
        # batch.
        model_name = "execute_error"
        shape = [2, 2]
        number_of_requests = 3
        user_data = UserData()
        triton_client = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")
        triton_client.start_stream(callback=partial(callback, user_data))

        with self._shm_leak_detector.Probe() as shm_probe:
            input_datas = []
            for i in range(number_of_requests):
                input_data = np.random.randn(*shape).astype(np.float32)
                input_datas.append(input_data)
                inputs = [
                    grpcclient.InferInput(
                        "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                triton_client.async_stream_infer(model_name=model_name, inputs=inputs)

            for i in range(number_of_requests):
                result = user_data._completed_requests.get()
                if i == 0 or i == 2:
                    self.assertIs(type(result), InferenceServerException)
                    continue

                print(result)
                output_data = result.as_numpy("OUT")
                self.assertIsNotNone(output_data, "error: expected 'OUT'")
                self.assertTrue(
                    np.array_equal(output_data, input_datas[i]),
                    "error: expected output {} to match input {}".format(
                        output_data, input_datas[i]
                    ),
                )

    def test_infer_pymodel_error(self):
        model_name = "wrong_model"
        shape = [2, 2]

        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data = (16384 * np.random.randn(*shape)).astype(np.uint32)
                inputs = [
                    httpclient.InferInput(
                        "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                try:
                    client.infer(model_name, inputs)
                except InferenceServerException as e:
                    print(e.message())
                    self.assertTrue(
                        e.message().startswith(
                            "Failed to process the request(s) for model "
                        ),
                        "Exception message is not correct",
                    )
                else:
                    self.assertTrue(
                        False, "Wrong exception raised or did not raise an exception"
                    )


if __name__ == "__main__":
    unittest.main()
