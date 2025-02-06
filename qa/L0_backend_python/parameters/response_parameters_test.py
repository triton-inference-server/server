# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import unittest

import numpy as np
import shm_util
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class ResponseParametersTest(unittest.TestCase):
    _server_address_grpc = "localhost:8001"
    _model_name = "response_parameters"
    _shape = [1, 1]

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _assert_response_parameters_match(self, infer_result, expected_params):
        res_params = {}
        for param_key, param_value in infer_result.get_response().parameters.items():
            if param_value.HasField("bool_param"):
                value = param_value.bool_param
            elif param_value.HasField("int64_param"):
                value = param_value.int64_param
            elif param_value.HasField("string_param"):
                value = param_value.string_param
            else:
                raise ValueError(f"Unsupported parameter choice: {param_value}")
            res_params[param_key] = value
        self.assertEqual(expected_params, res_params)

    def _assert_response_parameters_infer_success(self, params):
        params_str = json.dumps(params)

        inputs = [grpcclient.InferInput("RESPONSE_PARAMETERS", self._shape, "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[params_str]], dtype=np.object_))

        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(self._server_address_grpc) as client:
                result = client.infer(self._model_name, inputs)

        # verify the response parameters
        self._assert_response_parameters_match(result, params)

        # model returns the input as output
        output = str(result.as_numpy("OUTPUT")[0][0], encoding="utf-8")
        self.assertEqual(params_str, output)

    def _assert_response_parameters_infer_fail(self, params, expected_err_msg):
        params_str = json.dumps(params)

        inputs = [grpcclient.InferInput("RESPONSE_PARAMETERS", self._shape, "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[params_str]], dtype=np.object_))

        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(self._server_address_grpc) as client:
                with self.assertRaises(InferenceServerException) as e:
                    client.infer(self._model_name, inputs)

        self.assertIn("[StatusCode.INVALID_ARGUMENT] ", str(e.exception))
        self.assertIn(expected_err_msg, str(e.exception))

    def test_setting_empty_response_parameters(self):
        params = {}
        self._assert_response_parameters_infer_success(params)

    def test_setting_one_element_response_parameters(self):
        params = {"many_elements": False}
        self._assert_response_parameters_infer_success(params)

    def test_setting_three_element_response_parameters(self):
        params = {"bool": True, "str": "Hello World!", "int": 1024}
        self._assert_response_parameters_infer_success(params)

    def test_setting_multi_element_response_parameters(self):
        params = {"a": "1", "b": "2", "c": 3, "d": False, "e": 5, "f": ""}
        self._assert_response_parameters_infer_success(params)

    def test_setting_wrong_type_response_parameters(self):
        params = []
        expected_err_msg = ", got <class 'list'>"
        self._assert_response_parameters_infer_fail(params, expected_err_msg)

    def test_setting_int_key_type_response_parameters(self):
        params = {"1": "int key"}
        expected_err_msg = (
            "Expect parameters keys to have type str, found type <class 'int'>"
        )
        self._assert_response_parameters_infer_fail(params, expected_err_msg)

    def test_setting_float_response_parameters(self):
        params = {"int": 2, "float": 0.5}
        expected_err_msg = "Expect parameters values to have type bool/int/str, found type <class 'float'>"
        self._assert_response_parameters_infer_fail(params, expected_err_msg)

    def test_setting_null_response_parameters(self):
        params = {"bool": True, "null": None}
        expected_err_msg = "Expect parameters values to have type bool/int/str, found type <class 'NoneType'>"
        self._assert_response_parameters_infer_fail(params, expected_err_msg)

    def test_setting_nested_response_parameters(self):
        params = {"str": "", "list": ["variable"]}
        expected_err_msg = "Expect parameters values to have type bool/int/str, found type <class 'list'>"
        self._assert_response_parameters_infer_fail(params, expected_err_msg)

    def test_setting_response_parameters_decoupled(self):
        model_name = "response_parameters_decoupled"
        params = [{"bool": False, "int": 2048}, {"str": "Hello World!"}]
        params_str = json.dumps(params)

        inputs = [grpcclient.InferInput("RESPONSE_PARAMETERS", self._shape, "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[params_str]], dtype=np.object_))

        responses = []
        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(self._server_address_grpc) as client:
                client.start_stream(
                    callback=(lambda result, error: responses.append((result, error)))
                )
                client.async_stream_infer(model_name=model_name, inputs=inputs)
                client.stop_stream()

        self.assertEqual(len(params), len(responses))
        for i in range(len(params)):
            result, error = responses[i]
            self.assertIsNone(error)

            # Since this is a decoupled model, the 'triton_final_response' parameter
            # will be a part of the response parameters, so include it into the expected
            # parameters. The model sends the complete final flag separately from the
            # response, so the parameter is always False.
            expected_params = params[i].copy()
            expected_params["triton_final_response"] = False
            self._assert_response_parameters_match(result, expected_params)

            output = str(result.as_numpy("OUTPUT")[0][0], encoding="utf-8")
            self.assertEqual(json.dumps(params[i]), output)

    def test_setting_response_parameters_bls(self):
        model_name = "response_parameters_bls"
        params = {"bool": False, "int": 2048, "str": "Hello World!"}
        params_decoupled = [{}, {"bool": True, "int": 10000}, {"str": "?"}]
        params_str = json.dumps(params)
        params_decoupled_str = json.dumps(params_decoupled)

        inputs = [
            grpcclient.InferInput("RESPONSE_PARAMETERS", self._shape, "BYTES"),
            grpcclient.InferInput(
                "RESPONSE_PARAMETERS_DECOUPLED", self._shape, "BYTES"
            ),
        ]
        inputs[0].set_data_from_numpy(np.array([[params_str]], dtype=np.object_))
        inputs[1].set_data_from_numpy(
            np.array([[params_decoupled_str]], dtype=np.object_)
        )

        with self._shm_leak_detector.Probe() as shm_probe:
            with grpcclient.InferenceServerClient(self._server_address_grpc) as client:
                result = client.infer(model_name, inputs)

        output = str(result.as_numpy("OUTPUT")[0][0], encoding="utf-8")
        self.assertEqual(output, "True")


if __name__ == "__main__":
    unittest.main()
