#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
import socket
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class ClientTimeoutTest(tu.TestResultCollector):
    def setUp(self):
        self.model_name_ = "custom_identity_int32"
        self.input0_data_ = np.array([[10]], dtype=np.int32)

    def test_server_live(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.is_server_ready(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_model_ready(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.is_model_ready(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_server_metadata(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_server_metadata(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_model_config(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_model_config(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_model_repository_index(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_model_repository_index(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_load_model(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.load_model(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_unload_model(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unload_model(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_inference_statistics(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_inference_statistics(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_update_trace_settings(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.update_trace_settings(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_get_trace_settings(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_trace_settings(
                model_name=self.model_name_, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_update_log_settings(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.update_log_settings(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_get_log_settings(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_log_settings(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_get_system_shared_memory(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_system_shared_memory_status(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_register_system_shared_memory(self):
        input_data_serialized = utils.serialize_byte_tensor(self.input0_data_)
        input_byte_size = utils.serialized_byte_size(input_data_serialized)
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.register_system_shared_memory(
                "input_data", "/input_simple", input_byte_size, client_timeout=0.2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_unregister_system_shared_memory(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unregister_system_shared_memory(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_get_cuda_shared_memory_status(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_cuda_shared_memory_status(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_register_cuda_shared_memory(self):
        import tritonshmutils.cuda_shared_memory as cshm

        input_data = np.array([[10]], dtype=np.int32)
        byteSize = input_data.itemsize * input_data.size
        shm_op0_handle = cshm.create_shared_memory_region(
            "dummy_data", byte_size=byteSize, device_id=0
        )
        cshm.set_shared_memory_region(shm_op0_handle, [input_data])
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.register_cuda_shared_memory(
                "dummy_data",
                cshm.get_raw_handle(shm_op0_handle),
                device_id=0,
                byte_size=byteSize,
                client_timeout=0.2,
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unregister_cuda_shared_memory()
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_uregister_cuda_shared_memory(self):
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unregister_cuda_shared_memory(client_timeout=0.2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def _prepare_request(self, protocol):
        if protocol == "grpc":
            self.inputs_ = []
            self.inputs_.append(grpcclient.InferInput("INPUT0", [1, 1], "INT32"))
            self.outputs_ = []
            self.outputs_.append(grpcclient.InferRequestedOutput("OUTPUT0"))
        else:
            self.inputs_ = []
            self.inputs_.append(httpclient.InferInput("INPUT0", [1, 1], "INT32"))
            self.outputs_ = []
            self.outputs_.append(httpclient.InferRequestedOutput("OUTPUT0"))

        self.inputs_[0].set_data_from_numpy(self.input0_data_)

    def test_grpc_infer(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        self._prepare_request("grpc")

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.infer(
                model_name=self.model_name_,
                inputs=self.inputs_,
                outputs=self.outputs_,
                client_timeout=0.2,
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

        # Expect inference to pass successfully for a large timeout
        # value
        result = triton_client.infer(
            model_name=self.model_name_,
            inputs=self.inputs_,
            outputs=self.outputs_,
            client_timeout=10,
        )

        output0_data = result.as_numpy("OUTPUT0")
        self.assertTrue(np.array_equal(self.input0_data_, output0_data))

    def test_grpc_async_infer(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        self._prepare_request("grpc")

        user_data = UserData()

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        with self.assertRaises(InferenceServerException) as cm:
            triton_client.async_infer(
                model_name=self.model_name_,
                inputs=self.inputs_,
                callback=partial(callback, user_data),
                outputs=self.outputs_,
                client_timeout=2,
            )
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item
        self.assertIn("Deadline Exceeded", str(cm.exception))

        # Expect inference to pass successfully for a large timeout
        # value
        triton_client.async_infer(
            model_name=self.model_name_,
            inputs=self.inputs_,
            callback=partial(callback, user_data),
            outputs=self.outputs_,
            client_timeout=10,
        )

        # Wait until the results are available in user_data
        data_item = user_data._completed_requests.get()
        self.assertFalse(type(data_item) == InferenceServerException)

        output0_data = data_item.as_numpy("OUTPUT0")
        self.assertTrue(np.array_equal(self.input0_data_, output0_data))

    def test_grpc_stream_infer(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )

        self._prepare_request("grpc")
        user_data = UserData()

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        with self.assertRaises(InferenceServerException) as cm:
            triton_client.stop_stream()
            triton_client.start_stream(
                callback=partial(callback, user_data), stream_timeout=1
            )
            triton_client.async_stream_infer(
                model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
            )
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item
        self.assertIn("Deadline Exceeded", str(cm.exception))

        # Expect inference to pass successfully for a large timeout
        # value
        triton_client.stop_stream()
        triton_client.start_stream(
            callback=partial(callback, user_data), stream_timeout=100
        )

        triton_client.async_stream_infer(
            model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
        )
        data_item = user_data._completed_requests.get()
        triton_client.stop_stream()

        if type(data_item) == InferenceServerException:
            raise data_item
        output0_data = data_item.as_numpy("OUTPUT0")
        self.assertTrue(np.array_equal(self.input0_data_, output0_data))

    def test_http_infer(self):
        self._prepare_request("http")

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        with self.assertRaises(socket.timeout) as cm:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True, network_timeout=2.0
            )
            _ = triton_client.infer(
                model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
            )
        self.assertIn("timed out", str(cm.exception))

        # Expect to successfully pass with sufficiently large timeout
        triton_client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=True, connection_timeout=10.0
        )

        result = triton_client.infer(
            model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
        )

        output0_data = result.as_numpy("OUTPUT0")
        self.assertTrue(np.array_equal(self.input0_data_, output0_data))

    def test_http_async_infer(self):
        self._prepare_request("http")

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        with self.assertRaises(socket.timeout) as cm:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True, network_timeout=2.0
            )
            async_request = triton_client.async_infer(
                model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
            )
            result = async_request.get_result()
        self.assertIn("timed out", str(cm.exception))

        # Expect to successfully pass with sufficiently large timeout
        triton_client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=True, connection_timeout=10.0
        )

        async_request = triton_client.async_infer(
            model_name=self.model_name_, inputs=self.inputs_, outputs=self.outputs_
        )
        result = async_request.get_result()

        output0_data = result.as_numpy("OUTPUT0")
        self.assertTrue(np.array_equal(self.input0_data_, output0_data))


if __name__ == "__main__":
    unittest.main()
