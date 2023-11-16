#!/usr/bin/env python3

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

import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class ClientNonInferTimeoutTest(tu.TestResultCollector):
    def setUp(self):
        self.model_name_ = "custom_identity_int32"
        self.input0_data_ = np.array([[10]], dtype=np.int32)
        self.input0_data_byte_size_ = 32
        self.SMALL_INTERVAL = 0.1  # seconds for a timeout
        self.NORMAL_INTERVAL = 5.0  # seconds for server to load then receive request

    def test_grpc_server_live(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.is_server_live(client_timeout=self.SMALL_INTERVAL)
        self.assertIn("Deadline Exceeded", str(cm.exception))
        self.assertTrue(
            triton_client.is_server_live(client_timeout=self.NORMAL_INTERVAL)
        )

    def test_grpc_is_server_ready(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.is_server_ready(client_timeout=self.SMALL_INTERVAL)
        self.assertIn("Deadline Exceeded", str(cm.exception))
        self.assertTrue(
            triton_client.is_server_ready(client_timeout=self.NORMAL_INTERVAL)
        )

    def test_grpc_is_model_ready(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.is_model_ready(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        self.assertTrue(
            triton_client.is_model_ready(
                model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
            )
        )

    def test_grpc_get_server_metadata(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_server_metadata(client_timeout=self.SMALL_INTERVAL)
        self.assertIn("Deadline Exceeded", str(cm.exception))

        triton_client.get_server_metadata(client_timeout=self.NORMAL_INTERVAL)

    def test_grpc_get_model_metadata(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_model_metadata(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_model_metadata(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_get_model_config(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_model_config(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_model_config(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_model_repository_index(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_model_repository_index(
                client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_model_repository_index(client_timeout=self.NORMAL_INTERVAL)

    def test_grpc_load_model(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        triton_client.unload_model(model_name=self.model_name_)
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.load_model(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unload_model(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )
        triton_client.load_model(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_unload_model(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unload_model(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.load_model(model_name=self.model_name_)
        triton_client.unload_model(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )
        triton_client.load_model(model_name=self.model_name_)

    def test_grpc_get_inference_statistics(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_inference_statistics(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_inference_statistics(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_update_trace_settings(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.update_trace_settings(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.update_trace_settings(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_get_trace_settings(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_trace_settings(
                model_name=self.model_name_, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_trace_settings(
            model_name=self.model_name_, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_update_log_settings(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        settings = {}
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.update_log_settings(
                settings=settings, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.update_log_settings(
            settings=settings, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_get_log_settings(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_log_settings(
                as_json=True, client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_log_settings(
            as_json=True, client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_get_system_shared_memory_status(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_system_shared_memory_status(
                client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_system_shared_memory_status(
            client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_register_system_shared_memory(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        triton_client.unregister_system_shared_memory()
        import tritonclient.utils.shared_memory as shm

        shm_ip0_handle = shm.create_shared_memory_region(
            "input0_data", "/input_simple", self.input0_data_byte_size_
        )
        shm.set_shared_memory_region(shm_ip0_handle, [self.input0_data_])
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.register_system_shared_memory(
                "input0_data",
                "/input_simple",
                self.input0_data_byte_size_,
                client_timeout=self.SMALL_INTERVAL,
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unregister_system_shared_memory()
        triton_client.register_system_shared_memory(
            "input0_data",
            "/input_simple",
            self.input0_data_byte_size_,
            client_timeout=self.NORMAL_INTERVAL,
        )
        triton_client.unregister_system_shared_memory()

    def test_grpc_unregister_system_shared_memory(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unregister_system_shared_memory(
                client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unregister_system_shared_memory(
            client_timeout=self.NORMAL_INTERVAL
        )

    def test_grpc_get_cuda_shared_memory_status(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.get_cuda_shared_memory_status(
                client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.get_cuda_shared_memory_status(client_timeout=self.NORMAL_INTERVAL)

    def test_grpc_register_cuda_shared_memory(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        import tritonclient.utils.cuda_shared_memory as cshm

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
                client_timeout=self.SMALL_INTERVAL,
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unregister_cuda_shared_memory()
        triton_client.register_cuda_shared_memory(
            "dummy_data",
            cshm.get_raw_handle(shm_op0_handle),
            device_id=0,
            byte_size=byteSize,
            client_timeout=self.NORMAL_INTERVAL,
        )
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_grpc_unregister_cuda_shared_memory(self):
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        with self.assertRaises(InferenceServerException) as cm:
            _ = triton_client.unregister_cuda_shared_memory(
                client_timeout=self.SMALL_INTERVAL
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))
        triton_client.unregister_cuda_shared_memory(client_timeout=self.NORMAL_INTERVAL)


if __name__ == "__main__":
    unittest.main()
