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

import numpy as np
import unittest
import os
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils


class SharedMemoryTest(tu.TestResultCollector):

    def test_invalid_create_shm(self):
        # Raises error since tried to create invalid system shared memory region
        try:
            shm_op0_handle = shm.create_shared_memory_region(
                "dummy_data", "/dummy_data", -1)
            shm.destroy_shared_memory_region(shm_op0_handle)
        except Exception as ex:
            self.assertTrue(str(ex) == "unable to initialize the size")

    def test_valid_create_set_register(self):
        # Create a valid system shared memory region, fill data in it and register
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", "/dummy_data", 8)
        shm.set_shared_memory_region(shm_op0_handle,
                                     [np.array([1, 2], dtype=np.float32)])
        triton_client.register_system_shared_memory("dummy_data", "/dummy_data",
                                                    8)
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 1)
        else:
            self.assertTrue(len(shm_status.regions) == 1)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register(self):
        # Create a valid system shared memory region and unregister before register
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", "/dummy_data", 8)
        triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 0)
        else:
            self.assertTrue(len(shm_status.regions) == 0)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register(self):
        # Create a valid system shared memory region and unregister after register
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", "/dummy_data", 8)
        triton_client.register_system_shared_memory("dummy_data", "/dummy_data",
                                                    8)
        triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 0)
        else:
            self.assertTrue(len(shm_status.regions) == 0)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register(self):
        # Create a valid system shared memory region and unregister after register
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", "/dummy_data", 8)
        triton_client.register_system_shared_memory("dummy_data", "/dummy_data",
                                                    8)
        try:
            triton_client.register_system_shared_memory("dummy_data",
                                                        "/dummy_data", 8)
        except Exception as ex:
            self.assertTrue(
                "shared memory region 'dummy_data' already in manager" in str(
                    ex))
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 1)
        else:
            self.assertTrue(len(shm_status.regions) == 1)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def _configure_sever(self):
        shm_ip0_handle = shm.create_shared_memory_region(
            "input0_data", "/input0_data", 64)
        shm_ip1_handle = shm.create_shared_memory_region(
            "input1_data", "/input1_data", 64)
        shm_op0_handle = shm.create_shared_memory_region(
            "output0_data", "/output0_data", 64)
        shm_op1_handle = shm.create_shared_memory_region(
            "output1_data", "/output1_data", 64)
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        shm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        shm.set_shared_memory_region(shm_ip1_handle, [input1_data])
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        triton_client.register_system_shared_memory("input0_data",
                                                    "/input0_data", 64)
        triton_client.register_system_shared_memory("input1_data",
                                                    "/input1_data", 64)
        triton_client.register_system_shared_memory("output0_data",
                                                    "/output0_data", 64)
        triton_client.register_system_shared_memory("output1_data",
                                                    "/output1_data", 64)
        return [shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle]

    def _cleanup_server(self, shm_handles):
        for shm_handle in shm_handles:
            shm.destroy_shared_memory_region(shm_handle)

    def _basic_inference(self,
                         shm_ip0_handle,
                         shm_ip1_handle,
                         shm_op0_handle,
                         shm_op1_handle,
                         error_msg,
                         big_shm_name="",
                         big_shm_size=64):
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        inputs = []
        outputs = []
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
            inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
            inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))
            outputs.append(
                httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
            outputs.append(
                httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
            inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))
            inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))
            outputs.append(grpcclient.InferRequestedOutput('OUTPUT0'))
            outputs.append(grpcclient.InferRequestedOutput('OUTPUT1'))

        inputs[0].set_shared_memory("input0_data", 64)

        if type(shm_ip1_handle) == np.array:
            inputs[1].set_data_from_numpy(input0_data, binary_data=True)
        elif big_shm_name != "":
            inputs[1].set_shared_memory(big_shm_name, big_shm_size)
        else:
            inputs[1].set_shared_memory("input1_data", 64)

        outputs[0].set_shared_memory("output0_data", 64)
        outputs[1].set_shared_memory("output1_data", 64)

        try:
            results = triton_client.infer("simple",
                                          inputs,
                                          model_version="",
                                          outputs=outputs)
            output = results.get_output('OUTPUT0')
            if _protocol == "http":
                output_datatype = output['datatype']
                output_shape = output['shape']
            else:
                output_datatype = output.datatype
                output_shape = output.shape
            output_dtype = utils.triton_to_np_dtype(output_datatype)
            output_data = shm.get_contents_as_numpy(shm_op0_handle,
                                                    output_dtype, output_shape)
            self.assertTrue(
                (output_data[0] == (input0_data + input1_data)).all(),
                "Model output does not match expected output")
        except Exception as ex:
            error_msg.append(str(ex))

    def test_unregister_after_inference(self):
        # Unregister after inference
        error_msg = []
        shm_handles = self._configure_sever()
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2],
                              shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        triton_client.unregister_system_shared_memory("output0_data")
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 3)
        else:
            self.assertTrue(len(shm_status.regions) == 3)
        self._cleanup_server(shm_handles)

    def test_register_after_inference(self):
        # Register after inference
        error_msg = []
        shm_handles = self._configure_sever()
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2],
                              shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = shm.create_shared_memory_region(
            "input2_data", "/input2_data", 64)
        triton_client.register_system_shared_memory("input2_data",
                                                    "/input2_data", 64)
        shm_status = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(shm_status) == 5)
        else:
            self.assertTrue(len(shm_status.regions) == 5)
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_too_big_shm(self):
        # Shared memory input region larger than needed - Throws error
        error_msg = []
        shm_handles = self._configure_sever()
        shm_ip2_handle = shm.create_shared_memory_region(
            "input2_data", "/input2_data", 128)
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        triton_client.register_system_shared_memory("input2_data",
                                                    "/input2_data", 128)
        self._basic_inference(shm_handles[0], shm_ip2_handle, shm_handles[2],
                              shm_handles[3], error_msg, "input2_data", 128)
        if len(error_msg) > 0:
            self.assertTrue(
                "unexpected total byte size 128 for input 'INPUT1', expecting 64"
                in error_msg[-1])
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_mixed_raw_shm(self):
        # Mix of shared memory and RAW inputs
        error_msg = []
        shm_handles = self._configure_sever()
        input1_data = np.ones(shape=16, dtype=np.int32)
        self._basic_inference(shm_handles[0], [input1_data], shm_handles[2],
                              shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_server(shm_handles)

    def test_unregisterall(self):
        # Unregister all shared memory blocks
        shm_handles = self._configure_sever()
        if _protocol == "http":
            triton_client = httpclient.InferenceServerClient(_url, verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(_url, verbose=True)
        status_before = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(status_before) == 4)
        else:
            self.assertTrue(len(status_before.regions) == 4)
        triton_client.unregister_system_shared_memory()
        status_after = triton_client.get_system_shared_memory_status()
        if _protocol == "http":
            self.assertTrue(len(status_after) == 0)
        else:
            self.assertTrue(len(status_after.regions) == 0)
        self._cleanup_server(shm_handles)


if __name__ == '__main__':
    _protocol = os.environ.get('CLIENT_TYPE', "http")
    if _protocol == "http":
        _url = "localhost:8000"
    else:
        _url = "localhost:8001"
    unittest.main()
