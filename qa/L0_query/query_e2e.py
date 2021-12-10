#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append('../common')

import argparse
import numpy as np
import os
from builtins import range
import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import cuda_shared_memory as cudashm
import unittest
import test_util as tu


class QueryTest(tu.TestResultCollector):

    def test_http(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        try:
            triton_client.infer(model_name="query", inputs=inputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

    def test_http_shared_memory(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        # Set up CUDA shared memory for outputs
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region(
            "output0_data", 4, 0)
        shm_op1_handle = cudashm.create_shared_memory_region(
            "output1_data", 4, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 4)
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 4)
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs[-1].set_shared_memory("output0_data", 4)

        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT1', binary_data=True))
        outputs[-1].set_shared_memory("output1_data", 4)

        try:
            triton_client.infer(model_name="query",
                                inputs=inputs,
                                outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 GPU 0" in ex.message())
            self.assertTrue("OUTPUT1 GPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_http_out_of_shared_memory(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        # Set up too small CUDA shared memory for outputs, expect query
        # returns default value
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region(
            "output0_data", 1, 0)
        shm_op1_handle = cudashm.create_shared_memory_region(
            "output1_data", 1, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 1)
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 1)
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs[-1].set_shared_memory("output0_data", 1)

        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT1', binary_data=True))
        outputs[-1].set_shared_memory("output1_data", 1)

        try:
            triton_client.infer(model_name="query",
                                inputs=inputs,
                                outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_grpc(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        try:
            triton_client.infer(model_name="query", inputs=inputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

    def test_grpc_shared_memory(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        # Set up CUDA shared memory for outputs
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region(
            "output0_data", 4, 0)
        shm_op1_handle = cudashm.create_shared_memory_region(
            "output1_data", 4, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 4)
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 4)
        outputs = []
        outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
        outputs[-1].set_shared_memory("output0_data", 4)

        outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))
        outputs[-1].set_shared_memory("output1_data", 4)

        try:
            triton_client.infer(model_name="query",
                                inputs=inputs,
                                outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 GPU 0" in ex.message())
            self.assertTrue("OUTPUT1 GPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_grpc_out_of_shared_memory(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput('INPUT', [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        # Set up too small CUDA shared memory for outputs, expect query
        # returns default value
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region(
            "output0_data", 1, 0)
        shm_op1_handle = cudashm.create_shared_memory_region(
            "output1_data", 1, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 1)
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 1)
        outputs = []
        outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
        outputs[-1].set_shared_memory("output0_data", 1)

        outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))
        outputs[-1].set_shared_memory("output1_data", 1)

        try:
            triton_client.infer(model_name="query",
                                inputs=inputs,
                                outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()


if __name__ == '__main__':
    unittest.main()
