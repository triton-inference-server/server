#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import tensorrtserver.cuda_shared_memory as cshm
from tensorrtserver.api import *
import numpy as np
import unittest
import threading

class CudaSharedMemoryTest(unittest.TestCase):
    def test_invalid_create_shm():
        # Raises error since tried to create invalid cuda shared memory region
        try:
            shm_op0_handle = cshm.create_shared_memory_region("dummy_data", -1, 0)
        except Exception as ex:
            self.assertTrue(str(ex) == "unable to create cuda shared memory handle")
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_valid_create_set_register():
        # Create a valid cuda shared memory region, fill data in it and register
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        cshm.set_shared_memory_region(shm_op0_handle, [np.array([1,2], dtype=np.float32)])
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register():
        # Create a valid cuda shared memory region and unregister before register
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.unregister(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register():
        # Create a valid cuda shared memory region and unregister after register
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shared_memory_ctx.unregister(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register():
        # Create a valid cuda shared memory region and unregister after register
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        try:
            shared_memory_ctx.cuda_register(shm_op0_handle)
        except Exception as ex:
            self.assertTrue("shared memory block 'dummy_data' already in manager" in str(ex))
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def _configure_sever():
        shm_op0_handle = cshm.create_shared_memory_region("output0_data", 64, 0)
        shm_op1_handle = cshm.create_shared_memory_region("output1_data", 64, 0)
        shm_ip0_handle = cshm.create_shared_memory_region("input0_data", 64, 0)
        shm_ip1_handle = cshm.create_shared_memory_region("input1_data", 64, 0)
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        cshm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        cshm.set_shared_memory_region(shm_ip1_handle, [input1_data])
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shared_memory_ctx.cuda_register(shm_ip0_handle)
        shared_memory_ctx.cuda_register(shm_ip1_handle)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shared_memory_ctx.cuda_register(shm_op1_handle)
        return [shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle]

    def _cleanup_server(shm_handles):
        for shm_handle in shm_handles:
            cshm.destroy_shared_memory_region(shm_handle)

    def _basic_inference(shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle, error_msg):
        infer_ctx = InferContext("localhost:8000", ProtocolType.HTTP, "simple", -1, verbose=False)
        try:
            results = infer_ctx.run({ 'INPUT0' : shm_ip0_handle, 'INPUT1' : shm_ip1_handle, },
                    { 'OUTPUT0' : (InferContext.ResultFormat.RAW, shm_op0_handle),
                    'OUTPUT1' : (InferContext.ResultFormat.RAW, shm_op1_handle)}, 1)
            self.assertTrue((results['OUTPUT0'][0] == (input0_data + input1_data)).all())
        except Exception as ex:
            error_msg.append(str(ex.message()))

    def test_unregister_during_inference():
        # Unregister during inference - inference fails and unregisters
        error_msg = []
        threads = []
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        threads.append(threading.Thread(target=self._basic_inference,
            args=(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)))
        threads.append(threading.Thread(target=shared_memory_ctx.unregister, args=(shm_handles[2],)))
        threads[0].start()
        threads[1].start()
        threads[0].join()
        threads[1].join()
        print(error_msg)
        self.assertTrue(error_msg[0] == "shared memory block 'output0_data' not found in manager")
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 3)
        self._cleanup_server(shm_handles)

    def test_register_during_inference():
        # Register during inference - Registered successfully
        error_msg = []
        threads = []
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 64, 0)
        threads.append(threading.Thread(target=self._basic_inference,
            args=(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)))
        threads.append(threading.Thread(target=shared_memory_ctx.cuda_register, args=(shm_ip2_handle,)))
        threads[0].start()
        threads[1].start()
        threads[0].join()
        threads[1].join()

        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 5)
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_too_big_shm():
        # Shared memory input region larger than needed - Throws error
        error_msg = []
        threads = []
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 128, 0)
        shared_memory_ctx.cuda_register(shm_ip2_handle)
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)
        if len(error_msg) > 0:
            self.assertTrue(error_msg[-1] == "The input 'INPUT1' has shared memory of size 128 bytes"\
                                    " while the expected size is 1 * 64 = 64 bytes")
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_mixed_raw_shm():
        error_msg = []
        threads = []
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        self._basic_inference(shm_ip0_handle, [input1_data], shm_op0_handle, shm_op1_handle, error_msg)
        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_server(shm_handles)

    def test_unregisterall():
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
        status_before = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(status_before.shared_memory_region) == 4)
        shared_memory_ctx.unregister_all()
        status_after = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(status_after.shared_memory_region) == 0)
        self._cleanup_server(shm_handles)
