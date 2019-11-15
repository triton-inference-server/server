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

class CudaSharedMemoryTest(unittest.TestCase):
    def test_invalid_create_shm(self):
        # Raises error since tried to create invalid cuda shared memory region
        try:
            shm_op0_handle = cshm.create_shared_memory_region("dummy_data", -1, 0)
            cshm.destroy_shared_memory_region(shm_op0_handle)
        except Exception as ex:
            self.assertTrue(str(ex) == "unable to create cuda shared memory handle")

    def test_valid_create_set_register(self):
        # Create a valid cuda shared memory region, fill data in it and register
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        cshm.set_shared_memory_region(shm_op0_handle, [np.array([1,2], dtype=np.float32)])
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register(self):
        # Create a valid cuda shared memory region and unregister before register
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.unregister(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register(self):
        # Create a valid cuda shared memory region and unregister after register
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shared_memory_ctx.unregister(shm_op0_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register(self):
        # Create a valid cuda shared memory region and unregister after register
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        try:
            shared_memory_ctx.cuda_register(shm_op0_handle)
        except Exception as ex:
            self.assertTrue("shared memory block 'dummy_data' already in manager" in str(ex))
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def _configure_sever(self):
        shm_op0_handle = cshm.create_shared_memory_region("output0_data", 64, 0)
        shm_op1_handle = cshm.create_shared_memory_region("output1_data", 64, 0)
        shm_ip0_handle = cshm.create_shared_memory_region("input0_data", 64, 0)
        shm_ip1_handle = cshm.create_shared_memory_region("input1_data", 64, 0)
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        cshm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        cshm.set_shared_memory_region(shm_ip1_handle, [input1_data])
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shared_memory_ctx.cuda_register(shm_ip0_handle)
        shared_memory_ctx.cuda_register(shm_ip1_handle)
        shared_memory_ctx.cuda_register(shm_op0_handle)
        shared_memory_ctx.cuda_register(shm_op1_handle)
        return [shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle]

    def _cleanup_server(self, shm_handles):
        for shm_handle in shm_handles:
            cshm.destroy_shared_memory_region(shm_handle)

    def _basic_inference(self, shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle, error_msg):
        infer_ctx = InferContext(_url, _protocol, "simple", -1, verbose=True)
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        try:
            results = infer_ctx.run({ 'INPUT0' : shm_ip0_handle, 'INPUT1' : shm_ip1_handle, },
                    { 'OUTPUT0' : (InferContext.ResultFormat.RAW, shm_op0_handle),
                    'OUTPUT1' : (InferContext.ResultFormat.RAW, shm_op1_handle)}, 1)
            self.assertTrue((results['OUTPUT0'][0] == (input0_data + input1_data)).all())
        except Exception as ex:
            error_msg.append(str(ex))

    def test_unregister_after_inference(self):
        # Unregister after inference
        error_msg = []
        shm_handles = self._configure_sever()
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shared_memory_ctx.unregister(shm_handles[2])
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 3)
        self._cleanup_server(shm_handles)

    def test_register_after_inference(self):
        # Register after inference
        error_msg = []
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 64, 0)
        shared_memory_ctx.cuda_register(shm_ip2_handle)
        shm_status = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(shm_status.shared_memory_region) == 5)
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_too_big_shm(self):
        # Shared memory input region larger than needed - Throws error
        error_msg = []
        shm_handles = self._configure_sever()
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 128, 0)
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        shared_memory_ctx.cuda_register(shm_ip2_handle)
        self._basic_inference(shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg)
        if len(error_msg) > 0:
            self.assertTrue(error_msg[-1] == "The input 'INPUT1' has shared memory of size 128 bytes"\
                                    " while the expected size is 1 * 64 = 64 bytes")
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_mixed_raw_shm(self):
        # Mix of shared memory and RAW inputs
        error_msg = []
        shm_handles = self._configure_sever()
        input1_data = np.ones(shape=16, dtype=np.int32)
        self._basic_inference(shm_handles[0], [input1_data], shm_handles[2], shm_handles[3], error_msg)
        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_server(shm_handles)

    def test_unregisterall(self):
        # Unregister all shared memory blocks
        shm_handles = self._configure_sever()
        shared_memory_ctx = SharedMemoryControlContext(_url,  _protocol, verbose=True)
        status_before = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(status_before.shared_memory_region) == 4)
        shared_memory_ctx.unregister_all()
        status_after = shared_memory_ctx.get_shared_memory_status()
        self.assertTrue(len(status_after.shared_memory_region) == 0)
        self._cleanup_server(shm_handles)

if __name__ == '__main__':
    if os.environ.get('CLIENT_TYPE', "") == "http":
        _protocol = ProtocolType.HTTP
        _url = "localhost:8000"
    else:
        _protocol = ProtocolType.GRPC
        _url = "localhost:8001"
    unittest.main()