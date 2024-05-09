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

import sys

sys.path.append("../common")

import os
import unittest

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonshmutils.cuda_shared_memory as cshm
from tritonclient.utils import *


class CudaSharedMemoryTest(tu.TestResultCollector):
    DEFAULT_SHM_BYTE_SIZE = 64

    def setUp(self):
        self._setup_client()

    def _setup_client(self):
        self.protocol = os.environ.get("CLIENT_TYPE", "http")
        if self.protocol == "http":
            self.url = "localhost:8000"
            self.triton_client = httpclient.InferenceServerClient(
                self.url, verbose=True
            )
        else:
            self.url = "localhost:8001"
            self.triton_client = grpcclient.InferenceServerClient(
                self.url, verbose=True
            )

    def test_invalid_create_shm(self):
        # Raises error since tried to create invalid cuda shared memory region
        try:
            shm_op0_handle = cshm.create_shared_memory_region("dummy_data", -1, 0)
            cshm.destroy_shared_memory_region(shm_op0_handle)
        except Exception as ex:
            self.assertEqual(str(ex), "unable to create cuda shared memory handle")

    def test_valid_create_set_register(self):
        # Create a valid cuda shared memory region, fill data in it and register
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        cshm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register(self):
        # Create a valid cuda shared memory region and unregister before register
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.unregister_cuda_shared_memory("dummy_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register(self):
        # Create a valid cuda shared memory region and unregister after register
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        self.triton_client.unregister_cuda_shared_memory("dummy_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register(self):
        # Create a valid cuda shared memory region and unregister after register
        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        try:
            self.triton_client.register_cuda_shared_memory(
                "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
            )
        except Exception as ex:
            self.assertIn(
                "shared memory region 'dummy_data' already in manager", str(ex)
            )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def _configure_server(
        self,
        create_byte_size=DEFAULT_SHM_BYTE_SIZE,
        register_byte_size=DEFAULT_SHM_BYTE_SIZE,
        device_id=0,
    ):
        """Creates and registers cuda shared memory regions for testing.

        Parameters
        ----------
        create_byte_size: int
            Size of each cuda shared memory region to create.
            NOTE: This should be sufficiently large to hold the inputs/outputs
                  stored in shared memory.

        register_byte_size: int
            Size of each cuda shared memory region to register with server.
            NOTE: The register_byte_size should be less than or equal
            to the create_byte_size. Otherwise an exception will be raised for
            an invalid set of registration args.

        device_id: int
            The GPU device ID of the cuda shared memory region to be created.

        """

        shm_ip0_handle = cshm.create_shared_memory_region(
            "input0_data", create_byte_size, device_id
        )
        shm_ip1_handle = cshm.create_shared_memory_region(
            "input1_data", create_byte_size, device_id
        )
        shm_op0_handle = cshm.create_shared_memory_region(
            "output0_data", create_byte_size, device_id
        )
        shm_op1_handle = cshm.create_shared_memory_region(
            "output1_data", create_byte_size, device_id
        )

        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        cshm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        cshm.set_shared_memory_region(shm_ip1_handle, [input1_data])

        self.triton_client.register_cuda_shared_memory(
            "input0_data",
            cshm.get_raw_handle(shm_ip0_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "input1_data",
            cshm.get_raw_handle(shm_ip1_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "output0_data",
            cshm.get_raw_handle(shm_op0_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "output1_data",
            cshm.get_raw_handle(shm_op1_handle),
            device_id,
            register_byte_size,
        )
        return [shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle]

    def _cleanup_server(self, shm_handles):
        for shm_handle in shm_handles:
            cshm.destroy_shared_memory_region(shm_handle)

    def test_unregister_after_inference(self):
        # Unregister after inference
        error_msg = []
        shm_handles = self._configure_server()
        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            shm_handles[1],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))

        self.triton_client.unregister_cuda_shared_memory("output0_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 3)
        else:
            self.assertEqual(len(shm_status.regions), 3)
        self._cleanup_server(shm_handles)

    def test_register_after_inference(self):
        # Register after inference
        error_msg = []
        shm_handles = self._configure_server()
        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            shm_handles[1],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 64, 0)
        self.triton_client.register_cuda_shared_memory(
            "input2_data", cshm.get_raw_handle(shm_ip2_handle), 0, 64
        )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 5)
        else:
            self.assertEqual(len(shm_status.regions), 5)
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_too_big_shm(self):
        # Shared memory input region larger than needed - Throws error
        error_msg = []
        shm_handles = self._configure_server()
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 128, 0)
        self.triton_client.register_cuda_shared_memory(
            "input2_data", cshm.get_raw_handle(shm_ip2_handle), 0, 128
        )
        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            shm_ip2_handle,
            shm_handles[2],
            shm_handles[3],
            error_msg,
            big_shm_name="input2_data",
            big_shm_size=128,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            self.assertIn(
                "unexpected total byte size 128 for input 'INPUT1', expecting 64",
                error_msg[-1],
            )
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)

    def test_mixed_raw_shm(self):
        # Mix of shared memory and RAW inputs
        error_msg = []
        shm_handles = self._configure_server()
        input1_data = np.ones(shape=16, dtype=np.int32)
        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            [input1_data],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )

        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_server(shm_handles)

    def test_unregisterall(self):
        # Unregister all shared memory blocks
        shm_handles = self._configure_server()
        status_before = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(status_before), 4)
        else:
            self.assertEqual(len(status_before.regions), 4)
        self.triton_client.unregister_cuda_shared_memory()
        status_after = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(status_after), 0)
        else:
            self.assertEqual(len(status_after.regions), 0)
        self._cleanup_server(shm_handles)

    def test_register_out_of_bound(self):
        create_byte_size = self.DEFAULT_SHM_BYTE_SIZE
        # Verify various edge cases of registered region size don't go out of bounds of the actual created shm region's size.
        with self.assertRaisesRegex(
            InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=create_byte_size + 1,
            )

    def test_infer_offset_out_of_bound(self):
        # CUDA Shared memory offset outside output region - Throws error
        error_msg = []
        shm_handles = self._configure_server()
        if self.protocol == "http":
            # -32 when placed in an int64 signed type, to get a negative offset
            # by overflowing
            offset = 2**64 - 32
        else:
            # gRPC will throw an error if > 2**63 - 1, so instead test for
            # exceeding shm region size by 1 byte, given its size is 64 bytes
            offset = 64
        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            shm_handles[1],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            shm_output_offset=offset,
            protocol=self.protocol,
            use_system_shared_memory=False,
            use_cuda_shared_memory=True,
        )

        self.assertEqual(len(error_msg), 1)
        self.assertIn("Invalid offset for shared memory region", error_msg[0])
        self._cleanup_server(shm_handles)

    def test_infer_byte_size_out_of_bound(self):
        # Shared memory byte_size outside output region - Throws error
        error_msg = []
        shm_handles = self._configure_server()
        offset = 60
        byte_size = self.DEFAULT_SHM_BYTE_SIZE

        iu.shm_basic_infer(
            self,
            self.triton_client,
            shm_handles[0],
            shm_handles[1],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            shm_output_offset=offset,
            shm_output_byte_size=byte_size,
            protocol=self.protocol,
            use_system_shared_memory=False,
            use_cuda_shared_memory=True,
        )
        self.assertEqual(len(error_msg), 1)
        self.assertIn(
            "Invalid offset + byte size for shared memory region", error_msg[0]
        )
        self._cleanup_server(shm_handles)


if __name__ == "__main__":
    unittest.main()
