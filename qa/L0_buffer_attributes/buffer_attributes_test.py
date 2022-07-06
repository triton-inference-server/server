# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient.utils import triton_to_np_dtype
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


class BufferAttributesTest(tu.TestResultCollector):

    def test_buffer_attributes(self):
        model_name = 'bls'

        # Infer
        clients = [
            httpclient.InferenceServerClient(url='localhost:8000'),
            grpcclient.InferenceServerClient(url='localhost:8001')
        ]
        triton_clients = [httpclient, grpcclient]
        for i, client in enumerate(clients):

            # To make sure no shared memory regions are registered with the
            # server.
            client.unregister_system_shared_memory()
            client.unregister_cuda_shared_memory()

            triton_client = triton_clients[i]
            inputs = []
            outputs = []
            inputs.append(triton_client.InferInput('INPUT0', [1, 1000],
                                                   "INT32"))

            input0_data = np.arange(start=0, stop=1000, dtype=np.int32)
            input0_data = np.expand_dims(input0_data, axis=0)

            input_byte_size = input0_data.size * input0_data.itemsize
            output_byte_size = input_byte_size

            shm_ip0_handle = cudashm.create_shared_memory_region(
                "input0_data", input_byte_size, 0)
            shm_op0_handle = cudashm.create_shared_memory_region(
                "output0_data", output_byte_size, 0)

            client.register_cuda_shared_memory(
                "input0_data", cudashm.get_raw_handle(shm_ip0_handle), 0,
                input_byte_size)
            client.register_cuda_shared_memory(
                "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0,
                input_byte_size)

            cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data])
            inputs[0].set_shared_memory("input0_data", input_byte_size)

            if triton_client is grpcclient:
                outputs.append(triton_client.InferRequestedOutput('OUTPUT0'))
                outputs[0].set_shared_memory("output0_data", output_byte_size)
            else:
                outputs.append(
                    triton_client.InferRequestedOutput('OUTPUT0',
                                                       binary_data=True))
                outputs[0].set_shared_memory("output0_data", output_byte_size)

            results = client.infer(model_name=model_name,
                                   inputs=inputs,
                                   outputs=outputs)

            output0 = results.get_output("OUTPUT0")
            self.assertIsNotNone(output0)
            if triton_client is grpcclient:
                output0_data = cudashm.get_contents_as_numpy(
                    shm_op0_handle, triton_to_np_dtype(output0.datatype),
                    output0.shape)
            else:
                output0_data = cudashm.get_contents_as_numpy(
                    shm_op0_handle, triton_to_np_dtype(output0['datatype']),
                    output0['shape'])
            self.assertTrue(np.all(output0_data == input0_data))


if __name__ == '__main__':
    unittest.main()
