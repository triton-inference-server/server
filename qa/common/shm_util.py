# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
import os
import numpy as np
import tritonsharedmemoryutils.shared_memory as shm
import tritonsharedmemoryutils.cuda_shared_memory as cudashm
import tritonhttpclient.core as httpclient
from tritonhttpclient.utils import *
from ctypes import *

def _range_repr_dtype(dtype):
    if dtype == np.float64:
        return np.int32
    elif dtype == np.float32:
        return np.int16
    elif dtype == np.float16:
        return np.int8
    elif dtype == np.object:  # TYPE_STRING
        return np.int32
    return dtype

def create_register_shm_regions(input0_list, input1_list, output0_byte_size,
                                output1_byte_size, outputs, shm_region_names,
                                precreated_shm_regions, use_system_shared_memory,
                                use_cuda_shared_memory):
    if use_system_shared_memory and use_cuda_shared_memory:
        raise ValueError(
            "Cannot set both System and CUDA shared memory flags to 1")

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return [], None, None

    triton_client = httpclient.InferenceServerClient("localhost:8000")

    if input0_list[0].dtype == np.object:
        input0_list_tmp = [serialize_byte_tensor(i0) for i0 in input0_list]
        input1_list_tmp = [serialize_byte_tensor(i1) for i1 in input1_list]
    else:
        input0_list_tmp = input0_list
        input1_list_tmp = input1_list

    input0_byte_size = sum([i0.nbytes for i0 in input0_list_tmp])
    input1_byte_size = sum([i1.nbytes for i1 in input1_list_tmp])

    if shm_region_names is None:
        shm_region_names = ['input0', 'input1', 'output0', 'output1']

    if use_system_shared_memory:
        shm_ip0_handle = shm.create_shared_memory_region(shm_region_names[0]+'_data',
                                                         '/'+shm_region_names[0], input0_byte_size)
        shm_ip1_handle = shm.create_shared_memory_region(shm_region_names[1]+'_data',
                                                         '/'+shm_region_names[1], input1_byte_size)
        triton_client.unregister_system_shared_memory(
            shm_region_names[0]+'_data')
        triton_client.unregister_system_shared_memory(
            shm_region_names[1]+'_data')
        triton_client.register_system_shared_memory(shm_region_names[0]+'_data',
                                                    '/'+shm_region_names[0], input0_byte_size)
        triton_client.register_system_shared_memory(shm_region_names[1]+'_data',
                                                    '/'+shm_region_names[1], input1_byte_size)

        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = shm.create_shared_memory_region(shm_region_names[2]+'_data',
                                                                 '/'+shm_region_names[2], output0_byte_size)
                triton_client.unregister_system_shared_memory(
                    shm_region_names[2]+'_data')
                triton_client.register_system_shared_memory(shm_region_names[2]+'_data',
                                                            '/'+shm_region_names[2], output0_byte_size)
            else:
                shm_op0_handle = precreated_shm_regions[0]
            i += 1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = shm.create_shared_memory_region(shm_region_names[2+i]+'_data',
                                                                 '/'+shm_region_names[2+i], output1_byte_size)
                triton_client.unregister_system_shared_memory(
                    shm_region_names[2+i]+'_data')
                triton_client.register_system_shared_memory(shm_region_names[2+i]+'_data',
                                                            '/'+shm_region_names[2+i], output1_byte_size)
            else:
                shm_op1_handle = precreated_shm_regions[i]

        shm.set_shared_memory_region(shm_ip0_handle, input0_list_tmp)
        shm.set_shared_memory_region(shm_ip1_handle, input1_list_tmp)

    if use_cuda_shared_memory:
        shm_ip0_handle = cudashm.create_shared_memory_region(shm_region_names[0]+'_data', input0_byte_size, 0)
        shm_ip1_handle = cudashm.create_shared_memory_region(shm_region_names[1]+'_data', input1_byte_size, 0)
        triton_client.unregister_cuda_shared_memory(
            shm_region_names[0]+'_data')
        triton_client.unregister_cuda_shared_memory(
            shm_region_names[1]+'_data')
        triton_client.register_cuda_shared_memory(shm_region_names[0]+'_data',
                                                  cudashm.get_raw_handle(shm_ip0_handle), 0, input0_byte_size)
        triton_client.register_cuda_shared_memory(shm_region_names[1]+'_data',
                                                  cudashm.get_raw_handle(shm_ip1_handle), 0, input1_byte_size)
        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = cudashm.create_shared_memory_region(shm_region_names[2]+'_data',
                                                                     output0_byte_size, 0)
                triton_client.unregister_cuda_shared_memory(
                    shm_region_names[2]+'_data')
                triton_client.register_cuda_shared_memory(shm_region_names[2]+'_data',
                                                          cudashm.get_raw_handle(shm_op0_handle), 0, output0_byte_size)
            else:
                shm_op0_handle = precreated_shm_regions[0]
            i += 1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = cudashm.create_shared_memory_region(shm_region_names[2+i]+'_data',
                                                                     output1_byte_size, 0)
                triton_client.unregister_cuda_shared_memory(
                    shm_region_names[2+i]+'_data')
                triton_client.register_cuda_shared_memory(shm_region_names[2+i]+'_data',
                                                          cudashm.get_raw_handle(shm_op1_handle), 0, output1_byte_size)
            else:
                shm_op1_handle = precreated_shm_regions[i]

        cudashm.set_shared_memory_region(shm_ip0_handle, input0_list_tmp)
        cudashm.set_shared_memory_region(shm_ip1_handle, input1_list_tmp)

    return shm_region_names, shm_op0_handle, shm_op1_handle


def set_shm_regions(inputs, shm_region_names, use_system_shared_memory, use_cuda_shared_memory, 
                    input0_byte_size, input1_byte_size):
    if use_system_shared_memory:
        inputs[0].set_shared_memory(shm_region_names[0]+'_data', input0_byte_size)
        inputs[1].set_shared_memory(shm_region_names[1]+'_data', input1_byte_size)

    if use_cuda_shared_memory:
        inputs[0].set_shared_memory(shm_region_names[0]+'_data', input0_byte_size)
        inputs[1].set_shared_memory(shm_region_names[1]+'_data', input1_byte_size)


def unregister_cleanup_shm_regions(shm_regions, precreated_shm_regions, outputs, use_system_shared_memory, use_cuda_shared_memory):
    if not (use_system_shared_memory or use_cuda_shared_memory):
        return None

    triton_client = httpclient.InferenceServerClient("localhost:8000")

    if use_cuda_shared_memory:
        triton_client.unregister_cuda_shared_memory(shm_regions[0]+'_data')
        triton_client.unregister_cuda_shared_memory(shm_regions[1]+'_data')
    else:
        triton_client.unregister_system_shared_memory(shm_regions[0]+'_data')
        triton_client.unregister_system_shared_memory(shm_regions[1]+'_data')

    if precreated_shm_regions is None:
        i = 0
        if "OUTPUT0" in outputs:
            if use_cuda_shared_memory:
                triton_client.unregister_cuda_shared_memory(
                    shm_regions[2]+'_data')
            else:
                triton_client.unregister_system_shared_memory(
                    shm_regions[2]+'_data')
            i += 1
        if "OUTPUT1" in outputs:
            if use_cuda_shared_memory:
                triton_client.unregister_cuda_shared_memory(
                    shm_regions[2+i]+'_data')
            else:
                triton_client.unregister_system_shared_memory(
                    shm_regions[2+i]+'_data')

