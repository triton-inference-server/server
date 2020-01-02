# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
from tensorrtserver.api import *
import tensorrtserver.shared_memory as shm
import tensorrtserver.cuda_shared_memory as cudashm
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

def _prepend_string_size(input_values):
    input_list = []
    for input_value in input_values:
        input_list.append(serialize_string_tensor(input_value))
    return input_list

def create_register_set_shm_regions(input0_list, input1_list, expected0_list,
                                    expected1_list, outputs, shm_region_names,
                                    precreated_shm_regions, use_system_shared_memory,
                                    use_cuda_shared_memory):
    if use_system_shared_memory and use_cuda_shared_memory:
        raise ValueError("Cannot set both System and CUDA shared memory flags to 1")

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return []

    shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)

    input0_byte_size = sum([i0.nbytes for i0 in input0_list])
    input1_byte_size = sum([i1.nbytes for i1 in input1_list])
    output0_byte_size = sum([e0.nbytes for e0 in expected0_list])
    output1_byte_size = sum([e1.nbytes for e1 in expected1_list])
    shm_io_handles = []

    if shm_region_names is None:
        shm_region_names = ['input0', 'input1', 'output0', 'output1']

    if use_system_shared_memory:
        shm_ip0_handle = shm.create_shared_memory_region(shm_region_names[0]+'_data',
                                                    '/'+shm_region_names[0], input0_byte_size)
        shm_ip1_handle = shm.create_shared_memory_region(shm_region_names[1]+'_data',
                                                    '/'+shm_region_names[1], input1_byte_size)
        shm.set_shared_memory_region(shm_ip0_handle, input0_list)
        shm.set_shared_memory_region(shm_ip1_handle, input1_list)
        shared_memory_ctx.unregister(shm_ip0_handle)
        shared_memory_ctx.register(shm_ip0_handle)
        shared_memory_ctx.unregister(shm_ip1_handle)
        shared_memory_ctx.register(shm_ip1_handle)
        shm_io_handles.extend([shm_ip0_handle, shm_ip1_handle])

        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = shm.create_shared_memory_region(shm_region_names[2]+'_data',
                                                            '/'+shm_region_names[2], output0_byte_size)
                shared_memory_ctx.unregister(shm_op0_handle)
                shared_memory_ctx.register(shm_op0_handle)
            else:
                shm_op0_handle = precreated_shm_regions[0]
            shm_io_handles.append(shm_op0_handle)
            i +=1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = shm.create_shared_memory_region(shm_region_names[2+i]+'_data',
                                                            '/'+shm_region_names[2+i], output1_byte_size)
                shared_memory_ctx.unregister(shm_op1_handle)
                shared_memory_ctx.register(shm_op1_handle)
            else:
                shm_op1_handle = precreated_shm_regions[i]
            shm_io_handles.append(shm_op1_handle)

    if use_cuda_shared_memory:
        shm_ip0_handle = cudashm.create_shared_memory_region(shm_region_names[0]+'_data',
                                                    input0_byte_size, 0)
        shm_ip1_handle = cudashm.create_shared_memory_region(shm_region_names[1]+'_data',
                                                    input1_byte_size, 0)
        cudashm.set_shared_memory_region(shm_ip0_handle, input0_list)
        cudashm.set_shared_memory_region(shm_ip1_handle, input1_list)
        shared_memory_ctx.unregister(shm_ip0_handle)
        shared_memory_ctx.cuda_register(shm_ip0_handle)
        shared_memory_ctx.unregister(shm_ip1_handle)
        shared_memory_ctx.cuda_register(shm_ip1_handle)
        shm_io_handles.extend([shm_ip0_handle, shm_ip1_handle])

        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = cudashm.create_shared_memory_region(shm_region_names[2]+'_data',
                                                            output0_byte_size, 0)
                shared_memory_ctx.unregister(shm_op0_handle)
                shared_memory_ctx.cuda_register(shm_op0_handle)
            else:
                shm_op0_handle = precreated_shm_regions[0]
            shm_io_handles.append(shm_op0_handle)
            i+=1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = cudashm.create_shared_memory_region(shm_region_names[2+i]+'_data',
                                                            output1_byte_size, 0)
                shared_memory_ctx.unregister(shm_op1_handle)
                shared_memory_ctx.cuda_register(shm_op1_handle)
            else:
                shm_op1_handle = precreated_shm_regions[i]
            shm_io_handles.append(shm_op1_handle)

    return shm_io_handles

def unregister_cleanup_shm_regions(shm_handles, precreated_shm_regions, outputs, use_system_shared_memory, use_cuda_shared_memory):
    if not (use_system_shared_memory or use_cuda_shared_memory):
        return None

    shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)
    shared_memory_ctx.unregister(shm_handles[0])
    shared_memory_ctx.unregister(shm_handles[1])
    if use_cuda_shared_memory:
        cudashm.destroy_shared_memory_region(shm_handles[0])
        cudashm.destroy_shared_memory_region(shm_handles[1])
    else:
        shm.destroy_shared_memory_region(shm_handles[0])
        shm.destroy_shared_memory_region(shm_handles[1])

    if precreated_shm_regions is None:
        i = 0
        if "OUTPUT0" in outputs:
            shared_memory_ctx.unregister(shm_handles[2])
            if use_cuda_shared_memory:
                cudashm.destroy_shared_memory_region(shm_handles[2])
            else:
                shm.destroy_shared_memory_region(shm_handles[2])
            i +=1
        if "OUTPUT1" in outputs:
            shared_memory_ctx.unregister(shm_handles[2+i])
            if use_cuda_shared_memory:
                cudashm.destroy_shared_memory_region(shm_handles[2+i])
            else:
                shm.destroy_shared_memory_region(shm_handles[2+i])

def create_register_set_either_shm_region(shm_region_names, input_list, input_byte_size,
                                        output_byte_size, shared_memory_ctx,
                                        use_system_shared_memory, use_cuda_shared_memory):
    if use_cuda_shared_memory and use_system_shared_memory:
        raise ValueError("Cannot set both System and CUDA shared memory flags to 1")

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return []

    if use_system_shared_memory:
        shm_ip_handle = cudashm.create_shared_memory_region(shm_region_names[0]+"_data",
                                                            input_byte_size, 0)
        shm_op_handle = cudashm.create_shared_memory_region(shm_region_names[1]+"_data",
                                                            output_byte_size, 0)
        shared_memory_ctx.cuda_register(shm_ip_handle)
        shared_memory_ctx.cuda_register(shm_op_handle)
        # copy data into shared memory region for input values
        cudashm.set_shared_memory_region(shm_ip_handle, input_list)
    elif use_cuda_shared_memory:
        shm_ip_handle = shm.create_shared_memory_region(shm_region_names[0]+"_data",\
                                    "/"+shm_region_names[0], input_byte_size)
        shm_op_handle = shm.create_shared_memory_region(shm_region_names[1]+"_data",\
                                    "/"+shm_region_names[1], output_byte_size)
        shared_memory_ctx.register(shm_ip_handle)
        shared_memory_ctx.register(shm_op_handle)
        # copy data into shared memory region for input values
        shm.set_shared_memory_region(shm_ip_handle, input_list)
    
    return [shm_ip_handle, shm_op_handle]

def destroy_either_shm_region(shm_handle, use_system_shared_memory, use_cuda_shared_memory):
    if use_cuda_shared_memory:
        cudashm.destroy_shared_memory_region(shm_handle)
    else:
        shm.destroy_shared_memory_region(shm_handle)
