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
import threading

# Raises error since invalid cuda shm region
try:
    shm_op0_handle = cshm.create_shared_memory_region("dummy_data", -1, 0)
except Exception as ex:
    assert str(ex) == "unable to create cuda shared memory handle"

shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=False)

# Create a valid cuda shared memory region
shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
# Fill data in cuda shared memory region
cshm.set_shared_memory_region(shm_op0_handle, [np.array([1,2], dtype=np.float32)])
# Unregister before register does not fail - does nothing
shared_memory_ctx.unregister(shm_op0_handle)
# Test if register is working
shared_memory_ctx.cuda_register(shm_op0_handle)
shm_status = shared_memory_ctx.get_shared_memory_status()
assert len(shm_status.shared_memory_region) == 1

# Raises error if registering already registered region
try:
    shared_memory_ctx.cuda_register(shm_op0_handle)
except Exception as ex:
    assert "shared memory block 'dummy_data' already in manager" in str(ex)

# Test if unregister after register works
shared_memory_ctx.unregister(shm_op0_handle)
cshm.destroy_shared_memory_region(shm_op0_handle)
shm_status = shared_memory_ctx.get_shared_memory_status()
assert len(shm_status.shared_memory_region) == 0

shm_op0_handle = cshm.create_shared_memory_region("output0_data", 64, 0)
shm_op1_handle = cshm.create_shared_memory_region("output1_data", 64, 0)
shm_ip0_handle = cshm.create_shared_memory_region("input0_data", 64, 0)
shm_ip1_handle = cshm.create_shared_memory_region("input1_data", 64, 0)
input0_data = np.arange(start=0, stop=16, dtype=np.int32)
input1_data = np.ones(shape=16, dtype=np.int32)
cshm.set_shared_memory_region(shm_ip0_handle, [input0_data])
cshm.set_shared_memory_region(shm_ip1_handle, [input1_data])
shared_memory_ctx.cuda_register(shm_ip0_handle)
shared_memory_ctx.cuda_register(shm_ip1_handle)
shared_memory_ctx.cuda_register(shm_op0_handle)
shared_memory_ctx.cuda_register(shm_op1_handle)

infer_ctx = InferContext("localhost:8000", ProtocolType.HTTP, "simple", -1, verbose=False)
def basic_inference(shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle, error_msg):
    try:
        results = infer_ctx.run({ 'INPUT0' : shm_ip0_handle, 'INPUT1' : shm_ip1_handle, },
                { 'OUTPUT0' : (InferContext.ResultFormat.RAW, shm_op0_handle),
                'OUTPUT1' : (InferContext.ResultFormat.RAW, shm_op1_handle)}, 1)
        assert (results['OUTPUT0'][0] == (input0_data + input1_data)).all()
    except Exception as ex:
        error_msg.append(str(ex.message()))

# Unregister during inference - inference fails and unregisters
error_msg = []
threads = []
threads.append(threading.Thread(target=basic_inference,
    args=(shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle, error_msg)))
threads.append(threading.Thread(target=shared_memory_ctx.unregister, args=(shm_op0_handle,)))
for t in threads:
    t.start()
for t in threads:
    t.join()

assert error_msg[0] == "shared memory block 'output0_data' not found in manager"
shared_memory_ctx.cuda_register(shm_op0_handle)

# Register during inference - Registered successfully
shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 128, 0)
threads[0] = threading.Thread(target=basic_inference,
    args=(shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle, error_msg))
threads[1] = threading.Thread(target=shared_memory_ctx.cuda_register, args=(shm_ip2_handle,))
for t in threads:
    t.start()
for t in threads:
    t.join()
if len(error_msg) > 1:
    raise Exception(error_msg[-1])

# Shared memory input region larger than needed - Throws error
shared_memory_ctx.unregister(shm_ip2_handle)
shared_memory_ctx.cuda_register(shm_ip2_handle)
basic_inference(shm_ip0_handle, shm_ip2_handle, shm_op0_handle, shm_op1_handle, error_msg)
if len(error_msg) > 1:
    if error_msg[-1] != "The input 'INPUT1' has shared memory of size 128 bytes"\
                            " while the expected size is 1 * 64 = 64 bytes":
        raise Exception(error_msg[-1])

# One of the inputs - INPUT1 does not use shared memory
basic_inference(shm_ip0_handle, [input1_data], shm_op0_handle, shm_op1_handle, error_msg)
if len(error_msg) > 2:
    raise Exception(error_msg[-1])

# get status before and after shared memory
status_before = shared_memory_ctx.get_shared_memory_status()
assert len(status_before.shared_memory_region) == 5
shared_memory_ctx.unregister_all()
status_after = shared_memory_ctx.get_shared_memory_status()
assert len(status_after.shared_memory_region) == 0

# cleanup (error with shm_op0_handle destroy since open on server)
cshm.destroy_shared_memory_region(shm_ip0_handle)
cshm.destroy_shared_memory_region(shm_ip1_handle)
cshm.destroy_shared_memory_region(shm_ip2_handle)
cshm.destroy_shared_memory_region(shm_op0_handle)
cshm.destroy_shared_memory_region(shm_op1_handle)
try:
    cshm.destroy_shared_memory_region(shm_op0_handle)
except:
    pass

print(error_msg)
