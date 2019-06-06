#!/usr/bin/python

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

import argparse
import numpy as np
import os
from builtins import range
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='Name of the model to test')
    FLAGS = parser.parse_args()

    protocol = ProtocolType.from_str("http")

    # We use a simple model that takes 1 input tensors of 1 integer
    # and returns variable number of (1, 2 or 3) output tensors
    # of 1 integer each. Each output tensor is same as the input
    model_name = FLAGS.model_name
    version = 1
    batch_size = 1
    verbose = 1

    ctx = InferContext("localhost:8000", protocol, model_name, version, verbose)

    for if_type in (InferContext.ResultFormat.RAW, (InferContext.ResultFormat.CLASS, 1)):
        input0_data = np.array([0], dtype=np.int32)

        # Send inference request to the inference server.
        result = ctx.run({ 'INPUT__0' : (input0_data,) },
                         { 'OUTPUT__0' : if_type,
                           'OUTPUT__1' : if_type },
                         batch_size)

        # We expect there to be exactly 2 results (each with batch-size 1).
        # Verify the identity calculated by the model.
        assert len([ k for k in results.keys() if 'OUTPUT__' in k]) == 2
        output0_data = result['OUTPUT__0'][0]
        output1_data = result['OUTPUT__1'][0]
        if if_type == InferContext.ResultFormat.RAW:
            assert np.equal(input0_data, output0_data).all() == True
            assert np.equal(input0_data, output1_data).all() == True
        else:
            assert input0_data[0] == output0_data[1]
            assert input0_data[0] == output1_data[1]

        #  Similarly, this must have 1 o/p
        input0_data = np.array([-1], dtype=np.int32)
        result = ctx.run({ 'INPUT__0' : (input0_data,) },
                         { 'OUTPUT__0' : if_type},
                         batch_size)
        assert len([ k for k in results.keys() if 'OUTPUT__' in k]) == 1
        output0_data = result['OUTPUT__0'][0]
        if if_type == InferContext.ResultFormat.RAW:
            assert np.equal(input0_data, output0_data).all() == True
        else:
            assert input0_data[0] == output0_data[1]

        #  Similarly, this must have 3 o/p
        input0_data = np.array([1], dtype=np.int32)
        result = ctx.run({ 'INPUT__0' : (input0_data,) },
                         { 'OUTPUT__0' : if_type,
                           'OUTPUT__1' : if_type,
                           'OUTPUT__2' : if_type },
                         batch_size)
        assert len([ k for k in results.keys() if 'OUTPUT__' in k]) == 3
        output0_data = result['OUTPUT__0'][0]
        output1_data = result['OUTPUT__1'][0]
        output2_data = result['OUTPUT__2'][0]
        if if_type == InferContext.ResultFormat.RAW:
            assert np.equal(input0_data, output0_data).all() == True
            assert np.equal(input0_data, output1_data).all() == True
            assert np.equal(input0_data, output2_data).all() == True
        else:
            assert input0_data[0] == output0_data[1]
            assert input0_data[0] == output1_data[1]
            assert input0_data[0] == output2_data[1]

        #  Similarly, this must have 1 2x2 o/p
        input0_data = np.array([2], dtype=np.int32)
        result = ctx.run({ 'INPUT__0' : (input0_data,) },
                         { 'OUTPUT__0' : if_type},
                         batch_size)
        assert len([ k for k in results.keys() if 'OUTPUT__' in k]) == 1
        output0_data = result['OUTPUT__0'][0]
        if if_type == InferContext.ResultFormat.RAW:
            assert output0_data.shape == (2,2)
            assert np.equal(output0_data.flatten(), np.ones(4, dtype=np.int32)).all() == True

        #  Similarly, this must have 1 float o/p
        input0_data = np.array([3], dtype=np.int32)
        result = ctx.run({ 'INPUT__0' : (input0_data,) },
                         { 'OUTPUT__0' : if_type},
                         batch_size)
        assert len([ k for k in results.keys() if 'OUTPUT__' in k]) == 1
        output0_data = result['OUTPUT__0'][0]
        if if_type == InferContext.ResultFormat.RAW:
            assert output0_data.dtype == 'float32'
            assert output0_data[0] == 1.0
        else:
            assert output0_data[1] == 1.0
