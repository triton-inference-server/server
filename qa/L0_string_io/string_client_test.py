#!/usr/bin/env python
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
import unittest

class ClientStringTest(unittest.TestCase):
    def test_tf_unicode_bytes(self):
        # We use a simple model that takes an input tensor of 6000 strings
        # and returns an output tensors of 6000 strings. The input
        # strings must represent integers. The output tensor is the
        # same as the input tensor.
        model_name = "graphdef_nobatch_zero_1_object"
        model_version = -1
        batch_size = 1

        # Create the inference context for the model.
        ctx = InferContext("localhost:8000", ProtocolType.HTTP, model_name, model_version, verbose=True)

        # Create the data for the input tensor. Initialize the tensor to the
        # byte representation of the first 6000 unicode characters.
        in0 = np.array([chr(i) for i in range(6000)])
        in0n = np.array([bytes(x, encoding='utf8') for x in in0])

        # Send inference request to the inference server. Get results for
        # both output tensors.
        result = ctx.run({ 'INPUT0' : (in0n,) },
                        { 'OUTPUT0' : InferContext.ResultFormat.RAW },
                        batch_size)

        # We expect there to be 1 results (with batch-size 1). Verify
        # that all the 6000 result elements are the same as the input.
        self.assertTrue(all(np.equal(in0n, result['OUTPUT0'][0])))

if __name__ == '__main__':
    unittest.main()