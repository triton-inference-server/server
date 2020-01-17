#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Name of model.')

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str(FLAGS.protocol)

    # Run the custom_modulo model, which depends on a custom mod operation
    model_name = FLAGS.model
    model_version = -1
    batch_size = 1
    elements = 10

    # Create the inference context for the model.
    ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

    # Create the data for one input tensor.
    input0_data = np.arange(start=1, stop=1+elements, dtype=np.float32)
    input1_data = np.array([2] * elements, dtype=np.float32)

    result = ctx.run({ 'INPUT__0' : (input0_data,), 'INPUT__1' : (input1_data,)},
                     { 'OUTPUT__0' : InferContext.ResultFormat.RAW },
                     batch_size)

    # We expect 1 result of size 10 with alternating 1 and 0.
    print(result)
    output0_data = result['OUTPUT__0'][0]

    for i in range(elements):
        print(str(i) + ": " + str(input0_data[i]) + " % " +  str(input1_data[i]) + " = " + str(output0_data[i]))
        if ((input0_data[i] % input1_data[i]) != output0_data[i]):
            print("error: incorrect value");
            sys.exit(1);
