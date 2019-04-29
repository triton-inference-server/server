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

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:',
                        help='Inference server URL. Default is localhost:.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-sp', "--status_port", type=int,
     help="The port for the server to listen on for HTTP Status requests.")
    parser.add_argument('-hp', "--health_port", type=int,
     help="The port for the server to listen on for HTTP Health requests.")
    parser.add_argument('-pp', "--profile_port", type=int,
     help="The port for the server to listen on for HTTP Profile requests.")
    parser.add_argument('-ip', "--infer_port", type=int,
     help="The port for the server to listen on for HTTP Infer requests.")

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str(FLAGS.protocol)

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "simple"
    model_version = -1
    batch_size = 1

    # Create the inference context for the model.
    if FLAGS.protocol == "http":
        # infer
        if FLAGS.infer_port !=-1:
            ctx = InferContext(FLAGS.url+str(FLAGS.infer_port), protocol, model_name, model_version, FLAGS.verbose)
        # health
        if FLAGS.health_port !=-1:
            hctx = ServerHealthContext(FLAGS.url+str(FLAGS.health_port), protocol, True)
            assert hctx.is_ready() == True
            assert hctx.is_live() == True
        # status
        if FLAGS.status_port !=-1:
            sctx = ServerStatusContext(FLAGS.url+str(FLAGS.status_port), protocol, model_name, True)
            ss = sctx.get_server_status()
            assert server_status.SERVER_READY == ss.ready_state
    elif FLAGS.protocol == "grpc":
        # infer
        if FLAGS.infer_port !=-1:
            ctx = InferContext(FLAGS.url+str(FLAGS.infer_port), protocol, model_name, model_version, FLAGS.verbose)
        # health
        if FLAGS.health_port !=-1:
            hctx = ServerHealthContext(FLAGS.url+str(FLAGS.health_port), protocol, True)
            assert hctx.is_ready() == True
            assert hctx.is_live() == True
        # status
        if FLAGS.status_port !=-1:
            sctx = ServerStatusContext(FLAGS.url+str(FLAGS.status_port), protocol, model_name, True)
            ss = sctx.get_server_status()
            assert server_status.SERVER_READY == ss.ready_state

    if FLAGS.infer_port !=-1:
        # Create the data for the two input tensors. Initialize the first
        # to unique integers and the second to all ones.
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)

        # Send inference request to the inference server. Get results for
        # both output tensors.
        result = ctx.run({ 'INPUT0' : (input0_data,),
                           'INPUT1' : (input1_data,) },
                         { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                           'OUTPUT1' : InferContext.ResultFormat.RAW },
                         batch_size)

        # We expect there to be 2 results (each with batch-size 1). Walk
        # over all 16 result elements and print the sum and difference
        # calculated by the model.
        output0_data = result['OUTPUT0'][0]
        output1_data = result['OUTPUT1'][0]

        for i in range(16):
            print(str(input0_data[i]) + " + " + str(input1_data[i]) + " = " + str(output0_data[i]))
            print(str(input0_data[i]) + " - " + str(input1_data[i]) + " = " + str(output1_data[i]))
            if (input0_data[i] + input1_data[i]) != output0_data[i]:
                print("error: incorrect sum");
                sys.exit(1);
            if (input0_data[i] - input1_data[i]) != output1_data[i]:
                print("error: incorrect difference");
                sys.exit(1);
