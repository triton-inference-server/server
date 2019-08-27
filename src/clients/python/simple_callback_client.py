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
from functools import partial
import numpy as np
import os
import sys
from builtins import range
from tensorrtserver.api import *

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

FLAGS = None

# User defined class to store infer_ctx and request id
# from callback function and let main thread to handle them
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_run_with_cb(), it can capture
# additional information using functools.partial as long as the last
# two arguments are reserved for InferContext and request id
def completion_callback(user_data, idx, infer_ctx, request_id):
    print("Callback " + str(idx) + " is called")
    user_data._completed_requests.put((infer_ctx, request_id, idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')

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
    ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input1_data = np.ones(shape=16, dtype=np.int32)

    request_cnt = 2
    user_data = UserData()

    # Send async inference
    for idx in range(request_cnt):
        result = ctx.async_run_with_cb(partial(completion_callback, user_data, idx),
                                        { 'INPUT0' : (input0_data,),
                                          'INPUT1' : (input1_data,) },
                                        { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                          'OUTPUT1' : InferContext.ResultFormat.RAW },
                                        batch_size)

    done_cnt = 0
    while True:
        # Wait for deferred items from callback functions
        (infer_ctx, request_id, idx) = user_data._completed_requests.get()

        # Retrieve results and error checking
        result = infer_ctx.get_async_run_results(request_id, True)
        output0_data = result['OUTPUT0'][0]
        output1_data = result['OUTPUT1'][0]
        print("Main thread retrieved request " + str(idx) + "'s results:")
        print(output0_data)
        print(output1_data)
        for i in range(16):
            if (input0_data[i] + input1_data[i]) != output0_data[i]:
                print("error: incorrect sum")
                sys.exit(1)
            if (input0_data[i] - input1_data[i]) != output1_data[i]:
                print("error: incorrect difference")
                sys.exit(1)
        done_cnt += 1
        if done_cnt == request_cnt:
            break
