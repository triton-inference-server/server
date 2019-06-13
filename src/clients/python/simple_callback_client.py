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
from functools import partial
import numpy as np
import os
import threading
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

    # Send inference request to the inference server. Get results for
    # both output tensors.
    class Context:
        def __init__(self, input0_data, input1_data):
            self._lock = threading.Lock()
            self._cv = threading.Condition(self._lock)
            self._input0_data = input0_data
            self._input1_data = input1_data
            self._request_cnt = 2
            self._done_cnt = 0
            self._recent_ctx_obj = None
            self._recent_request_id = None
            self._success = True
    
    context = Context(input0_data, input1_data)
    def active_callback(context, idx, ctx_obj, request_id):
        result = ctx_obj.get_async_run_results(request_id, True)
        context._cv.acquire(True)
        print("Callback " + str(idx) + " is called")
        output0_data = result['OUTPUT0'][0]
        output1_data = result['OUTPUT1'][0]
        print(output0_data)
        print(output1_data)
        for i in range(16):
            if (context._input0_data[i] + context._input1_data[i]) != output0_data[i]:
                print("error: incorrect sum")
                context._success = False
                break
            if (context._input0_data[i] - context._input1_data[i]) != output1_data[i]:
                print("error: incorrect difference")
                context._success = False
                break
        context._done_cnt += 1
        context._cv.notify_all()
        context._cv.release()

    # Send async inference and wait for them to finish
    for idx in range(context._request_cnt):
        result = ctx.async_run_with_cb(partial(active_callback, context, idx),
                                  { 'INPUT0' : (input0_data,),
                                    'INPUT1' : (input1_data,) },
                                  { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                    'OUTPUT1' : InferContext.ResultFormat.RAW },
                                  batch_size)

    context._cv.acquire()
    while context._done_cnt != context._request_cnt:
        context._cv.wait()
    context._cv.release()

    if not context._success:
      sys.exit(1)

    # defer retrieval to main thread
    def passive_callback(context, ctx_obj, request_id):
        context._cv.acquire(True)
        # set related InferContext and request id to shared Context object
        context._recent_ctx_obj = ctx_obj
        context._recent_request_id = request_id
        context._cv.notify_all()
        context._cv.release()

    # Send async inference and wait for its callback is invoked
    result = ctx.async_run_with_cb(partial(passive_callback, context),
                              { 'INPUT0' : (input0_data,),
                                'INPUT1' : (input1_data,) },
                              { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                'OUTPUT1' : InferContext.ResultFormat.RAW },
                              batch_size)

    context._cv.acquire()
    while context._recent_ctx_obj is None:
        context._cv.wait()
    context._cv.release()

    # Retrieve InferContext and request id from the shared context
    result = context._recent_ctx_obj.get_async_run_results(context._recent_request_id, True)
    print("Deferred retrieval to main thread")
    output0_data = result['OUTPUT0'][0]
    output1_data = result['OUTPUT1'][0]
    print(output0_data)
    print(output1_data)
    for i in range(16):
        if (context._input0_data[i] + context._input1_data[i]) != output0_data[i]:
            print("error: incorrect sum")
            sys.exit(1)
        if (context._input0_data[i] - context._input1_data[i]) != output1_data[i]:
            print("error: incorrect difference")
            self._success = False
            sys.exit(1)
