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
from functools import partial
from builtins import range
from tensorrtserver.api import *
from time import time

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

FLAGS = None

start_time = None
finish_times = queue.Queue()

# Callback function used for async_run(), it can capture
# additional information using functools.partial as long as the last
# two arguments are reserved for InferContext and request id
def completion_callback(infer_ctx, request_id):
    finish_times.put((request_id, time()-start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-m', '--model', type=str, required=False, default='graphdef_busyop',
                        help='Model to request for inference. Default is graphdef_busyop')
    parser.add_argument('-c', '--count', type=int, required=False, default=1,
                        help='Number of inference requests to send. Default is 1')
    parser.add_argument('-n', '--delay', type=int, required=False, default=12000000,
                        help='Number of cycles to perform the busyloop op. Default is 12000000')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str(FLAGS.protocol)

    # We use model that takes 1 input tensor containing the delay number of cycles
    # to occupy an SM
    model_name = FLAGS.model
    model_version = 1
    batch_size = 1

    # Create the inference context for the model.
    infer_ctx = InferContext(FLAGS.url, protocol, model_name, model_version,
                             http_headers=FLAGS.http_headers, verbose=FLAGS.verbose)

    # Create the data for the input tensor.
    input_data = np.array([FLAGS.delay], dtype=np.int32)

    # Send N inference requests to the inference server. Time the inference for both 
    # requests
    start_time = time()

    for i in range(FLAGS.count):
        infer_ctx.async_run(partial(completion_callback), 
            { 'in' : (input_data,) }, 
            { 'out' : InferContext.ResultFormat.RAW }, 
            batch_size)

    # Wait for N requests to finish
    finished_requests = 0
    max_completion_time = 0
    while True:
        request_id, finish_time = finish_times.get()
        result = infer_ctx.get_async_run_results(request_id)
        finished_requests += 1
        print("Request %d"%request_id + " finished in %f"%finish_time)
        max_completion_time = max(max_completion_time, finish_time)
        if (finished_requests == FLAGS.count):
            break
    
    print("Completion time for %d instances: %f secs"%(i+1, max_completion_time))
