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
import sys
import tritonhttpclient
import tritongrpcclient
from tritonclientutils.utils import InferenceServerException
from time import time

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

FLAGS = None

start_time = None
finish_times = queue.Queue()

# Callback function used for async_infer(), it can capture
# additional information using functools.partial as long as the last
# two arguments are reserved for result and error
def completion_callback(result, error):
    if error is None:
      finish_times.put((result, time()-start_time))
    else:
      finish_times.put((error, time()-start_time))

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

    # We use model that takes 1 input tensor containing the delay number of cycles
    # to occupy an SM
    model_name = FLAGS.model
    model_version = "1"

    # Create the data for the input tensor.
    input_data = np.array([FLAGS.delay], dtype=np.int32)

    # Create the inference context for the model.
    if FLAGS.protocol.lower() == "grpc":
      triton_client = tritongrpcclient.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)
      inputs = [tritongrpcclient.InferInput('in', input_data.shape, "INT32")]
    else:
      triton_client = tritonhttpclient.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)
      inputs = [tritonhttpclient.InferInput('in', input_data.shape, "INT32")]

    inputs[0].set_data_from_numpy(input_data)

    # Send N inference requests to the inference server. Time the inference for both 
    # requests
    start_time = time()

    for i in range(FLAGS.count):
        triton_client.async_infer(model_name, inputs, partial(completion_callback),
                                  model_version=model_version,
                                  request_id=str(i),
                                  headers=FLAGS.http_headers)

    # Wait for N requests to finish
    finished_requests = 0
    max_completion_time = 0
    timeout = None

    while True:
        try:
          result, finish_time = finish_times.get()
        
        if type(result) == InferenceServerException:
            print(result)
            sys.exit(1)
        if FLAGS.protocol.lower() == "grpc":
            request_id = int(result.get_response().id)
        else:
            request_id = int(result.get_response()["id"])
        finished_requests += 1
        print("Request %d"%request_id + " finished in %f"%finish_time)
        max_completion_time = max(max_completion_time, finish_time)
        if (finished_requests == FLAGS.count):
            break
    
    print("Completion time for %d instances: %f secs"%(i+1, max_completion_time))
