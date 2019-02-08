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
import time
from builtins import range
from tensorrtserver.api import *

FLAGS = None


def send(ctx, control, value):
    # Create the tensor for CONTROL and INPUT values.
    control_data = np.full(shape=[1], fill_value=control, dtype=np.int32)
    value_data = np.full(shape=[1], fill_value=value, dtype=np.int32)

    result = ctx.run({ 'CONTROL' : (control_data,),
                       'INPUT' : (value_data,) },
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     1)
    return result

def async_send(ctx, control, value):
    # Create the tensor for CONTROL and INPUT values.
    control_data = np.full(shape=[1], fill_value=control, dtype=np.int32)
    value_data = np.full(shape=[1], fill_value=value, dtype=np.int32)

    request_id = ctx.async_run({ 'CONTROL' : (control_data,),
                       'INPUT' : (value_data,) },
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     1)
    return request_id

def async_receive(ctx, request_id):
    return ctx.get_async_run_results(request_id, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL and it gRPC port. Default is localhost:8001.')
    parser.add_argument('-a', '--async', action="store_true", required=False, default=False,
                        help='Enable asynchronous inference')
    parser.add_argument('-r', '--reverse', action="store_true", required=False, default=False,
                        help='Enable to run non-streaming context first')

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str("grpc")

    # We use the custom "sequence" model which takes 2 inputs, one
    # control and one the actual input value. The output is the
    # accumulated value of the inputs. See src/custom/sequence.
    model_name = "simple_sequence"
    model_version = -1
    batch_size = 1

    # Create 2 inference context with different correlation ID. We
    # will use these to send to sequences of inference requests. Must
    # use a non-zero correlation ID since zero indicates no
    # correlation ID.
    
    # For the two different contexts, one is using streaming while the other
    # isn't. Then we can compare their difference in sync/async runs
    correlation_id0 = 1
    ctx0 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id0, verbose=FLAGS.verbose, streaming=True)

    correlation_id1 = 2
    ctx1 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id1, verbose=FLAGS.verbose, streaming=False)

    # Create warmup context and warm up the server to avoid time difference due to run order
    warmup_correlation_id1 = 3
    warmup_ctx = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=warmup_correlation_id1, verbose=False, streaming=False)

    # Now send the inference sequences.. FIXME, for now must send the
    # proper control values since TRTIS is not yet doing it.
    #
    # First reset accumulator for both sequences.
    values = [11, 7, 5, 3, 1, 0]
    result0_list = []
    result1_list = []
    seq0_sec = 0
    seq1_sec = 0

    # warmup
    send(warmup_ctx, control=1, value=0)
    for v in values:
        send(warmup_ctx, control=0, value=v)

    ctxs = []
    if not FLAGS.reverse:
        ctxs = [ctx0, ctx1]
    else:
        ctxs = [ctx1, ctx0]
    
    if FLAGS.async:
        request0_ids = []
        request1_ids = []

        start = time.time()
        request0_ids.append(async_send(ctxs[0], control=1, value=0))
        for v in values:
            request0_ids.append(async_send(ctxs[0], control=0, value=v))
        for request_id in request0_ids:
            result0_list.append(async_receive(ctxs[0], request_id))
        seq0_sec = time.time() - start

        start = time.time()
        request1_ids.append(async_send(ctxs[1], control=1, value=0))
        for v in values:
            request1_ids.append(async_send(ctxs[1], control=0, value=v))
        for request_id in request1_ids:
            result1_list.append(async_receive(ctxs[1], request_id))
        seq1_sec = time.time() - start
    else:
        start = time.time()
        result0_list.append(send(ctxs[0], control=1, value=0))
        for v in values:
            result0_list.append(send(ctxs[0], control=0, value=v))
        seq0_sec = time.time() - start

        start = time.time()
        result1_list.append(send(ctxs[1], control=1, value=0))
        for v in values:
            result1_list.append(send(ctxs[1], control=0, value=v))
        seq1_sec = time.time() - start

    NANOS = 1000000000
    if not FLAGS.reverse:
        print("streaming : non-streaming")
    else:
        print("non-streaming : streaming")
    print(str(seq0_sec * NANOS) + " ns : " + str(seq1_sec * NANOS) + " ns")
    print("Results")
    for i in range(len(result0_list)):
        print("[" + str(i) + "] " + str(result0_list[i]['OUTPUT'][0]) + " : " + str(result1_list[i]['OUTPUT'][0]))
