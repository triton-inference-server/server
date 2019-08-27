#!/usr/bin/env python

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
import sys
from builtins import range
from tensorrtserver.api import *

FLAGS = None

def send(ctx, value, start_of_sequence=False, end_of_sequence=False):
    # Create the tensor for INPUT.
    value_data = np.full(shape=[1], fill_value=value, dtype=np.int32)

    flags = InferRequestHeader.FLAG_NONE
    if start_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
    if end_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

    result = ctx.run({ 'INPUT' : (value_data,) },
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     batch_size=1, flags=flags)
    return result

def async_send(ctx, value, start_of_sequence=False, end_of_sequence=False):
    # Create the tensor for INPUT.
    value_data = np.full(shape=[1], fill_value=value, dtype=np.int32)

    flags = InferRequestHeader.FLAG_NONE
    if start_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
    if end_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

    request_id = ctx.async_run({ 'INPUT' : (value_data,) },
                               { 'OUTPUT' : InferContext.ResultFormat.RAW },
                               batch_size=1, flags=flags)
    return request_id

def async_receive(ctx, request_id):
    return ctx.get_async_run_results(request_id, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL and it gRPC port. Default is localhost:8001.')
    parser.add_argument('-a', '--async', dest="async_set", action="store_true", required=False,
                        default=False, help='Enable asynchronous inference')
    parser.add_argument('-r', '--reverse', action="store_true", required=False, default=False,
                        help='Enable to run non-streaming context first')

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str("grpc")

    # We use the custom "sequence" model which takes 1 input
    # value. The output is the accumulated value of the inputs. See
    # src/custom/sequence.
    model_name = "simple_sequence"
    model_version = -1
    batch_size = 1

    # Create 2 inference context with different correlation ID. We
    # will use these to send to sequences of inference requests. Must
    # use a non-zero correlation ID since zero indicates no
    # correlation ID.
    values = [11, 7, 5, 3, 2, 0, 1]

    # Create two different contexts, in the sync case we can use one
    # streaming and one not streaming. In the async case must use
    # streaming for both since async+non-streaming means that order of
    # requests reaching inference server is not guaranteed.
    correlation_id0 = 1000
    ctx0 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id0, verbose=FLAGS.verbose,
                        streaming=True)

    correlation_id1 = 1001
    ctx1 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id1, verbose=FLAGS.verbose,
                        streaming=FLAGS.async_set)

    # Now send the inference sequences..
    ctxs = []
    if not FLAGS.reverse:
        ctxs = [ctx0, ctx1]
    else:
        ctxs = [ctx1, ctx0]

    result0_list = []
    result1_list = []

    if FLAGS.async_set:
        request0_ids = []
        request1_ids = []

        request0_ids.append(async_send(ctxs[0], value=0, start_of_sequence=True))
        request1_ids.append(async_send(ctxs[1], value=100, start_of_sequence=True))
        for v in values:
            request0_ids.append(async_send(ctxs[0], value=v,
                                           start_of_sequence=False, end_of_sequence=(v == 1)))
            request1_ids.append(async_send(ctxs[1], value=-v,
                                           start_of_sequence=False, end_of_sequence=(v == 1)))
        for request_id in request0_ids:
            result0_list.append(async_receive(ctxs[0], request_id))
        for request_id in request1_ids:
            result1_list.append(async_receive(ctxs[1], request_id))
    else:
        result0_list.append(send(ctxs[0], value=0, start_of_sequence=True))
        result1_list.append(send(ctxs[1], value=100, start_of_sequence=True))
        for v in values:
            result0_list.append(send(ctxs[0], value=v,
                                     start_of_sequence=False, end_of_sequence=(v == 1)))
            result1_list.append(send(ctxs[1], value=-v,
                                     start_of_sequence=False, end_of_sequence=(v == 1)))

    if not FLAGS.reverse:
        print("streaming : non-streaming")
    else:
        print("non-streaming : streaming")

    seq0_expected = 0
    seq1_expected = 100

    for i in range(len(result0_list)):
        print("[" + str(i) + "] " +
              str(result0_list[i]['OUTPUT'][0][0]) + " : " +
              str(result1_list[i]['OUTPUT'][0][0]))

        if ((seq0_expected != result0_list[i]['OUTPUT'][0][0]) or
            (seq1_expected != result1_list[i]['OUTPUT'][0][0])):
            print("[ expected ] " + str(seq0_expected) + " : " + str(seq1_expected))
            sys.exit(1)

        if i < len(values):
            seq0_expected += values[i]
            seq1_expected -= values[i]
