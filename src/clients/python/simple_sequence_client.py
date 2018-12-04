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


def send(ctx, control, value):
    # Create the tensor for CONTROL and INPUT values.
    control_data = np.full(shape=[1], fill_value=control, dtype=np.int32)
    value_data = np.full(shape=[1], fill_value=value, dtype=np.int32)

    result = ctx.run({ 'CONTROL' : (control_data,),
                       'INPUT' : (value_data,) },
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     1)
    return result

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
    correlation_id0 = 1
    ctx0 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id0, verbose=FLAGS.verbose)

    correlation_id1 = 2
    ctx1 = InferContext(FLAGS.url, protocol, model_name, model_version,
                        correlation_id=correlation_id1, verbose=FLAGS.verbose)

    # Now send the inference sequences.. FIXME, for now must send the
    # proper control values since TRTIS is not yet doing it.
    #
    # First reset accumulator for both sequences.
    result0 = send(ctx0, control=1, value=0);
    result1 = send(ctx1, control=1, value=100);

    # Now send a sequence of values...
    for v in (11, 7, 5, 3, 1, 0):
        result0 = send(ctx0, control=0, value=v);
        result1 = send(ctx1, control=0, value=-v);
        print("sequence0 = " + str(result0['OUTPUT'][0]))
        print("sequence1 = " + str(result1['OUTPUT'][0]))
