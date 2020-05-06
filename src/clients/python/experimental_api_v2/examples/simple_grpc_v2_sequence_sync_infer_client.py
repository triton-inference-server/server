#!/usr/bin/env python
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
import sys
import queue

import tritongrpcclient
from tritonclientutils.utils import InferenceServerException

FLAGS = None


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def sync_send(triton_client, result_list, values, batch_size, sequence_id,
              model_name, model_version):

    count = 1
    for value in values:
        # Create the tensor for INPUT
        value_data = np.full(shape=[batch_size, 1],
                             fill_value=value,
                             dtype=np.int32)
        inputs = []
        inputs.append(tritongrpcclient.InferInput('INPUT', value_data.shape, "INT32"))
        # Initialize the data
        inputs[0].set_data_from_numpy(value_data)
        outputs = []
        outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT'))
        # Issue the synchronous sequence inference.
        result = triton_client.infer(model_name=model_name,
                                     inputs=inputs,
                                     outputs=outputs,
                                     sequence_id=sequence_id,
                                     sequence_start=(count == 1),
                                     sequence_end=(count == len(values)))
        result_list.append(result.as_numpy('OUTPUT'))
        count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.'
    )
    parser.add_argument('-d',
                        '--dyna',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Assume dynamic sequence model')
    parser.add_argument('-o',
                        '--offset',
                        type=int,
                        required=False,
                        default=0,
                        help='Add offset to sequence ID used')

    FLAGS = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # We use the custom "sequence" model which takes 1 input
    # value. The output is the accumulated value of the inputs. See
    # src/custom/sequence.
    model_name = "simple_sequence"
    model_version = ""
    batch_size = 1

    values = [11, 7, 5, 3, 2, 0, 1]

    # Will use two sequences and send them synchronously. Note the
    # sequence IDs should be non-zero because zero is reserved for
    # non-sequence requests.
    sequence_id0 = 1000 + FLAGS.offset * 2
    sequence_id1 = 1001 + FLAGS.offset * 2

    result0_list = []
    result1_list = []

    user_data = UserData()

    try:
        sync_send(triton_client, result0_list, [0] + values, batch_size,
                  sequence_id0, model_name, model_version)
        sync_send(triton_client, result1_list,
                  [100] + [-1 * val for val in values], batch_size,
                  sequence_id1, model_name, model_version)
    except InferenceServerException as error:
        print(error)
        sys.exit(1)

    seq0_expected = 0
    seq1_expected = 100

    for i in range(len(result0_list)):
        print("[" + str(i) + "] " + str(result0_list[i][0][0]) + " : " +
              str(result1_list[i][0][0]))

        if ((seq0_expected != result0_list[i][0][0]) or
            (seq1_expected != result1_list[i][0][0])):
            print("[ expected ] " + str(seq0_expected) + " : " +
                  str(seq1_expected))
            sys.exit(1)

        if i < len(values):
            seq0_expected += values[i]
            seq1_expected -= values[i]

            # The dyna_sequence custom backend adds the correlation ID
            # to the last request in a sequence.
            if FLAGS.dyna and (values[i] == 1):
                seq0_expected += sequence_id0
                seq1_expected += sequence_id0

    print("PASS: Sequence")
