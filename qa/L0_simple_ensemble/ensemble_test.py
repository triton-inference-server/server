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

import sys
sys.path.append("../common")
sys.path.append("../clients")

import logging

from builtins import range
from future.utils import iteritems
import os
import unittest
from PIL import Image
import numpy as np
import infer_util as iu
import test_util as tu
import image_client as ic
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

# specialiation of code from infer_util, should abstract this part as
# helper function infer_util and generalize it.
def create_subaddsub_input_outputs(input_counts, tensor_shape, input_dtype,
                                output0_dtype, output1_dtype,
                                output0_raw=True, output1_raw=True):
    # outputs are sum and difference of inputs so set max input
    # values so that they will not overflow the output. This
    # allows us to do an exact match. For float types use 8, 16,
    # 32 int range for fp 16, 32, 64 respectively. When getting
    # class outputs the result value/probability is returned as a
    # float so must use fp32 range in that case.
    rinput_dtype = iu._range_repr_dtype(input_dtype)
    routput0_dtype = iu._range_repr_dtype(output0_dtype if output0_raw else np.float32)
    routput1_dtype = iu._range_repr_dtype(output1_dtype if output1_raw else np.float32)
    val_min = max(np.iinfo(rinput_dtype).min,
                np.iinfo(routput0_dtype).min,
                np.iinfo(routput1_dtype).min) / 3
    val_max = min(np.iinfo(rinput_dtype).max,
                np.iinfo(routput0_dtype).max,
                np.iinfo(routput1_dtype).max) / 3

    input0_list = list()
    input1_list = list()
    input2_list = list()
    expected0_list = list()
    expected1_list = list()
    expected0_val_list = list()
    expected1_val_list = list()
    for i in range(input_counts):
        in0 = np.random.randint(low=val_max*2, high=val_max*3,
                                size=tensor_shape, dtype=rinput_dtype)
        in1 = np.random.randint(low=val_min, high=val_max,
                                size=tensor_shape, dtype=rinput_dtype)
        in2 = np.random.randint(low=val_min, high=val_max,
                                size=tensor_shape, dtype=rinput_dtype)
        if input_dtype != np.object:
            in0 = in0.astype(input_dtype)
            in1 = in1.astype(input_dtype)
            in2 = in2.astype(input_dtype)

        op0 = in0 - in1 + in2
        op1 = in0 - in1 - in2

        expected0_val_list.append(op0)
        expected1_val_list.append(op1)
        if output0_dtype == np.object:
            expected0_list.append(np.array([bytes(str(x), encoding='utf-8')
                                            for x in (op0.flatten())],
                                            dtype=object).reshape(op1.shape))
        else:
            expected0_list.append(op0)
        if output1_dtype == np.object:
            expected1_list.append(np.array([bytes(str(x), encoding='utf-8')
                                            for x in (op1.flatten())],
                                            dtype=object).reshape(op1.shape))
        else:
            expected1_list.append(op1)

        if input_dtype == np.object:
            in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
            in0 = in0n.reshape(in0.shape)
            in1n = np.array([str(x) for x in in1.reshape(in1.size)], dtype=object)
            in1 = in1n.reshape(in1.shape)
            in2n = np.array([str(x) for x in in2.reshape(in2.size)], dtype=object)
            in2 = in2n.reshape(in2.shape)

        input0_list.append(in0)
        input1_list.append(in1)
        input2_list.append(in2)
    return input0_list, input1_list, input2_list, expected0_list, expected1_list, \
            expected0_val_list, expected1_val_list

class EnsembleTest(unittest.TestCase):
    # prepare image inputs, addsub inputs and expected addsub outputs
    def setUp(self):
        self.logger = None
        self.model_name = 'ensemble_sub_addsub'
        self.input0_list, self.input1_list, self.input2_list, self.expected0_list, \
        self.expected1_list, self.expected0_val_list, self.expected1_val_list \
            = create_subaddsub_input_outputs(50, (16,), np.int32, np.int32, np.int32)
        self.output_req = {
            "OUTPUT0" : InferContext.ResultFormat.RAW,
            "OUTPUT1" : InferContext.ResultFormat.RAW
        }

    def test_sub_addsub(self):
        self.logger = logging.getLogger("EnsembleTest.test_sub_addsub")
        self.logger.setLevel(logging.DEBUG)
        self._run_sub_addsub("localhost:8000", ProtocolType.HTTP, False)

    def _run_sub_addsub(self, url, protocol, streaming):
        try:
            ctx = InferContext(url, protocol, self.model_name,
                            model_version=None, verbose=False, streaming=streaming)
            
            # Send a series of requests
            req_id_idx = {}
            req_ids = []
            id_results = []
            for idx in range(len(self.input0_list)):
                req_ids.append(ctx.async_run(
                                {"INPUT0" : [self.input0_list[idx]],
                                "INPUT1" : [self.input1_list[idx]],
                                "INPUT2" : [self.input2_list[idx]]},
                                self.output_req, 1))
                req_id_idx[req_ids[-1]] = idx
            # Always get the first completed request
            # Repeat until all responses are received
            while len(id_results) != len(req_id_idx):
                ready_id = None
                ready_id = ctx.get_ready_async_request(False)
                if ready_id is not None:
                    id_results.append((ready_id, ctx.get_async_run_results(ready_id, True)))

            # evaluate results
            self.assertEqual(len(req_id_idx), len(id_results),
                            "Unexpected response number not equals to request number")

            for pair in id_results:
                idx = req_id_idx[pair[0]]
                self.assertTrue(np.array_equal(pair[1]["OUTPUT0"][0], self.expected0_list[idx]),
                                "{}, OUTPUT0 expected: {}, got {}".format(
                                    self.model_name, self.expected0_list[idx],
                                    pair[1]["OUTPUT0"][0]))
                self.assertTrue(np.array_equal(pair[1]["OUTPUT1"][0], self.expected1_list[idx]),
                                "{}, OUTPUT1 expected: {}, got {}".format(
                                    self.model_name, self.expected1_list[idx],
                                    pair[1]["OUTPUT1"][0]))

        except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    unittest.main()
