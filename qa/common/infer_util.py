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

import numpy as np
from tensorrtserver.api import *
import test_util as tu

_last_request_id = 0

def _range_repr_dtype(dtype):
    if dtype == np.float64:
        return np.int32
    elif dtype == np.float32:
        return np.int16
    elif dtype == np.float16:
        return np.int8
    elif dtype == np.object:  # TYPE_STRING
        return np.int32
    return dtype

def infer_exact(tester, pf, tensor_shape, batch_size, req_raw,
                input_dtype, output0_dtype, output1_dtype,
                model_version=None, swap=False,
                outputs=("OUTPUT0", "OUTPUT1"), use_http=True, use_grpc=True,
                skip_request_id_check=False, send_input_shape=False):
    tester.assertTrue(use_http or use_grpc)
    protocols = []
    if use_http:
        protocols.append(("localhost:8000", ProtocolType.HTTP))
    if use_grpc:
        protocols.append(("localhost:8001", ProtocolType.GRPC))

    for pair in protocols:
        model_name = tu.get_model_name(pf, input_dtype, output0_dtype, output1_dtype)

        # outputs are sum and difference of inputs so set max input
        # values so that they will not overflow the output. This
        # allows us to do an exact match. For float types use 8, 16,
        # 32 int range for fp 16, 32, 64 respectively. When getting
        # class outputs the result value/probability is returned as a
        # float so must use fp32 range in that case.
        rinput_dtype = _range_repr_dtype(input_dtype)
        routput0_dtype = _range_repr_dtype(output0_dtype if req_raw else np.float32)
        routput1_dtype = _range_repr_dtype(output1_dtype if req_raw else np.float32)
        val_min = max(np.iinfo(rinput_dtype).min,
                    np.iinfo(routput0_dtype).min,
                    np.iinfo(routput1_dtype).min) / 2
        val_max = min(np.iinfo(rinput_dtype).max,
                    np.iinfo(routput0_dtype).max,
                    np.iinfo(routput1_dtype).max) / 2

        num_classes = 3

        input0_list = list()
        input1_list = list()
        expected0_list = list()
        expected1_list = list()
        expected0_val_list = list()
        expected1_val_list = list()
        for b in range(batch_size):
            in0 = np.random.randint(low=val_min, high=val_max,
                                    size=tensor_shape, dtype=rinput_dtype)
            in1 = np.random.randint(low=val_min, high=val_max,
                                    size=tensor_shape, dtype=rinput_dtype)
            if input_dtype != np.object:
                in0 = in0.astype(input_dtype)
                in1 = in1.astype(input_dtype)

            if not swap:
                op0 = in0 + in1
                op1 = in0 - in1
            else:
                op0 = in0 - in1
                op1 = in0 + in1

            expected0_val_list.append(op0)
            expected1_val_list.append(op1)
            if output0_dtype == np.object:
                expected0_list.append(np.array([bytes(str(x), encoding='utf-8') for x in (op0)], dtype=object))
            else:
                expected0_list.append(op0)
            if output1_dtype == np.object:
                expected1_list.append(np.array([bytes(str(x), encoding='utf-8') for x in (op1)], dtype=object))
            else:
                expected1_list.append(op1)

            if input_dtype == np.object:
                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                in0 = in0n.reshape(in0.shape)
                in1n = np.array([str(x) for x in in1.reshape(in1.size)], dtype=object)
                in1 = in1n.reshape(in1.shape)

            input0_list.append(in0)
            input1_list.append(in1)

        expected0_sort_idx = [ np.flip(np.argsort(x.flatten()), 0) for x in expected0_val_list ]
        expected1_sort_idx = [ np.flip(np.argsort(x.flatten()), 0) for x in expected1_val_list ]

        output_req = {}
        for o in outputs:
            if req_raw:
                output_req[o] = InferContext.ResultFormat.RAW
            else:
                output_req[o] = (InferContext.ResultFormat.CLASS, num_classes)

        ctx = InferContext(pair[0], pair[1], model_name, model_version, True)
        results = ctx.run(
            { "INPUT0" : input0_list, "INPUT1" : input1_list },
            output_req, batch_size,
            { "INPUT0" : tensor_shape, "INPUT1" : tensor_shape } if (send_input_shape) else None)

        if not skip_request_id_check:
            global _last_request_id
            min_request_id = _last_request_id + 1
            request_id = ctx.get_last_request_id()
            _last_request_id = request_id
            tester.assertGreaterEqual(request_id, min_request_id)

        tester.assertEqual(ctx.get_last_request_model_name(), model_name)
        if model_version is not None:
            tester.assertEqual(ctx.get_last_request_model_version(), model_version)

        tester.assertEqual(len(results), len(outputs))
        for (result_name, result_val) in iteritems(results):
            for b in range(batch_size):
                if req_raw:
                    if result_name == "OUTPUT0":
                        tester.assertTrue(np.array_equal(result_val[b], expected0_list[b]),
                                        "{}, OUTPUT0 expected: {}, got {}".format(
                                            model_name, expected0_list[b], result_val[b]))
                    elif result_name == "OUTPUT1":
                        tester.assertTrue(np.array_equal(result_val[b], expected1_list[b]),
                                        "{}, OUTPUT1 expected: {}, got {}".format(
                                            model_name, expected1_list[b], result_val[b]))
                    else:
                        tester.assertTrue(False, "unexpected raw result {}".format(result_name))
                else:
                    # num_classes values must be returned and must
                    # match expected top values
                    class_list = result_val[b]
                    tester.assertEqual(len(class_list), num_classes)
                    for idx, ctuple in enumerate(class_list):
                        if result_name == "OUTPUT0":
                            # can't compare indices since could have
                            # different indices with the same
                            # value/prob, so compare that the value of
                            # each index equals the expected
                            # value. Can only compare labels when the
                            # indices are equal.
                            tester.assertEqual(ctuple[1], expected0_list[b][ctuple[0]])
                            tester.assertEqual(ctuple[1], expected0_list[b][expected0_sort_idx[b][idx]])
                            if ctuple[0] == expected0_sort_idx[b][idx]:
                                tester.assertEqual(ctuple[2], 'label{}'.format(expected0_sort_idx[b][idx]))
                        elif result_name == "OUTPUT1":
                            tester.assertEqual(ctuple[1], expected1_list[b][ctuple[0]])
                            tester.assertEqual(ctuple[1], expected1_list[b][expected1_sort_idx[b][idx]])
                        else:
                            tester.assertTrue(False, "unexpected class result {}".format(result_name))
    return results
