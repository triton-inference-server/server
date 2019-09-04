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

import sys
import numpy as np
from tensorrtserver.api import *
import tensorrtserver.shared_memory as shm
import test_util as tu
from sets import Set
from ctypes import *

# unicode() doesn't exist on python3, for how we use it the
# corresponding function is bytes()
if sys.version_info.major == 3:
    unicode = bytes

_seen_request_ids = Set()

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

# Perform inference using an "addsum" type verification backend.
def infer_exact(tester, pf, tensor_shape, batch_size,
                input_dtype, output0_dtype, output1_dtype,
                output0_raw=True, output1_raw=True,
                model_version=None, swap=False,
                outputs=("OUTPUT0", "OUTPUT1"), use_http=True, use_grpc=True,
                skip_request_id_check=False, use_streaming=True,
                correlation_id=0, include_shared_memory_test=False):
    tester.assertTrue(use_http or use_grpc or use_streaming)
    configs = []
    if use_http:
        if include_shared_memory_test:
            configs.append(("localhost:8000", ProtocolType.HTTP, False, True))
        configs.append(("localhost:8000", ProtocolType.HTTP, False, False))
    if use_grpc:
        if include_shared_memory_test:
            configs.append(("localhost:8001", ProtocolType.GRPC, False, True))
        configs.append(("localhost:8001", ProtocolType.GRPC, False, False))
    if use_streaming:
        if include_shared_memory_test:
            configs.append(("localhost:8001", ProtocolType.GRPC, True, True))
        configs.append(("localhost:8001", ProtocolType.GRPC, True, False))

    for config in configs:
        model_name = tu.get_model_name(pf, input_dtype, output0_dtype, output1_dtype)

        # outputs are sum and difference of inputs so set max input
        # values so that they will not overflow the output. This
        # allows us to do an exact match. For float types use 8, 16,
        # 32 int range for fp 16, 32, 64 respectively. When getting
        # class outputs the result value/probability is returned as a
        # float so must use fp32 range in that case.
        rinput_dtype = _range_repr_dtype(input_dtype)
        routput0_dtype = _range_repr_dtype(output0_dtype if output0_raw else np.float32)
        routput1_dtype = _range_repr_dtype(output1_dtype if output1_raw else np.float32)
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
                expected0_list.append(np.array([unicode(str(x), encoding='utf-8')
                                                for x in (op0.flatten())], dtype=object).reshape(op1.shape))
            else:
                expected0_list.append(op0)
            if output1_dtype == np.object:
                expected1_list.append(np.array([unicode(str(x), encoding='utf-8')
                                                for x in (op1.flatten())], dtype=object).reshape(op1.shape))
            else:
                expected1_list.append(op1)

            if input_dtype == np.object:
                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                in0 = in0n.reshape(in0.shape)
                in1n = np.array([str(x) for x in in1.reshape(in1.size)], dtype=object)
                in1 = in1n.reshape(in1.shape)

            input0_list.append(in0)
            input1_list.append(in1)

        if config[3]:
            input0_byte_size = input0_list[0].nbytes * batch_size
            output0_byte_size = expected0_val_list[0].nbytes * batch_size
            output1_byte_size = expected1_val_list[0].nbytes * batch_size
            # create and register shared memory region for inputs and outputs
            shm_ip0_handle = shm.create_shared_memory_region("input0_data", "/input0", input0_byte_size)
            shm_ip1_handle = shm.create_shared_memory_region("input1_data", "/input1", input0_byte_size)
            shm_op0_handle = shm.create_shared_memory_region("output0_data", "/output0", output0_byte_size)
            shm_op1_handle = shm.create_shared_memory_region("output1_data", "/output1", output1_byte_size)
            # copy data into shared memory region for input values
            shm.set_shared_memory_region(shm_ip0_handle, input0_list)
            shm.set_shared_memory_region(shm_ip1_handle, input1_list)

            shared_memory_ctx = SharedMemoryControlContext(config[0], config[1], verbose=True)
            shared_memory_ctx.register(shm_ip0_handle)
            shared_memory_ctx.register(shm_ip1_handle)
            shared_memory_ctx.register(shm_op0_handle)
            shared_memory_ctx.register(shm_op1_handle)

        expected0_sort_idx = [ np.flip(np.argsort(x.flatten()), 0) for x in expected0_val_list ]
        expected1_sort_idx = [ np.flip(np.argsort(x.flatten()), 0) for x in expected1_val_list ]

        output_req = {}
        OUTPUT0 = "OUTPUT0"
        OUTPUT1 = "OUTPUT1"
        INPUT0 = "INPUT0"
        INPUT1 = "INPUT1"
        if pf == "libtorch" or pf == "libtorch_nobatch":
            OUTPUT0 = "OUTPUT__0"
            OUTPUT1 = "OUTPUT__1"
            INPUT0 = "INPUT__0"
            INPUT1 = "INPUT__1"
        if "OUTPUT0" in outputs:
            if config[3]:
                output_req[OUTPUT0] = (InferContext.ResultFormat.RAW, shm_op0_handle)
            else:
                if output0_raw:
                    output_req[OUTPUT0] = InferContext.ResultFormat.RAW
                else:
                    output_req[OUTPUT0] = (InferContext.ResultFormat.CLASS, num_classes)
        if "OUTPUT1" in outputs:
            if config[3]:
                output_req[OUTPUT1] = (InferContext.ResultFormat.RAW, shm_op1_handle)
            else:
                if output1_raw:
                    output_req[OUTPUT1] = InferContext.ResultFormat.RAW
                else:
                    output_req[OUTPUT1] = (InferContext.ResultFormat.CLASS, num_classes)

        ctx = InferContext(config[0], config[1], model_name, model_version,
                           correlation_id=correlation_id, streaming=config[2],
                           verbose=True)
        if config[3]:
            results = ctx.run(
                    { INPUT0 : shm_ip0_handle, INPUT1 : shm_ip1_handle },
                    output_req, batch_size)
        else:
            results = ctx.run(
                    { INPUT0 : input0_list, INPUT1 : input1_list },
                    output_req, batch_size)

        if not skip_request_id_check:
            global _seen_request_ids
            request_id = ctx.get_last_request_id()
            tester.assertFalse(request_id in _seen_request_ids)
            _seen_request_ids.add(request_id)

        tester.assertEqual(ctx.get_last_request_model_name(), model_name)
        if model_version is not None:
            tester.assertEqual(ctx.get_last_request_model_version(), model_version)

        tester.assertEqual(len(results), len(outputs))
        for (result_name, result_val) in iteritems(results):
            for b in range(batch_size):
                if ((result_name == OUTPUT0 and output0_raw) or
                    (result_name == OUTPUT1 and output1_raw)):
                    if result_name == OUTPUT0:
                        tester.assertTrue(np.array_equal(result_val[b], expected0_list[b]),
                                        "{}, "+OUTPUT0+" expected: {}, got {}".format(
                                            model_name, expected0_list[b], result_val[b]))
                    elif result_name == OUTPUT1:
                        tester.assertTrue(np.array_equal(result_val[b], expected1_list[b]),
                                        "{}, "+OUTPUT1+" expected: {}, got {}".format(
                                            model_name, expected1_list[b], result_val[b]))
                    else:
                        tester.assertTrue(False, "unexpected raw result {}".format(result_name))
                else:
                    # num_classes values must be returned and must
                    # match expected top values
                    class_list = result_val[b]
                    tester.assertEqual(len(class_list), num_classes)

                    expected0_flatten = expected0_list[b].flatten()
                    expected1_flatten = expected1_list[b].flatten()

                    for idx, ctuple in enumerate(class_list):
                        if result_name == OUTPUT0:
                            # can't compare indices since could have
                            # different indices with the same
                            # value/prob, so compare that the value of
                            # each index equals the expected
                            # value. Can only compare labels when the
                            # indices are equal.
                            tester.assertEqual(ctuple[1], expected0_flatten[ctuple[0]])
                            tester.assertEqual(ctuple[1], expected0_flatten[expected0_sort_idx[b][idx]])
                            if ctuple[0] == expected0_sort_idx[b][idx]:
                                tester.assertEqual(ctuple[2], 'label{}'.format(expected0_sort_idx[b][idx]))
                        elif result_name == OUTPUT1:
                            tester.assertEqual(ctuple[1], expected1_flatten[ctuple[0]])
                            tester.assertEqual(ctuple[1], expected1_flatten[expected1_sort_idx[b][idx]])
                        else:
                            tester.assertTrue(False, "unexpected class result {}".format(result_name))

        if config[3]:
            shared_memory_ctx.unregister(shm_ip0_handle)
            shm.destroy_shared_memory_region(shm_ip0_handle)
            shared_memory_ctx.unregister(shm_ip1_handle)
            shm.destroy_shared_memory_region(shm_ip1_handle)
            shared_memory_ctx.unregister(shm_op0_handle)
            shm.destroy_shared_memory_region(shm_op0_handle)
            shared_memory_ctx.unregister(shm_op1_handle)
            shm.destroy_shared_memory_region(shm_op1_handle)

    return results


# Perform inference using a "nop" model that expects some form or
# zero-sized input/output tensor.
def infer_zero(tester, pf, batch_size, tensor_dtype, input_shapes, output_shapes,
               model_version=None, use_http=True, use_grpc=True,
               use_streaming=True, include_shared_memory_test=False):
    tester.assertTrue(use_http or use_grpc or use_streaming)
    configs = []
    if use_http:
        if include_shared_memory_test:
            configs.append(("localhost:8000", ProtocolType.HTTP, False, True))
        configs.append(("localhost:8000", ProtocolType.HTTP, False, False))
    if use_grpc:
        if include_shared_memory_test:
            configs.append(("localhost:8001", ProtocolType.GRPC, False, True))
        configs.append(("localhost:8001", ProtocolType.GRPC, False, False))
    if use_streaming:
        if include_shared_memory_test:
            configs.append(("localhost:8001", ProtocolType.GRPC, True, True))
        configs.append(("localhost:8001", ProtocolType.GRPC, True, False))
    tester.assertEqual(len(input_shapes), len(output_shapes))
    io_cnt = len(input_shapes)

    for config in configs:
        model_name = tu.get_zero_model_name(pf, io_cnt, tensor_dtype)
        input_dict = {}
        output_dict = {}
        expected_dict = {}

        if config[3]:
            # create and register shared memory region for inputs and outputs
            shm_ip_handles = list()
            shm_op_handles = list()
            shared_memory_ctx = SharedMemoryControlContext(config[0], config[1], verbose=True)
            for io_num in range(io_cnt):
                input0_byte_size = shape_element_count(input_shapes[io_num]) *\
                                    np.dtype(tensor_dtype).itemsize * batch_size
                output0_byte_size = shape_element_count(output_shapes[io_num]) *\
                                    np.dtype(tensor_dtype).itemsize * batch_size
                shm_ip_handles.append(shm.create_shared_memory_region("input"+str(io_num)+"_data",\
                                            "/input"+str(io_num), input0_byte_size))
                shm_op_handles.append(shm.create_shared_memory_region("output"+str(io_num)+"_data",\
                                            "/output"+str(io_num), output0_byte_size))

                shm.register(shm_ip_handles[io_num])
                shm.register(shm_op_handles[io_num])

            offset_input = 0
            offset_output = 0

        for io_num in range(io_cnt):
            if pf == "libtorch" or pf == "libtorch_nobatch":
                input_name = "INPUT__{}".format(io_num)
                output_name = "OUTPUT__{}".format(io_num)
            else:
                input_name = "INPUT{}".format(io_num)
                output_name = "OUTPUT{}".format(io_num)

            input_list = list()
            expected_list = list()
            for b in range(batch_size):
                rtensor_dtype = _range_repr_dtype(tensor_dtype)
                in0 = np.random.randint(low=np.iinfo(rtensor_dtype).min,
                                        high=np.iinfo(rtensor_dtype).max,
                                        size=input_shapes[io_num], dtype=rtensor_dtype)
                if tensor_dtype != np.object:
                    in0 = in0.astype(tensor_dtype)
                    expected0 = np.ndarray.copy(in0)
                else:
                    expected0 = np.array([unicode(str(x), encoding='utf-8')
                                    for x in in0.flatten()], dtype=object)
                    in0 = np.array([str(x) for x in in0.flatten()],
                                   dtype=object).reshape(in0.shape)

                expected0 = expected0.reshape(output_shapes[io_num])

                input_list.append(in0)
                expected_list.append(expected0)

            expected_dict[output_name] = expected_list
            if config[3]:
                # copy data into shared memory region for input values
                shm.set_shared_memory_region(shm_ip_handles[io_num], input_list)
                input_dict[input_name] = shm_ip_handles[io_num]
                output_dict[output_name] = (InferContext.ResultFormat.RAW, shm_op_handles[io_num])
            else:
                input_dict[input_name] = input_list
                output_dict[output_name] = InferContext.ResultFormat.RAW

        ctx = InferContext(config[0], config[1], model_name, model_version,
                           correlation_id=0, streaming=config[2],
                           verbose=True)
        results = ctx.run(input_dict, output_dict, batch_size)

        tester.assertEqual(ctx.get_last_request_model_name(), model_name)
        if model_version is not None:
            tester.assertEqual(ctx.get_last_request_model_version(), model_version)

        tester.assertEqual(len(results), io_cnt)
        for (result_name, result_val) in iteritems(results):
            tester.assertTrue(result_name in output_dict)
            tester.assertTrue(result_name in expected_dict)
            for b in range(batch_size):
                expected = expected_dict[result_name][b]
                tester.assertEqual(result_val[b].shape, expected.shape)
                tester.assertTrue(np.array_equal(result_val[b], expected),
                                  "{}, {}, slot {}, expected: {}, got {}".format(
                                      model_name, result_name, b, expected, result_val[b]))
        if config[3]:
            for io_num in range(io_cnt):
                shared_memory_ctx.unregister(shm_ip_handles[io_num])
                shm.destroy_shared_memory_region(shm_ip_handles[io_num])
                shared_memory_ctx.unregister(shm_op_handles[io_num])
                shm.destroy_shared_memory_region(shm_op_handles[io_num])

    return results
