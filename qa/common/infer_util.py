# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
import os
import numpy as np
import tritongrpcclient.core as grpcclient
import tritonhttpclient.core as httpclient
from tritonhttpclient.utils import *
import tritonsharedmemoryutils.shared_memory as shm
import tritonsharedmemoryutils.cuda_shared_memory as cudashm
import test_util as tu
import shm_util as su

# unicode() doesn't exist on python3, for how we use it the
# corresponding function is bytes()
if sys.version_info.major == 3:
    unicode = bytes

_seen_request_ids = set()


def _unique_request_id():
    if len(_seen_request_ids) == 0:
        return 1
    else:
        return max(_seen_request_ids) + 1


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


def _prepend_string_size(input_values):
    input_list = []
    for input_value in input_values:
        input_list.append(serialize_byte_tensor(input_value))
    return input_list

# Perform inference using an "addsum" type verification backend.


def infer_exact(tester, pf, tensor_shape, batch_size,
                input_dtype, output0_dtype, output1_dtype,
                output0_raw=True, output1_raw=True,
                model_version=None, swap=False,
                outputs=("OUTPUT0", "OUTPUT1"), use_http=True, use_grpc=True,
                use_http_json_tensors=True, skip_request_id_check=False, use_streaming=True,
                correlation_id=0, shm_region_names=None, precreated_shm_regions=None,
                use_system_shared_memory=False, use_cuda_shared_memory=False,
                priority=0, timeout_us=0):
    tester.assertTrue(
        use_http or use_http_json_tensors or use_grpc or use_streaming)
    configs = []
    if use_http:
        configs.append(("localhost:8000", "http", False, True))
    if use_http_json_tensors and (input_dtype != np.float16):
        configs.append(("localhost:8000", "http", False, False))
    if use_grpc:
        configs.append(("localhost:8001", "grpc", False, False))
    if use_streaming:
        configs.append(("localhost:8001", "grpc", True, False))

    # outputs are sum and difference of inputs so set max input
    # values so that they will not overflow the output. This
    # allows us to do an exact match. For float types use 8, 16,
    # 32 int range for fp 16, 32, 64 respectively. When getting
    # class outputs the result value/probability is returned as a
    # float so must use fp32 range in that case.
    rinput_dtype = _range_repr_dtype(input_dtype)
    routput0_dtype = _range_repr_dtype(
        output0_dtype if output0_raw else np.float32)
    routput1_dtype = _range_repr_dtype(
        output1_dtype if output1_raw else np.float32)
    val_min = max(np.iinfo(rinput_dtype).min,
                  np.iinfo(routput0_dtype).min,
                  np.iinfo(routput1_dtype).min) / 2
    val_max = min(np.iinfo(rinput_dtype).max,
                  np.iinfo(routput0_dtype).max,
                  np.iinfo(routput1_dtype).max) / 2

    num_classes = 3

    input0_array = np.random.randint(low=val_min, high=val_max,
                                     size=tensor_shape, dtype=rinput_dtype)
    input1_array = np.random.randint(low=val_min, high=val_max,
                                     size=tensor_shape, dtype=rinput_dtype)
    if input_dtype != np.object:
        input0_array = input0_array.astype(input_dtype)
        input1_array = input1_array.astype(input_dtype)

    if not swap:
        output0_array = input0_array + input1_array
        output1_array = input0_array - input1_array
    else:
        output0_array = input0_array - input1_array
        output1_array = input0_array + input1_array

    if output0_dtype == np.object:
        output0_array = np.array([unicode(str(x), encoding='utf-8')
                                  for x in (output0_array.flatten())], dtype=object).reshape(output0_array.shape)
    else:
        output0_array = output0_array.astype(output0_dtype)
    if output1_dtype == np.object:
        output1_array = np.array([unicode(str(x), encoding='utf-8')
                                  for x in (output1_array.flatten())], dtype=object).reshape(output1_array.shape)
    else:
        output1_array = output1_array.astype(output1_dtype)

    if input_dtype == np.object:
        in0n = np.array([str(x)
                         for x in input0_array.reshape(input0_array.size)], dtype=object)
        input0_array = in0n.reshape(input0_array.shape)
        in1n = np.array([str(x)
                         for x in input1_array.reshape(input1_array.size)], dtype=object)
        input1_array = in1n.reshape(input1_array.shape)

    # prepend size of string to output string data
    if output0_dtype == np.object:
        if batch_size == 1:
            output0_array_tmp = _prepend_string_size([output0_array])
        else:
            output0_array_tmp = _prepend_string_size(output0_array)
    else:
        output0_array_tmp = output0_array

    if output1_dtype == np.object:
        if batch_size == 1:
            output1_array_tmp = _prepend_string_size([output1_array])
        else:
            output1_array_tmp = _prepend_string_size(output1_array)
    else:
        output1_array_tmp = output1_array

    OUTPUT0 = "OUTPUT0"
    OUTPUT1 = "OUTPUT1"
    INPUT0 = "INPUT0"
    INPUT1 = "INPUT1"
    if pf == "libtorch" or pf == "libtorch_nobatch":
        OUTPUT0 = "OUTPUT__0"
        OUTPUT1 = "OUTPUT__1"
        INPUT0 = "INPUT__0"
        INPUT1 = "INPUT__1"

    output0_byte_size = sum([o0.nbytes for o0 in output0_array_tmp])
    output1_byte_size = sum([o1.nbytes for o1 in output1_array_tmp])

    if batch_size == 1:
        input0_list = [input0_array]
        input1_list = [input1_array]
    else:
        input0_list = [x for x in input0_array]
        input1_list = [x for x in input1_array]

    # Create and register system/cuda shared memory regions if needed
    shm_regions, op0_handle, op1_handle = su.create_register_shm_regions(input0_list, input1_list, output0_byte_size,
                                                                         output1_byte_size, outputs, shm_region_names, precreated_shm_regions,
                                                                         use_system_shared_memory, use_cuda_shared_memory)

    # Run inference and check results for each config
    for config in configs:
        model_name = tu.get_model_name(
            pf, input_dtype, output0_dtype, output1_dtype)

        if config[1] == "http":
            triton_client = httpclient.InferenceServerClient(
                config[0], verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(
                config[0], verbose=True)

        inputs = []
        if config[1] == "http":
            inputs.append(httpclient.InferInput(
                INPUT0, tensor_shape, np_to_triton_dtype(input_dtype)))
            inputs.append(httpclient.InferInput(
                INPUT1, tensor_shape, np_to_triton_dtype(input_dtype)))
        else:
            inputs.append(grpcclient.InferInput(
                INPUT0, tensor_shape, np_to_triton_dtype(input_dtype)))
            inputs.append(grpcclient.InferInput(
                INPUT1, tensor_shape, np_to_triton_dtype(input_dtype)))

        if not (use_cuda_shared_memory or use_system_shared_memory):
            if config[1] == "http":
                inputs[0].set_data_from_numpy(
                    input0_array, binary_data=config[3])
                inputs[1].set_data_from_numpy(
                    input1_array, binary_data=config[3])
            else:
                inputs[0].set_data_from_numpy(input0_array)
                inputs[1].set_data_from_numpy(input1_array)
        else:
            su.set_shm_regions(inputs, shm_regions, use_system_shared_memory,
                               use_cuda_shared_memory, input0_list, input1_list)

        if batch_size == 1:
            expected0_sort_idx = [np.flip(np.argsort(x.flatten()), 0)
                                  for x in output0_array.reshape((1,) + tensor_shape)]
            expected1_sort_idx = [np.flip(np.argsort(x.flatten()), 0)
                                  for x in output1_array.reshape((1,) + tensor_shape)]
        else:
            expected0_sort_idx = [np.flip(np.argsort(x.flatten()), 0)
                                  for x in output0_array.reshape(tensor_shape)]
            expected1_sort_idx = [np.flip(np.argsort(x.flatten()), 0)
                                  for x in output1_array.reshape(tensor_shape)]

        # Force binary_data = False for shared memory and class
        output_req = []
        i = 0
        if "OUTPUT0" in outputs:
            if len(shm_regions) != 0:
                if config[1] == "http":
                    output_req.append(httpclient.InferRequestedOutput(
                        OUTPUT0, binary_data=False))
                else:
                    output_req.append(grpcclient.InferRequestedOutput(OUTPUT0))

                if precreated_shm_regions is None:
                    output_req[-1].set_shared_memory(
                        shm_regions[2]+'_data', output0_byte_size)
                else:
                    output_req[-1].set_shared_memory(
                        precreated_shm_regions[0], output0_byte_size)
            else:
                if output0_raw:
                    if config[1] == "http":
                        output_req.append(httpclient.InferRequestedOutput(
                            OUTPUT0, binary_data=config[3]))
                    else:
                        output_req.append(
                            grpcclient.InferRequestedOutput(OUTPUT0))
                else:
                    if config[1] == "http":
                        output_req.append(httpclient.InferRequestedOutput(
                            OUTPUT0, binary_data=False, class_count=num_classes))
                    else:
                        output_req.append(grpcclient.InferRequestedOutput(
                            OUTPUT0, class_count=num_classes))
            i += 1
        if "OUTPUT1" in outputs:
            if len(shm_regions) != 0:
                if config[1] == "http":
                    output_req.append(httpclient.InferRequestedOutput(
                        OUTPUT1, binary_data=False))
                else:
                    output_req.append(grpcclient.InferRequestedOutput(OUTPUT1))

                if precreated_shm_regions is None:
                    output_req[-1].set_shared_memory(
                        shm_regions[2+i]+'_data', output1_byte_size)
                else:
                    output_req[-1].set_shared_memory(
                        precreated_shm_regions[i], output1_byte_size)
            else:
                if output1_raw:
                    if config[1] == "http":
                        output_req.append(httpclient.InferRequestedOutput(
                            OUTPUT1, binary_data=config[3]))
                    else:
                        output_req.append(
                            grpcclient.InferRequestedOutput(OUTPUT1))
                else:
                    if config[1] == "http":
                        output_req.append(httpclient.InferRequestedOutput(
                            OUTPUT1, binary_data=False, class_count=num_classes))
                    else:
                        output_req.append(grpcclient.InferRequestedOutput(
                            OUTPUT1, class_count=num_classes))

        if model_version is not None:
            model_version = str(model_version)
        else:
            model_version = ""

        if config[2]:
            # TODO fix for streaming case
            continue
            # results = triton_client.async_stream_infer(model_name,
            #                                  inputs,
            #                                  model_version=model_version,
            #                                  stream=stream,
            #                                  outputs=output_req)
        else:
            results = triton_client.infer(model_name,
                                          inputs,
                                          model_version=model_version,
                                          outputs=output_req,
                                          request_id=str(_unique_request_id()))

        last_response = results.get_response()
        if config[1] == "http":
            if 'error' in last_response:
                raise InferenceServerException(msg=last_response['error'])

        if not skip_request_id_check:
            global _seen_request_ids
            if config[1] == "http":
                request_id = int(last_response["id"])
            else:
                request_id = int(last_response.id)
            tester.assertFalse(request_id in _seen_request_ids,
                               "request_id: {}".format(request_id))
            _seen_request_ids.add(request_id)

        if config[1] == "http":
            response_model_name = last_response["model_name"]
        else:
            response_model_name = last_response.model_name
        tester.assertEqual(response_model_name, model_name)

        if model_version != "":
            if config[1] == "http":
                response_model_version = last_response["model_version"]
            else:
                response_model_version = last_response.model_version
            tester.assertEqual(response_model_version, model_version)

        if config[1] == "http":
            response_outputs = last_response["outputs"]
        else:
            response_outputs = last_response.outputs
        tester.assertEqual(len(response_outputs), len(outputs))

        for result in response_outputs:
            if config[1] == "http":
                result_name = result["name"]
            else:
                result_name = result.name

            if ((result_name == OUTPUT0 and output0_raw) or
                    (result_name == OUTPUT1 and output1_raw)):
                if result_name == OUTPUT0:
                    shm_handle = op0_handle
                else:
                    shm_handle = op1_handle

                if use_system_shared_memory or use_cuda_shared_memory:
                    output = results.get_output(result_name)
                    if config[1] == "http":
                        output_datatype = output['datatype']
                        output_shape = output['shape']
                    else:
                        output_datatype = output.datatype
                        output_shape = output.shape
                    output_dtype = triton_to_np_dtype(output_datatype)
                if use_system_shared_memory:
                    output_data = shm.get_contents_as_numpy(
                        shm_handle, output_dtype, output_shape)
                elif use_cuda_shared_memory:
                    output_data = cudashm.get_contents_as_numpy(
                        shm_handle, output_dtype, output_shape)
                else:
                    output_data = results.as_numpy(result_name)

                if result_name == OUTPUT0:
                    tester.assertTrue(np.array_equal(output_data, output0_array),
                                      "{}, {} expected: {}, got {}".format(
                        model_name, OUTPUT0, output0_array, output_data))
                elif result_name == OUTPUT1:
                    tester.assertTrue(np.array_equal(output_data, output1_array),
                                      "{}, {} expected: {}, got {}".format(
                        model_name, OUTPUT1, output1_array, output_data))
                else:
                    tester.assertTrue(
                        False, "unexpected raw result {}".format(result_name))
            else:
                for b in range(batch_size):
                    # num_classes values must be returned and must
                    # match expected top values
                    class_list = results.as_numpy(result_name)[b]
                    tester.assertEqual(len(class_list), num_classes)
                    if batch_size == 1:
                        expected0_flatten = output0_array.flatten()
                        expected1_flatten = output1_array.flatten()
                    else:
                        expected0_flatten = output0_array[b].flatten()
                        expected1_flatten = output1_array[b].flatten()

                    for idx, class_label in enumerate(class_list):
                        # can't compare indices since could have different
                        # indices with the same value/prob, so check that
                        # the value of each index equals the expected value.
                        # Only compare labels when the indices are equal.
                        ctuple = "".join(chr(x)
                                         for x in class_label).split(':')
                        cidx = int(ctuple[0])
                        cval = float(ctuple[1])
                        if result_name == OUTPUT0:
                            tester.assertEqual(cval, expected0_flatten[cidx])
                            tester.assertEqual(
                                cval, expected0_flatten[expected0_sort_idx[b][idx]])
                            if cidx == expected0_sort_idx[b][idx]:
                                tester.assertEqual(ctuple[2], 'label{}'.format(
                                    expected0_sort_idx[b][idx]))
                        elif result_name == OUTPUT1:
                            tester.assertEqual(cval, expected1_flatten[cidx])
                            tester.assertEqual(
                                cval, expected1_flatten[expected1_sort_idx[b][idx]])
                        else:
                            tester.assertTrue(
                                False, "unexpected class result {}".format(result_name))

    # Unregister system/cuda shared memory regions if they exist
    su.unregister_cleanup_shm_regions(shm_regions, precreated_shm_regions, outputs,
                                      use_system_shared_memory, use_cuda_shared_memory)

    return results


# Perform inference using a "nop" model that expects some form or
# zero-sized input/output tensor.
def infer_zero(tester, pf, batch_size, tensor_dtype, input_shapes, output_shapes,
               model_version=None, use_http=True, use_grpc=True,
               use_http_json_tensors=True, use_streaming=True, shm_region_name_prefix=None,
               use_system_shared_memory=False, use_cuda_shared_memory=False,
               priority=0, timeout_us=0):
    tester.assertTrue(use_http or use_grpc or use_http_json_tensors or use_streaming)
    configs = []
    if use_http:
        configs.append(("localhost:8000", "http", False, True))
    if use_http_json_tensors and (tensor_dtype != np.float16):
        configs.append(("localhost:8000", "http", False, False))
    if use_grpc:
        configs.append(("localhost:8001", "grpc", False, False))
    if use_streaming:
        configs.append(("localhost:8001", "grpc", True, False))
    tester.assertEqual(len(input_shapes), len(output_shapes))
    io_cnt = len(input_shapes)

    if shm_region_name_prefix is None:
        shm_region_name_prefix = ["input", "output"]

    input_dict = {}
    output_dict = {}
    expected_dict = {}
    shm_ip_handles = list()
    shm_op_handles = list()
    shm_client = httpclient.InferenceServerClient("localhost:8000")

    for io_num in range(io_cnt):
        if pf == "libtorch" or pf == "libtorch_nobatch":
            input_name = "INPUT__{}".format(io_num)
            output_name = "OUTPUT__{}".format(io_num)
        else:
            input_name = "INPUT{}".format(io_num)
            output_name = "OUTPUT{}".format(io_num)

        input_list = list()
        expected_list = list()

        input_shape = [batch_size,] + input_shapes[io_num]
        output_shape = [batch_size,] + output_shapes[io_num]

        rtensor_dtype = _range_repr_dtype(tensor_dtype)
        if (rtensor_dtype != np.bool):
            input_array = np.random.randint(low=np.iinfo(rtensor_dtype).min,
                                    high=np.iinfo(rtensor_dtype).max,
                                    size=input_shape, dtype=rtensor_dtype)
        else:
            input_array = np.random.choice(a=[False, True], size=input_shape)
        if tensor_dtype != np.object:
            input_array = input_array.astype(tensor_dtype)
            expected_array = np.ndarray.copy(input_array)
        else:
            expected_array = np.array([unicode(str(x), encoding='utf-8')
                            for x in input_array.flatten()], dtype=object)
            input_array = np.array([str(x) for x in input_array.flatten()],
                            dtype=object).reshape(input_array.shape)

        expected_array = expected_array.reshape(output_shapes[io_num])

        input_list = [x for x in input_array]
        expected_dict[output_name] = [x for x in expected_array]

        input_byte_size = tu.shape_element_count(input_shape) *\
                            np.dtype(tensor_dtype).itemsize
        output_byte_size = tu.shape_element_count(output_shape) *\
                            np.dtype(tensor_dtype).itemsize

        # create and register shared memory region for inputs and outputs
        shm_io_handles = su.create_register_set_either_shm_region([shm_region_name_prefix[0]+str(io_num),
                                                shm_region_name_prefix[1]+str(io_num)], input_array,
                                                input_byte_size, output_byte_size, shm_client,
                                                use_system_shared_memory, use_cuda_shared_memory)
        if len(shm_io_handles) != 0:
            shm_ip_handles.append(shm_io_handles[0])
            shm_op_handles.append(shm_io_handles[1])
            input_dict[input_name] = (shm_ip_handles[io_num], input_shapes)
            output_dict[output_name] = (InferContext.ResultFormat.RAW, shm_op_handles[io_num])
        else:
            input_dict[input_name] = input_list
            output_dict[output_name] = InferContext.ResultFormat.RAW

    # Run inference and check results for each config
    for config in configs:
        model_name = tu.get_zero_model_name(pf, io_cnt, tensor_dtype)

        ctx = InferContext(config[0], config[1], model_name, model_version,
                           correlation_id=0, streaming=config[2],
                           verbose=True)
        results = ctx.run(input_dict, output_dict, batch_size,
                          priority=priority, timeout_us=timeout_us)

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

    if len(shm_ip_handles) != 0:
        for io_num in range(io_cnt):
            shared_memory_ctx.unregister(shm_ip_handles[io_num])
            shared_memory_ctx.unregister(shm_op_handles[io_num])
            su.destroy_either_shm_region(shm_ip_handles[io_num], use_system_shared_memory, use_cuda_shared_memory)
            su.destroy_either_shm_region(shm_op_handles[io_num], use_system_shared_memory, use_cuda_shared_memory)

    return results


# Perform inference on a model that takes a shape and a dummy tensors as inputs,
# resize the dummy tensor with the provided values in the shape tensor and finally
# return the shape of the resized tensor.
def infer_shape_tensor(tester, pf, batch_size, tensor_dtype, input_shape_values, dummy_input_shapes,
                       model_version=None, use_http=True, use_grpc=True, use_http_json_tensors=True,
                       use_streaming=True, shm_suffix="", use_system_shared_memory=False,
                       use_cuda_shared_memory=False, priority=0, timeout_us=0):
    tester.assertTrue(
        use_http or use_http_json_tensors or use_grpc or use_streaming)
    configs = []
    if use_http:
        configs.append(("localhost:8000", "http", False, True))
    if use_http_json_tensors and (tensor_dtype != np.float16):
        configs.append(("localhost:8000", "http", False, False))
    if use_grpc:
        configs.append(("localhost:8001", "grpc", False, False))
    if use_streaming:
        configs.append(("localhost:8001", "grpc", True, False))
    tester.assertEqual(len(input_shape_values), len(dummy_input_shapes))
    io_cnt = len(input_shape_values)

    if use_system_shared_memory and use_cuda_shared_memory:
        raise ValueError(
            "Cannot set both System and CUDA shared memory flags to 1")

    input_dict = {}
    # output_dict = {}
    expected_dict = {}
    shm_ip_handles = list()
    shm_op_handles = list()

    shm_client = httpclient.InferenceServerClient("localhost:8000")

    for io_num in range(io_cnt):
        tester.assertTrue(pf == "plan" or pf == "plan_nobatch")

        input_name = "INPUT{}".format(io_num)
        output_name = "OUTPUT{}".format(io_num)
        dummy_input_name = "DUMMY_INPUT{}".format(io_num)
        dummy_output_name = "DUMMY_OUTPUT{}".format(io_num)

        input_list = list()
        dummy_input_list = list()
        expected_list = list()
        for b in range(batch_size):
            # Prepare the dummy tensor
            rtensor_dtype = _range_repr_dtype(tensor_dtype)
            if (rtensor_dtype != np.bool):
                dummy_in0 = np.random.randint(low=np.iinfo(rtensor_dtype).min,
                                              high=np.iinfo(rtensor_dtype).max,
                                              size=dummy_input_shapes[io_num], dtype=rtensor_dtype)
            else:
                dummy_in0 = np.random.choice(
                    a=[False, True], size=dummy_input_shapes[io_num])
            if tensor_dtype != np.object:
                dummy_in0 = dummy_in0.astype(tensor_dtype)
            else:
                dummy_in0 = np.array([str(x) for x in in0.flatten()],
                                     dtype=object).reshape(in0.shape)

            dummy_input_list.append(dummy_in0)

        # Prepare shape input tensor. Only one tensor per batch
        in0 = np.asarray(input_shape_values[io_num], dtype=np.int32)
        input_list.append(in0)

        # Prepare the expected list for the output
        expected0 = np.ndarray.copy(in0)
        expected_list.append(expected0)

        expected_dict[output_name] = expected_list

        input_byte_size = len(in0) * np.dtype(tensor_dtype).itemsize
        output_byte_size = input_byte_size * batch_size
        dummy_input_byte_size = tu.shape_element_count(dummy_input_shapes[io_num]) *\
            np.dtype(tensor_dtype).itemsize * batch_size
        # The dimension of this tensor will be the value of the shape tensor
        dummy_output_byte_size = tu.shape_element_count(in0) *\
            np.dtype(tensor_dtype).itemsize * batch_size

        # create and register shared memory region for inputs and outputs
        if use_cuda_shared_memory:
            shm_ip_handles.append(cudashm.create_shared_memory_region("input"+str(io_num)+"_data"+shm_suffix,
                                                                      input_byte_size, 0))
            shm_ip_handles.append(cudashm.create_shared_memory_region("dummy_input"+str(io_num)+"_data"+shm_suffix,
                                                                      dummy_input_byte_size, 0))
            shm_op_handles.append(cudashm.create_shared_memory_region("output"+str(io_num)+"_data"+shm_suffix,
                                                                      output_byte_size, 0))
            shm_op_handles.append(cudashm.create_shared_memory_region("dummy_output"+str(io_num)+"_data"+shm_suffix,
                                                                      dummy_output_byte_size, 0))

            shm_client.register_cuda_shared_memory("input"+str(io_num)+"_data"+shm_suffix,
                                                   cudashm.get_raw_handle(shm_ip_handles[2 * io_num]),
                                                   0, input_byte_size)
            shm_client.register_cuda_shared_memory("dummy_input"+str(io_num)+"_data"+shm_suffix,
                                                    cudashm.get_raw_handle(shm_ip_handles[2 * io_num + 1]),
                                                    0, dummy_input_byte_size)
            shm_client.register_cuda_shared_memory("output"+str(io_num)+"_data"+shm_suffix,
                                                    cudashm.get_raw_handle(shm_op_handles[2 * io_num]),
                                                    0, output_byte_size)
            shm_client.register_cuda_shared_memory("dummy_output"+str(io_num)+"_data"+shm_suffix,
                                                    cudashm.get_raw_handle(shm_op_handles[2 * io_num + 1]),
                                                   0, dummy_output_byte_size)                                                                                                      

            # copy data into shared memory region for input values
            cudashm.set_shared_memory_region(shm_ip_handles[2 * io_num], input_list)
            cudashm.set_shared_memory_region(shm_ip_handles[2 * io_num + 1], dummy_input_list)
        elif use_system_shared_memory:
            shm_ip_handles.append(shm.create_shared_memory_region("input"+str(io_num)+"_data"+shm_suffix,
                                                                  "/input"+str(io_num)+shm_suffix, input_byte_size))
            shm_ip_handles.append(shm.create_shared_memory_region("dummy_input"+str(io_num)+"_data"+shm_suffix,
                                                                  "/dummy_input"+str(io_num)+shm_suffix, dummy_input_byte_size))
            shm_op_handles.append(shm.create_shared_memory_region("output"+str(io_num)+"_data"+shm_suffix,
                                                                  "/output"+str(io_num)+shm_suffix, output_byte_size))
            shm_op_handles.append(shm.create_shared_memory_region("dummy_output"+str(io_num)+"_data"+shm_suffix,
                                                                  "/dummy_output"+str(io_num)+shm_suffix, dummy_output_byte_size))

            shm_client.register_system_shared_memory("input"+str(io_num)+"_data"+shm_suffix,
                                                     "/input"+str(io_num)+shm_suffix, input_byte_size)
            shm_client.register_system_shared_memory("dummy_input"+str(io_num)+"_data"+shm_suffix,
                                                     "/dummy_output"+str(io_num)+shm_suffix, dummy_input_byte_size)
            shm_client.register_system_shared_memory("output"+str(io_num)+"_data"+shm_suffix,
                                                        "/output"+str(io_num)+shm_suffix, output_byte_size)
            shm_client.register_system_shared_memory("dummy_output"+str(io_num)+"_data"+shm_suffix,
                                                        "/dummy_output"+str(io_num)+shm_suffix, dummy_output_byte_size)

            # copy data into shared memory region for input values
            shm.set_shared_memory_region(shm_ip_handles[2 * io_num], input_list)
            shm.set_shared_memory_region(shm_ip_handles[2 * io_num + 1], dummy_input_list)

        # if use_system_shared_memory or use_cuda_shared_memory:
        #     input_dict[input_name] = (
        #         shm_ip_handles[2 * io_num], [len(input_shape_values[0])])
        #     input_dict[dummy_input_name] = (
        #         shm_ip_handles[2 * io_num + 1], dummy_input_shapes[io_num])
        #     output_dict[output_name] = (
        #         InferContext.ResultFormat.RAW, shm_op_handles[2 * io_num])
        #     output_dict[dummy_output_name] = (
        #         InferContext.ResultFormat.RAW, shm_op_handles[2 * io_num + 1])
        # else:
        input_dict[input_name] = input_list
        input_dict[dummy_input_name] = dummy_input_list
        #     output_dict[output_name] = InferContext.ResultFormat.RAW
        #     output_dict[dummy_output_name] = InferContext.ResultFormat.RAW

    # Run inference and check results for each config
    for config in configs:
        model_name = tu.get_zero_model_name(pf, io_cnt, tensor_dtype)

        if config[1] == "http":
            triton_client = httpclient.InferenceServerClient(
                config[0], verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(
                config[0], verbose=True)

        inputs = []
        outputs = []
        for io_num in range(io_cnt):
            tester.assertTrue(pf == "plan" or pf == "plan_nobatch")

            input_name = "INPUT{}".format(io_num)
            dummy_input_name = "DUMMY_INPUT{}".format(io_num)
            output_name = "OUTPUT{}".format(io_num)
            dummy_output_name = "DUMMY_OUTPUT{}".format(io_num)
            if config[1] == "http":
                inputs.append(httpclient.InferInput(
                    input_name, input_shape_values[io_num],
                    np_to_triton_dtype(tensor_dtype)))
                inputs.append(httpclient.InferInput(
                    dummy_input_name, dummy_input_shapes[io_num],
                    np_to_triton_dtype(tensor_dtype)))
                outputs.append(httpclient.InferRequestedOutput(
                    output_name, binary_data=config[3]))
                outputs.append(httpclient.InferRequestedOutput(
                    dummy_output_name, binary_data=config[3]))
            else:
                inputs.append(grpcclient.InferInput(
                    input_name, input_shape_values[io_num],
                    np_to_triton_dtype(tensor_dtype)))
                inputs.append(grpcclient.InferInput(
                    dummy_input_name, dummy_input_shapes[io_num],
                    np_to_triton_dtype(tensor_dtype)))
                outputs.append(grpcclient.InferRequestedOutput(output_name))
                outputs.append(grpcclient.InferRequestedOutput(dummy_output_name))

            if not (use_cuda_shared_memory or use_system_shared_memory):
                if config[1] == "http":
                    inputs[0].set_data_from_numpy(
                        input_dict[input_name], binary_data=config[3])
                    inputs[1].set_data_from_numpy(
                        input_dict[dummy_input_name], binary_data=config[3])
                else:
                    inputs[0].set_data_from_numpy(input_dict[input_name])
                    inputs[1].set_data_from_numpy(input_dict[dummy_input_name])
            else:
                input_byte_size = tu.shape_element_count(input_shape_values[io_num]) *\
                    np.dtype(tensor_dtype).itemsize
                output_byte_size = input_byte_size * batch_size
                dummy_input_byte_size = tu.shape_element_count(dummy_input_shapes[io_num]) *\
                    np.dtype(tensor_dtype).itemsize * batch_size
                dummy_output_byte_size = tu.shape_element_count(in0) *\
                    np.dtype(tensor_dtype).itemsize * batch_size

                inputs[0].set_shared_memory("input"+str(io_num)+"_data"+shm_suffix, 
                                            input_byte_size)
                inputs[1].set_shared_memory("dummy_input"+str(io_num)+"_data"+shm_suffix, 
                                            dummy_input_byte_size)
                outputs[0].set_shared_memory("output"+str(io_num)+"_data"+shm_suffix, 
                                            output_byte_size)
                outputs[1].set_shared_memory("dummy_output"+str(io_num)+"_data"+shm_suffix, 
                                            dummy_output_byte_size)

        results = triton_client.infer(model_name,
                                    inputs,
                                    model_version=model_version,
                                    outputs=outputs,
                                    request_id=str(_unique_request_id()),
                                    priority=priority, 
                                    timeout_us=timeout_us)

        last_response = results.get_response()
        if config[1] == "http":
            if 'error' in last_response:
                raise InferenceServerException(msg=last_response['error'])

        if config[1] == "http":
            response_model_name = last_response["model_name"]
        else:
            response_model_name = last_response.model_name
        tester.assertEqual(response_model_name, model_name)

        if model_version != "":
            if config[1] == "http":
                response_model_version = last_response["model_version"]
            else:
                response_model_version = last_response.model_version
            tester.assertEqual(response_model_version, model_version)

        if config[1] == "http":
            response_outputs = last_response["outputs"]
        else:
            response_outputs = last_response.outputs
        tester.assertEqual(len(response_outputs), 2 * io_cnt)

        for result in response_outputs:
            if config[1] == "http":
                result_name = result["name"]
            else:
                result_name = result.name
            tester.assertTrue(result_name in expected_dict)

            if result_name == OUTPUT0:
                shm_handle = op0_handle
            else:
                shm_handle = op1_handle

            if use_system_shared_memory or use_cuda_shared_memory:
                output = results.get_output(result_name)
                if config[1] == "http":
                    output_datatype = output['datatype']
                    output_shape = output['shape']
                else:
                    output_datatype = output.datatype
                    output_shape = output.shape
                output_dtype = triton_to_np_dtype(output_datatype)
            if use_system_shared_memory:
                result_val = shm.get_contents_as_numpy(
                    shm_handle, output_dtype, output_shape)
            elif use_cuda_shared_memory:
                result_val = cudashm.get_contents_as_numpy(
                    shm_handle, output_dtype, output_shape)
            else:
                result_val = results.as_numpy(result_name)

            expected = expected_dict[output_name][0]
            for b in range(batch_size):
                if result_name == output_name:
                    tester.assertEqual(result_val[b].shape, expected.shape)
                    tester.assertTrue(np.array_equal(result_val[b], expected),
                                      "{}, {}, slot {}, expected: {}, got {}".format(
                        model_name, result_name, b, expected, result_val[b]))
                elif result_name == dummy_output_name:
                    # The shape of the dummy output should be equal to the shape values
                    # specified in the shape tensor
                    tester.assertTrue(np.array_equal(result_val[b].shape, expected),
                                      "{}, {}, slot {}, expected: {}, got {}".format(
                        model_name, result_name, b, expected, result_val[b]))
            
    if use_cuda_shared_memory or use_system_shared_memory:
        for io_num in range(io_cnt):
            if use_cuda_shared_memory:
                shm_client.unregister_cuda_shared_memory(
                    "input"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_cuda_shared_memory(
                    "dummy_input"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_cuda_shared_memory(
                    "output"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_cuda_shared_memory(
                    "dummy_output"+str(io_num)+"_data"+shm_suffix)

                cudashm.destroy_shared_memory_region(shm_ip_handles[2*io_num])
                cudashm.destroy_shared_memory_region(shm_ip_handles[2*io_num + 1])
                cudashm.destroy_shared_memory_region(shm_op_handles[2*io_num])
                cudashm.destroy_shared_memory_region(shm_op_handles[2*io_num + 1])
            else:
                shm_client.unregister_system_shared_memory(
                    "input"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_system_shared_memory(
                    "dummy_input"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_system_shared_memory(
                    "output"+str(io_num)+"_data"+shm_suffix)
                shm_client.unregister_system_shared_memory(
                    "dummy_output"+str(io_num)+"_data"+shm_suffix)

                shm.destroy_shared_memory_region(shm_ip_handles[2*io_num])
                shm.destroy_shared_memory_region(shm_ip_handles[2*io_num + 1])
                shm.destroy_shared_memory_region(shm_op_handles[2*io_num])
                shm.destroy_shared_memory_region(shm_op_handles[2*io_num + 1])

    return results
