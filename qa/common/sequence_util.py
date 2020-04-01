# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from builtins import str
from future.utils import iteritems
import os
import time
import threading
import traceback
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import sequence_util as su
from functools import partial
from tensorrtserver.api import *
import tensorrtserver.shared_memory as shm
import tensorrtserver.cuda_shared_memory as cudashm
import tensorrtserver.api.server_status_pb2 as server_status
if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

_test_system_shared_memory = bool(int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
_test_cuda_shared_memory = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY', 0)))

_max_sequence_idle_ms = 5000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = None

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_run()
def completion_callback(user_data, infer_ctx, request_id):
    user_data._completed_requests.put(request_id)

class SequenceBatcherTestUtil(unittest.TestCase):
    def setUp(self):
        self.clear_deferred_exceptions()

    def clear_deferred_exceptions(self):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
          _deferred_exceptions = []

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        # Just raise one of the exceptions...
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_failure(self):
        # Check securely whether a failure has been registered
        # This is generic because the failure behavior is undefined
        # for ragged batches.
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) == 0:
                raise Exception("Unexpected inference success")

    def precreate_register_regions(self, value_list, dtype, i,
                                batch_size=1, tensor_shape=(1,)):
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",
                                                           ProtocolType.HTTP, verbose=True)
            shm_region_handles = []
            for j, value in enumerate(value_list):
                # For string we can't know the size of the output
                # so we conservatively assume 64 bytes for each
                # element of the output
                if dtype == np.object:
                    output_byte_size = 4 # size of empty string
                else:
                    output_byte_size = 0

                # create data
                input_list = list()
                for b in range(batch_size):
                    if dtype == np.object:
                        in0 = np.full(tensor_shape, value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                        in0 = in0n.reshape(tensor_shape)
                        output_byte_size += 64 * in0.size
                    else:
                        in0 = np.full(tensor_shape, value, dtype=dtype)
                        output_byte_size += np.dtype(dtype).itemsize * in0.size
                    input_list.append(in0)

                input_list_tmp = iu._prepend_string_size(input_list) if (dtype == np.object) else input_list
                input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])

                # create shared memory regions and copy data for input values
                if _test_system_shared_memory:
                    shm_ip_handle = shm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), '/ip{}{}'.format(i,j), input_byte_size)
                    shm_op_handle = shm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), '/op{}{}'.format(i,j), output_byte_size)
                    shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    shared_memory_ctx.register(shm_ip_handle)
                    shared_memory_ctx.register(shm_op_handle)
                elif _test_cuda_shared_memory:
                    shm_ip_handle = cudashm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), input_byte_size, 0)
                    shm_op_handle = cudashm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), output_byte_size, 0)
                    cudashm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    shared_memory_ctx.cuda_register(shm_ip_handle)
                    shared_memory_ctx.cuda_register(shm_op_handle)
                shm_region_handles.append(shm_ip_handle)
                shm_region_handles.append(shm_op_handle)
            return shm_region_handles
        else:
            return []

    def precreate_register_shape_tensor_regions(self, value_list, dtype, i,
                                            batch_size=1, tensor_shape=(1,)):
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",
                                                           ProtocolType.HTTP, verbose=True)
            shm_region_handles = []
            for j, (shape_value, value) in enumerate(value_list):
                input_list = list()
                shape_input_list = list()

                for b in range(batch_size):
                    if dtype == np.object:
                        in0 = np.full(tensor_shape, value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                        in0 = in0n.reshape(tensor_shape)
                    else:
                        in0 = np.full(tensor_shape, value, dtype=dtype)
                    input_list.append(in0)

                # Only one shape tensor input per batch
                shape_input_list.append(np.full(tensor_shape, shape_value, dtype=np.int32))

                input_list_tmp = iu._prepend_string_size(input_list) if (dtype == np.object) else input_list
                input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])
                shape_input_byte_size = sum([i0.nbytes for i0 in shape_input_list])
                shape_output_byte_size = shape_input_byte_size
                output_byte_size = np.dtype(dtype).itemsize + 2
                resized_output_byte_size = 32 * shape_value

                # create shared memory regions and copy data for input values
                if _test_system_shared_memory:
                    shm_ip_handle = shm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), '/ip{}{}'.format(i,j), input_byte_size)
                    shm_shape_ip_handle = shm.create_shared_memory_region(
                        'shape_ip{}{}_data'.format(i,j), '/shape_ip{}{}'.format(i,j), shape_input_byte_size)
                    shm_shape_op_handle = shm.create_shared_memory_region(
                        'shape_op{}{}_data'.format(i,j), '/shape_op{}{}'.format(i,j), shape_output_byte_size)
                    shm_op_handle = shm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), '/op{}{}'.format(i,j), output_byte_size)
                    shm_resized_op_handle = shm.create_shared_memory_region(
                        'resized_op{}{}_data'.format(i,j), '/resized_op{}{}'.format(i,j), resized_output_byte_size)
                    shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    shm.set_shared_memory_region(shm_shape_ip_handle, shape_input_list)
                    shared_memory_ctx.register(shm_ip_handle)
                    shared_memory_ctx.register(shm_shape_ip_handle)
                    shared_memory_ctx.register(shm_shape_op_handle)
                    shared_memory_ctx.register(shm_op_handle)
                    shared_memory_ctx.register(shm_resized_op_handle)
                elif _test_cuda_shared_memory:
                    shm_ip_handle = cudashm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), input_byte_size, 0)
                    shm_shape_ip_handle = cudashm.create_shared_memory_region(
                        'shape_ip{}{}_data'.format(i,j), shape_input_byte_size, 0)
                    shm_shape_op_handle = cudashm.create_shared_memory_region(
                        'shape_op{}{}_data'.format(i,j), shape_output_byte_size, 0)
                    shm_op_handle = cudashm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), output_byte_size, 0)
                    shm_resized_op_handle = cudashm.create_shared_memory_region(
                        'resized_op{}{}_data'.format(i,j), resized_output_byte_size, 0)
                    cudashm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    cudashm.set_shared_memory_region(shm_shape_ip_handle, shape_input_list)
                    shared_memory_ctx.cuda_register(shm_ip_handle)
                    shared_memory_ctx.cuda_register(shm_shape_ip_handle)
                    shared_memory_ctx.cuda_register(shm_shape_op_handle)
                    shared_memory_ctx.cuda_register(shm_op_handle)
                    shared_memory_ctx.cuda_register(shm_resized_op_handle)
                shm_region_handles.append(shm_ip_handle)
                shm_region_handles.append(shm_shape_ip_handle)
                shm_region_handles.append(shm_shape_op_handle)
                shm_region_handles.append(shm_op_handle)
                shm_region_handles.append(shm_resized_op_handle)
            return shm_region_handles
        else:
            return []

    def precreate_register_dynaseq_shape_tensor_regions(self, value_list, dtype, i,
                                            batch_size=1, tensor_shape=(1,)):
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",
                                                           ProtocolType.HTTP, verbose=True)
            shm_region_handles = []
            for j, (shape_value, value) in enumerate(value_list):
                input_list = list()
                shape_input_list = list()
                dummy_input_list = list()

                for b in range(batch_size):
                    if dtype == np.object:
                        dummy_in0 = np.full(tensor_shape, value, dtype=np.int32)
                        dummy_in0n = np.array([str(x) for x in dummy_in0.reshape(in0.size)], dtype=object)
                        dummy_in0 = dummy_in0n.reshape(tensor_shape)
                    else:
                        dummy_in0 = np.full(tensor_shape, value, dtype=dtype)
                    dummy_input_list.append(dummy_in0)
                    in0 =  np.full(tensor_shape, value, dtype=np.int32)
                    input_list.append(in0)

                # Only one shape tensor input per batch
                shape_input_list.append(np.full(tensor_shape, shape_value, dtype=np.int32))

                input_list_tmp = iu._prepend_string_size(input_list) if (dtype == np.object) else input_list
                input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])
                shape_input_byte_size = sum([i0.nbytes for i0 in shape_input_list])
                dummy_input_byte_size = sum([i0.nbytes for i0 in dummy_input_list])
                shape_output_byte_size = shape_input_byte_size
                output_byte_size = np.dtype(np.int32).itemsize + 2
                resized_output_byte_size = 32 * shape_value

                # create shared memory regions and copy data for input values
                if _test_system_shared_memory:
                    shm_ip_handle = shm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), '/ip{}{}'.format(i,j), input_byte_size)
                    shm_shape_ip_handle = shm.create_shared_memory_region(
                        'shape_ip{}{}_data'.format(i,j), '/shape_ip{}{}'.format(i,j), shape_input_byte_size)
                    shm_dummy_ip_handle = shm.create_shared_memory_region(
                        'dummy_ip{}{}_data'.format(i,j), '/dummy_ip{}{}'.format(i,j), dummy_input_byte_size)
                    shm_shape_op_handle = shm.create_shared_memory_region(
                        'shape_op{}{}_data'.format(i,j), '/shape_op{}{}'.format(i,j), shape_output_byte_size)
                    shm_op_handle = shm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), '/op{}{}'.format(i,j), output_byte_size)
                    shm_resized_op_handle = shm.create_shared_memory_region(
                        'resized_op{}{}_data'.format(i,j), '/resized_op{}{}'.format(i,j), resized_output_byte_size)
                    shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    shm.set_shared_memory_region(shm_shape_ip_handle, shape_input_list)
                    shm.set_shared_memory_region(shm_dummy_ip_handle, dummy_input_list)
                    shared_memory_ctx.register(shm_ip_handle)
                    shared_memory_ctx.register(shm_shape_ip_handle)
                    shared_memory_ctx.register(shm_dummy_ip_handle)
                    shared_memory_ctx.register(shm_shape_op_handle)
                    shared_memory_ctx.register(shm_op_handle)
                    shared_memory_ctx.register(shm_resized_op_handle)
                elif _test_cuda_shared_memory:
                    shm_ip_handle = cudashm.create_shared_memory_region(
                        'ip{}{}_data'.format(i,j), input_byte_size, 0)
                    shm_shape_ip_handle = cudashm.create_shared_memory_region(
                        'shape_ip{}{}_data'.format(i,j), shape_input_byte_size, 0)
                    shm_dummy_ip_handle = cudashm.create_shared_memory_region(
                        'dummy_ip{}{}_data'.format(i,j), dummy_input_byte_size, 0)
                    shm_shape_op_handle = cudashm.create_shared_memory_region(
                        'shape_op{}{}_data'.format(i,j), shape_output_byte_size, 0)
                    shm_op_handle = cudashm.create_shared_memory_region(
                        'op{}{}_data'.format(i,j), output_byte_size, 0)
                    shm_resized_op_handle = cudashm.create_shared_memory_region(
                        'resized_op{}{}_data'.format(i,j), resized_output_byte_size, 0)
                    cudashm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    cudashm.set_shared_memory_region(shm_shape_ip_handle, shape_input_list)
                    cudashm.set_shared_memory_region(shm_dummy_ip_handle, dummy_input_list)
                    shared_memory_ctx.cuda_register(shm_ip_handle)
                    shared_memory_ctx.cuda_register(shm_shape_ip_handle)
                    shared_memory_ctx.cuda_register(shm_dummy_ip_handle)
                    shared_memory_ctx.cuda_register(shm_shape_op_handle)
                    shared_memory_ctx.cuda_register(shm_op_handle)
                    shared_memory_ctx.cuda_register(shm_resized_op_handle)
                shm_region_handles.append(shm_ip_handle)
                shm_region_handles.append(shm_shape_ip_handle)
                shm_region_handles.append(shm_dummy_ip_handle)
                shm_region_handles.append(shm_shape_op_handle)
                shm_region_handles.append(shm_op_handle)
                shm_region_handles.append(shm_resized_op_handle)
            return shm_region_handles
        else:
            return []

    def cleanup_shm_regions(self, shm_handles):
        if len(shm_handles) != 0:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000", ProtocolType.HTTP, verbose=True)
            for shm_tmp_handle in shm_handles:
                shared_memory_ctx.unregister(shm_tmp_handle)
                if _test_system_shared_memory:
                    shm.destroy_shared_memory_region(shm_tmp_handle)
                elif _test_cuda_shared_memory:
                    cudashm.destroy_shared_memory_region(shm_tmp_handle)

    def check_sequence(self, trial, model_name, input_dtype, correlation_id,
                       sequence_thresholds, values, expected_result,
                       protocol, batch_size=1, sequence_name="<unknown>", tensor_shape=(1,)):
        """Perform sequence of inferences. The 'values' holds a list of
        tuples, one for each inference with format:

        (flag_str, value, (ls_ms, gt_ms), (pre_delay_ms, post_delay_ms)

        """
        if (("savedmodel" not in trial) and ("graphdef" not in trial) and
            ("netdef" not in trial) and ("custom" not in trial) and
            ("onnx" not in trial) and ("libtorch" not in trial) and
	    ("plan" not in trial)):
            self.assertFalse(True, "unknown trial type: " + trial)

        # Can only send the request exactly once since it is a
        # sequence model with state, so can have only a single config.
        configs = []
        if protocol == "http":
            configs.append(("localhost:8000", ProtocolType.HTTP, False))
        if protocol == "grpc":
            configs.append(("localhost:8001", ProtocolType.GRPC, False))
        if protocol == "streaming":
            configs.append(("localhost:8001", ProtocolType.GRPC, True))

        self.assertFalse(_test_system_shared_memory and _test_cuda_shared_memory,
                        "Cannot set both System and CUDA shared memory flags to 1")

        self.assertEqual(len(configs), 1)

        # create and register shared memory output region in advance
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",
                                                           ProtocolType.HTTP, verbose=True)
            output_byte_size = 512
            if _test_system_shared_memory:
                shm_op_handle = shm.create_shared_memory_region("output_data", "/output", output_byte_size)
                shared_memory_ctx.unregister(shm_op_handle)
                shared_memory_ctx.register(shm_op_handle)
            elif _test_cuda_shared_memory:
                shm_op_handle = cudashm.create_shared_memory_region("output_data", output_byte_size, 0)
                shared_memory_ctx.unregister(shm_op_handle)
                shared_memory_ctx.cuda_register(shm_op_handle)

        for config in configs:
            ctx = InferContext(config[0], config[1], model_name,
                               correlation_id=correlation_id, streaming=config[2],
                               verbose=True)
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))

                for flag_str, value, thresholds, delay_ms in values:
                    if delay_ms is not None:
                        time.sleep(delay_ms[0] / 1000.0)

                    flags = InferRequestHeader.FLAG_NONE
                    if flag_str is not None:
                        if "start" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
                        if "end" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

                    input_list = list()
                    for b in range(batch_size):
                        if input_dtype == np.object:
                            in0 = np.full(tensor_shape, value, dtype=np.int32)
                            in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                            in0 = in0n.reshape(tensor_shape)
                        else:
                            in0 = np.full(tensor_shape, value, dtype=input_dtype)
                        input_list.append(in0)

                    # create input shared memory and copy input data values into it
                    if _test_system_shared_memory or _test_cuda_shared_memory:
                        input_list_tmp = iu._prepend_string_size(input_list) if (input_dtype == np.object) else input_list
                        input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])
                        if _test_system_shared_memory:
                            shm_ip_handle = shm.create_shared_memory_region("input_data", "/input", input_byte_size)
                            shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                            shared_memory_ctx.unregister(shm_ip_handle)
                            shared_memory_ctx.register(shm_ip_handle)
                        elif _test_cuda_shared_memory:
                            shm_ip_handle = cudashm.create_shared_memory_region("input_data", input_byte_size, 0)
                            cudashm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                            shared_memory_ctx.unregister(shm_ip_handle)
                            shared_memory_ctx.cuda_register(shm_ip_handle)

                        input_info = (shm_ip_handle, tensor_shape)
                        output_info = (InferContext.ResultFormat.RAW, shm_op_handle)
                    else:
                        input_info = input_list
                        output_info = InferContext.ResultFormat.RAW

                    start_ms = int(round(time.time() * 1000))
                    INPUT = "INPUT__0" if trial.startswith("libtorch") else "INPUT"
                    OUTPUT = "OUTPUT__0" if trial.startswith("libtorch") else "OUTPUT"

                    results = ctx.run(
                        { INPUT : input_info }, { OUTPUT : output_info},
                        batch_size=batch_size, flags=flags)

                    end_ms = int(round(time.time() * 1000))

                    self.assertEqual(len(results), 1)
                    self.assertTrue(OUTPUT in results)
                    result = results[OUTPUT][0][0]
                    print("{}: {}".format(sequence_name, result))

                    if thresholds is not None:
                        lt_ms = thresholds[0]
                        gt_ms = thresholds[1]
                        if lt_ms is not None:
                            self.assertTrue((end_ms - start_ms) < lt_ms,
                                            "expected less than " + str(lt_ms) +
                                            "ms response time, got " + str(end_ms - start_ms) + " ms")
                        if gt_ms is not None:
                            self.assertTrue((end_ms - start_ms) > gt_ms,
                                            "expected greater than " + str(gt_ms) +
                                            "ms response time, got " + str(end_ms - start_ms) + " ms")
                    if delay_ms is not None:
                        time.sleep(delay_ms[1] / 1000.0)

                seq_end_ms = int(round(time.time() * 1000))

                if input_dtype == np.object:
                    self.assertEqual(int(result), expected_result)
                else:
                    self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " + str(lt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " + str(gt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                self.add_deferred_exception(ex)

        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx.unregister(shm_op_handle)
            if _test_system_shared_memory:
                shm.destroy_shared_memory_region(shm_op_handle)
            elif _test_cuda_shared_memory:
                cudashm.destroy_shared_memory_region(shm_op_handle)


    def check_sequence_async(self, trial, model_name, input_dtype, correlation_id,
                             sequence_thresholds, values, expected_result,
                             protocol, shm_region_handles, batch_size=1,
                             sequence_name="<unknown>", tensor_shape=(1,)):
        """Perform sequence of inferences using async run. The 'values' holds
        a list of tuples, one for each inference with format:

        (flag_str, value, pre_delay_ms)

        """
        if (("savedmodel" not in trial) and ("graphdef" not in trial) and
            ("netdef" not in trial) and ("custom" not in trial) and
            ("onnx" not in trial) and ("libtorch" not in trial) and
	    ("plan" not in trial)):
            self.assertFalse(True, "unknown trial type: " + trial)

        self.assertFalse(_test_system_shared_memory and _test_cuda_shared_memory,
                        "Cannot set both System and CUDA shared memory flags to 1")

        # Can only send the request exactly once since it is a
        # sequence model with state
        configs = []
        if protocol == "http":
            configs.append(("localhost:8000", ProtocolType.HTTP, False))
        if protocol == "grpc":
            configs.append(("localhost:8001", ProtocolType.GRPC, False))
        if protocol == "streaming":
            configs.append(("localhost:8001", ProtocolType.GRPC, True))
        self.assertEqual(len(configs), 1)

        for config in configs:
            ctx = InferContext(config[0], config[1], model_name,
                               correlation_id=correlation_id, streaming=config[2],
                               verbose=True)
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))
                user_data = UserData()

                sent_count = 0
                for flag_str, value, pre_delay_ms in values:
                    flags = InferRequestHeader.FLAG_NONE
                    if flag_str is not None:
                        if "start" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
                        if "end" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

                    if not (_test_system_shared_memory or _test_cuda_shared_memory):
                        input_list = list()
                        for b in range(batch_size):
                            if input_dtype == np.object:
                                in0 = np.full(tensor_shape, value, dtype=np.int32)
                                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                                in0 = in0n.reshape(tensor_shape)
                            else:
                                in0 = np.full(tensor_shape, value, dtype=input_dtype)
                            input_list.append(in0)

                        input_info = input_list
                        output_info = InferContext.ResultFormat.RAW
                    else:
                        input_info = (shm_region_handles[2*sent_count], tensor_shape)
                        output_info = (InferContext.ResultFormat.RAW, shm_region_handles[2*sent_count+1])

                    if pre_delay_ms is not None:
                        time.sleep(pre_delay_ms / 1000.0)

                    INPUT = "INPUT__0" if trial.startswith("libtorch") else "INPUT"
                    OUTPUT = "OUTPUT__0" if trial.startswith("libtorch") else "OUTPUT"

                    ctx.async_run(partial(completion_callback, user_data),
                        { INPUT :input_info }, { OUTPUT :output_info },
                        batch_size=batch_size, flags=flags)
                    sent_count+=1

                # Wait for the results in the order sent
                result = None
                processed_count = 0
                while processed_count < sent_count:
                    id = user_data._completed_requests.get()
                    results = ctx.get_async_run_results(id)
                    self.assertEqual(len(results), 1)
                    self.assertTrue(OUTPUT in results)
                    result = results[OUTPUT][0][0]
                    print("{}: {}".format(sequence_name, result))
                    processed_count+=1

                seq_end_ms = int(round(time.time() * 1000))

                if input_dtype == np.object:
                    self.assertEqual(int(result), expected_result)
                else:
                    self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " + str(lt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " + str(gt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                self.add_deferred_exception(ex)

    def check_sequence_shape_tensor_io(self, model_name, input_dtype, correlation_id,
                             sequence_thresholds, values, expected_result,
                             protocol, shm_region_handles, using_dynamic_batcher=False,
                             batch_size=1, sequence_name="<unknown>"):

        """Perform sequence of inferences using async run. The 'values' holds
        a list of tuples, one for each inference with format:

        (flag_str, shape_value, value, pre_delay_ms)

        """
        tensor_shape = (1,)
        self.assertFalse(_test_system_shared_memory and _test_cuda_shared_memory,
                        "Cannot set both System and CUDA shared memory flags to 1")

        # Can only send the request exactly once since it is a
        # sequence model with state
        configs = []
        if protocol == "http":
            configs.append(("localhost:8000", ProtocolType.HTTP, False))
        if protocol == "grpc":
            configs.append(("localhost:8001", ProtocolType.GRPC, False))
        if protocol == "streaming":
            configs.append(("localhost:8001", ProtocolType.GRPC, True))
        self.assertEqual(len(configs), 1)

        for config in configs:
            ctx = InferContext(config[0], config[1], model_name,
                               correlation_id=correlation_id, streaming=config[2],
                               verbose=True)
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))
                user_data = UserData()

                sent_count = 0
                shape_values = list()
                for flag_str, shape_value, value, pre_delay_ms in values:
                    flags = InferRequestHeader.FLAG_NONE
                    if flag_str is not None:
                        if "start" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
                        if "end" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

                    shape_values.append(np.full(tensor_shape, shape_value, dtype=np.int32))
                    if not (_test_system_shared_memory or _test_cuda_shared_memory):
                        input_list = list()
                        shape_input_list = list()
                        dummy_input_list = list()

                        for b in range(batch_size):
                            if using_dynamic_batcher:
                                if input_dtype == np.object:
                                    dummy_in0 = np.full(tensor_shape, value, dtype=np.int32)
                                    dummy_in0n = np.array([str(x) for x in in0.reshape(dummy_in0.size)], dtype=object)
                                    dummy_in0 = dummy_in0n.reshape(tensor_shape)
                                else:
                                    dummy_in0 = np.full(tensor_shape, value, dtype=input_dtype)
                                dummy_input_list.append(dummy_in0)

                                in0 = np.full(tensor_shape, value, dtype=np.int32)
                                input_list.append(in0)
                            else:
                                if input_dtype == np.object:
                                    in0 = np.full(tensor_shape, value, dtype=np.int32)
                                    in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                                    in0 = in0n.reshape(tensor_shape)
                                else:
                                    in0 = np.full(tensor_shape, value, dtype=input_dtype)
                                input_list.append(in0)


                        input_info = input_list
                        dummy_input_info = dummy_input_list
                        # Only one shape tensor input per batch
                        shape_input_list.append(shape_values[sent_count])
                        shape_input_info = shape_input_list

                        output_info = InferContext.ResultFormat.RAW
                        shape_output_info = InferContext.ResultFormat.RAW
                        resized_output_info = InferContext.ResultFormat.RAW
                    else:
                        if using_dynamic_batcher:
                            input_info = (shm_region_handles[6*sent_count], tensor_shape)
                            shape_input_info = (shm_region_handles[6*sent_count+1], tensor_shape)
                            dummy_input_info = (shm_region_handles[6*sent_count+2], tensor_shape)
                            shape_output_info = (InferContext.ResultFormat.RAW, shm_region_handles[6*sent_count+3])
                            output_info = (InferContext.ResultFormat.RAW, shm_region_handles[6*sent_count+4])
                            resized_output_info = (InferContext.ResultFormat.RAW, shm_region_handles[6*sent_count+5])
                        else:
                            input_info = (shm_region_handles[5*sent_count], tensor_shape)
                            shape_input_info = (shm_region_handles[5*sent_count+1], tensor_shape)
                            shape_output_info = (InferContext.ResultFormat.RAW, shm_region_handles[5*sent_count+2])
                            output_info = (InferContext.ResultFormat.RAW, shm_region_handles[5*sent_count+3])
                            resized_output_info = (InferContext.ResultFormat.RAW, shm_region_handles[5*sent_count+4])

                    if pre_delay_ms is not None:
                        time.sleep(pre_delay_ms / 1000.0)

                    INPUT = "INPUT"
                    OUTPUT = "OUTPUT"
                    if using_dynamic_batcher:
                        ctx.async_run(partial(completion_callback, user_data),
                            { INPUT : input_info, "SHAPE_INPUT" :shape_input_info, "DUMMY_INPUT" :dummy_input_info },
                            { OUTPUT : output_info,"SHAPE_OUTPUT" :shape_output_info, "RESIZED_OUTPUT" :resized_output_info},
                            batch_size=batch_size, flags=flags)
                    else:
                        ctx.async_run(partial(completion_callback, user_data),
                            { INPUT : input_info, "SHAPE_INPUT" :shape_input_info },
                            { OUTPUT : output_info,"SHAPE_OUTPUT" :shape_output_info, "RESIZED_OUTPUT" :resized_output_info},
                            batch_size=batch_size, flags=flags)
                    sent_count+=1

                # Wait for the results in the order sent
                result = None
                processed_count = 0
                while processed_count < sent_count:
                    id = user_data._completed_requests.get()
                    results = ctx.get_async_run_results(id)
                    self.assertEqual(len(results), 3)
                    self.assertTrue(OUTPUT in results)
                    result = results[OUTPUT][0][0]
                    # Validate the shape of the resized output match with the shape input values
                    self.assertTrue(np.array_equal(results["RESIZED_OUTPUT"][0].shape, shape_values[processed_count]),
                                  "{}, {}, slot {}, expected: {}, got {}".format(
                                  model_name, "RESIZED_OUTPUT", processed_count, shape_values[processed_count],
                                  results["RESIZED_OUTPUT"][0].shape))
                    print("{}: {}".format(sequence_name, result))
                    processed_count+=1

                seq_end_ms = int(round(time.time() * 1000))

                if input_dtype == np.object:
                    self.assertEqual(int(result), expected_result)
                else:
                    self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " + str(lt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " + str(gt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                self.add_deferred_exception(ex)

    def check_setup(self, model_name):
        # Make sure test.sh set up the correct batcher settings
        ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        # Skip the sequence batching check on ensemble model
        if ss.model_status[model_name].config.platform != "ensemble":
            bconfig = ss.model_status[model_name].config.sequence_batching
            self.assertEqual(bconfig.max_sequence_idle_microseconds, _max_sequence_idle_ms * 1000) # 5 secs

    def check_status(self, model_name, static_bs, exec_cnt, infer_cnt):
        ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        vs = ss.model_status[model_name].version_status
        self.assertEqual(len(vs), 1)
        self.assertTrue(1 in vs, "expected status for version 1")
        infer = vs[1].infer_stats
        self.assertEqual(len(infer), len(static_bs),
                         "expected batch-sizes (" + ",".join(str(b) for b in static_bs) +
                         "), got " + str(vs[1]))
        for b in static_bs:
            self.assertTrue(b in infer,
                            "expected batch-size " + str(b) + ", got " + str(vs[1]))

        # Skip checking on ensemble because its execution count isn't modified like
        # sequence batcher.
        if ss.model_status[model_name].config.platform != "ensemble":
            self.assertEqual(vs[1].model_execution_count, exec_cnt,
                            "expected model-execution-count " + str(exec_cnt) + ", got " +
                            str(vs[1].model_execution_count))
            self.assertEqual(vs[1].model_inference_count, infer_cnt,
                            "expected model-inference-count " + str(infer_cnt) + ", got " +
                            str(vs[1].model_inference_count))
