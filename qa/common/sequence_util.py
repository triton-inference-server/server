# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys
import time
import threading
import numpy as np
import infer_util as iu
import test_util as tu
from functools import partial

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import *

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get('TRITONSERVER_IPADDR', 'localhost')

_test_system_shared_memory = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
_test_cuda_shared_memory = bool(
    int(os.environ.get('TEST_CUDA_SHARED_MEMORY', 0)))

if _test_system_shared_memory:
    import tritonclient.utils.shared_memory as shm
if _test_cuda_shared_memory:
    import tritonclient.utils.cuda_shared_memory as cudashm

_test_valgrind = bool(int(os.environ.get('TEST_VALGRIND', 0)))
_test_jetson = bool(int(os.environ.get('TEST_JETSON', 0)))

_max_sequence_idle_ms = 5000
_valgrind_delay_ms = bool(int(os.environ.get('TEST_DELAY_MS', 50)))

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = None
_jetson_slowdown_factor = 3


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class SequenceBatcherTestUtil(tu.TestResultCollector):

    def setUp(self):
        # The helper client for setup will be GRPC for simplicity.
        self.triton_client_ = grpcclient.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001")
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

    def precreate_register_regions(self,
                                   value_list,
                                   dtype,
                                   i,
                                   batch_size=1,
                                   tensor_shape=(1,)):
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shm_region_handles = []
            for j, value in enumerate(value_list):
                # For string we can't know the size of the output
                # so we conservatively assume 64 bytes for each
                # element of the output
                if dtype == np.object_:
                    output_byte_size = 4  # size of empty string
                else:
                    output_byte_size = 0

                # create data
                input_list = list()
                for b in range(batch_size):
                    if dtype == np.object_:
                        in0 = np.full(tensor_shape, value, dtype=np.int32)
                        in0n = np.array([
                            str(x).encode('utf-8')
                            for x in in0.reshape(in0.size)
                        ],
                                        dtype=object)
                        in0 = in0n.reshape(tensor_shape)
                        output_byte_size += 64 * in0.size
                    else:
                        in0 = np.full(tensor_shape, value, dtype=dtype)
                        output_byte_size += np.dtype(dtype).itemsize * in0.size
                    input_list.append(in0)

                if dtype == np.object_:
                    input_list_tmp = iu.serialize_byte_tensor_list(input_list)
                    input_byte_size = sum(
                        [serialized_byte_size(i0) for i0 in input_list_tmp])
                else:
                    input_list_tmp = input_list
                    input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])

                # create shared memory regions and copy data for input values
                ip_name = 'ip{}{}'.format(i, j)
                op_name = 'op{}{}_data'.format(i, j)
                if _test_system_shared_memory:
                    shm_ip_handle = shm.create_shared_memory_region(
                        ip_name, '/' + ip_name, input_byte_size)
                    shm_op_handle = shm.create_shared_memory_region(
                        op_name, '/' + op_name, output_byte_size)
                    shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                    self.triton_client_.register_system_shared_memory(
                        ip_name, '/' + ip_name, input_byte_size)
                    self.triton_client_.register_system_shared_memory(
                        op_name, '/' + op_name, output_byte_size)
                elif _test_cuda_shared_memory:
                    shm_ip_handle = cudashm.create_shared_memory_region(
                        ip_name, input_byte_size, 0)
                    shm_op_handle = cudashm.create_shared_memory_region(
                        op_name, output_byte_size, 0)
                    cudashm.set_shared_memory_region(shm_ip_handle,
                                                     input_list_tmp)
                    self.triton_client_.register_cuda_shared_memory(
                        ip_name, cudashm.get_raw_handle(shm_ip_handle), 0,
                        input_byte_size)
                    self.triton_client_.register_cuda_shared_memory(
                        op_name, cudashm.get_raw_handle(shm_op_handle), 0,
                        output_byte_size)
                shm_region_handles.append(
                    (ip_name, input_byte_size, shm_ip_handle))
                shm_region_handles.append(
                    (op_name, output_byte_size, shm_op_handle))
            return shm_region_handles
        else:
            return []

    # Returns (name, byte size, shm_handle)
    def precreate_register_shape_tensor_regions(self,
                                                value_list,
                                                dtype,
                                                i,
                                                batch_size=1,
                                                tensor_shape=(1,)):
        self.assertFalse(_test_cuda_shared_memory,
                         "Shape tensors does not support CUDA shared memory")
        if _test_system_shared_memory:
            shm_region_handles = []
            for j, (shape_value, value) in enumerate(value_list):
                input_list = list()
                shape_input_list = list()

                for b in range(batch_size):
                    if dtype == np.object_:
                        in0 = np.full(tensor_shape, value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)],
                                        dtype=object)
                        in0 = in0n.reshape(tensor_shape)
                    else:
                        in0 = np.full(tensor_shape, value, dtype=dtype)
                    input_list.append(in0)

                # Only one shape tensor input per batch
                shape_input_list.append(
                    np.full(tensor_shape, shape_value, dtype=np.int32))

                if dtype == np.object_:
                    input_list_tmp = iu.serialize_byte_tensor_list(input_list)
                    input_byte_size = sum(
                        [serialized_byte_size(i0) for i0 in input_list_tmp])
                else:
                    input_list_tmp = input_list
                    input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])

                shape_input_byte_size = sum(
                    [i0.nbytes for i0 in shape_input_list])
                shape_output_byte_size = shape_input_byte_size
                output_byte_size = np.dtype(dtype).itemsize + 2
                resized_output_byte_size = 32 * shape_value

                # create shared memory regions and copy data for input values
                ip_name = 'ip{}{}'.format(i, j)
                shape_ip_name = 'shape_ip{}{}'.format(i, j)
                shape_op_name = 'shape_op{}{}'.format(i, j)
                op_name = 'op{}{}'.format(i, j)
                resized_op_name = 'resized_op{}{}'.format(i, j)

                shm_ip_handle = shm.create_shared_memory_region(
                    ip_name, '/' + ip_name, input_byte_size)
                shm_shape_ip_handle = shm.create_shared_memory_region(
                    shape_ip_name, '/' + shape_ip_name, shape_input_byte_size)
                shm_shape_op_handle = shm.create_shared_memory_region(
                    shape_op_name, '/' + shape_op_name, shape_output_byte_size)
                shm_op_handle = shm.create_shared_memory_region(
                    op_name, '/' + op_name, output_byte_size)
                shm_resized_op_handle = shm.create_shared_memory_region(
                    resized_op_name, '/' + resized_op_name,
                    resized_output_byte_size)
                shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                shm.set_shared_memory_region(shm_shape_ip_handle,
                                             shape_input_list)
                self.triton_client_.register_system_shared_memory(
                    ip_name, '/' + ip_name, input_byte_size)
                self.triton_client_.register_system_shared_memory(
                    shape_ip_name, '/' + shape_ip_name, shape_input_byte_size)
                self.triton_client_.register_system_shared_memory(
                    shape_op_name, '/' + shape_op_name, shape_output_byte_size)
                self.triton_client_.register_system_shared_memory(
                    op_name, '/' + op_name, output_byte_size)
                self.triton_client_.register_system_shared_memory(
                    resized_op_name, '/' + resized_op_name,
                    resized_output_byte_size)

                shm_region_handles.append(
                    (ip_name, input_byte_size, shm_ip_handle))
                shm_region_handles.append(
                    (shape_ip_name, shape_input_byte_size, shm_shape_ip_handle))
                shm_region_handles.append(
                    (shape_op_name, shape_output_byte_size,
                     shm_shape_op_handle))
                shm_region_handles.append(
                    (op_name, output_byte_size, shm_op_handle))
                shm_region_handles.append(
                    (resized_op_name, resized_output_byte_size,
                     shm_resized_op_handle))
            return shm_region_handles
        else:
            return []

    # Returns (name, byte size, shm_handle)
    def precreate_register_dynaseq_shape_tensor_regions(self,
                                                        value_list,
                                                        dtype,
                                                        i,
                                                        batch_size=1,
                                                        tensor_shape=(1,)):
        self.assertFalse(_test_cuda_shared_memory,
                         "Shape tensors does not support CUDA shared memory")
        if _test_system_shared_memory:
            shm_region_handles = []
            for j, (shape_value, value) in enumerate(value_list):
                input_list = list()
                shape_input_list = list()
                dummy_input_list = list()

                for b in range(batch_size):
                    if dtype == np.object_:
                        dummy_in0 = np.full(tensor_shape, value, dtype=np.int32)
                        dummy_in0n = np.array(
                            [str(x) for x in dummy_in0.reshape(in0.size)],
                            dtype=object)
                        dummy_in0 = dummy_in0n.reshape(tensor_shape)
                    else:
                        dummy_in0 = np.full(tensor_shape, value, dtype=dtype)
                    dummy_input_list.append(dummy_in0)
                    in0 = np.full(tensor_shape, value, dtype=np.int32)
                    input_list.append(in0)

                # Only one shape tensor input per batch
                shape_input_list.append(
                    np.full(tensor_shape, shape_value, dtype=np.int32))

                if dtype == np.object_:
                    input_list_tmp = iu.serialize_byte_tensor_list(input_list)
                    input_byte_size = sum(
                        [serialized_byte_size(i0) for i0 in input_list_tmp])
                else:
                    input_list_tmp = input_list
                    input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])

                dummy_input_byte_size = sum(
                    [i0.nbytes for i0 in dummy_input_list])

                shape_input_byte_size = sum(
                    [i0.nbytes for i0 in shape_input_list])
                shape_output_byte_size = shape_input_byte_size
                output_byte_size = np.dtype(np.int32).itemsize + 2
                resized_output_byte_size = 32 * shape_value

                # create shared memory regions and copy data for input values
                ip_name = 'ip{}{}'.format(i, j)
                shape_ip_name = 'shape_ip{}{}'.format(i, j)
                dummy_ip_name = 'dummy_ip{}{}'.format(i, j)
                shape_op_name = 'shape_op{}{}'.format(i, j)
                op_name = 'op{}{}'.format(i, j)
                resized_op_name = 'resized_op{}{}'.format(i, j)

                shm_ip_handle = shm.create_shared_memory_region(
                    ip_name, '/' + ip_name, input_byte_size)
                shm_shape_ip_handle = shm.create_shared_memory_region(
                    shape_ip_name, '/' + shape_ip_name, shape_input_byte_size)
                shm_dummy_ip_handle = shm.create_shared_memory_region(
                    dummy_ip_name, '/' + dummy_ip_name, dummy_input_byte_size)
                shm_shape_op_handle = shm.create_shared_memory_region(
                    shape_op_name, '/' + shape_op_name, shape_output_byte_size)
                shm_op_handle = shm.create_shared_memory_region(
                    op_name, '/' + op_name, output_byte_size)
                shm_resized_op_handle = shm.create_shared_memory_region(
                    resized_op_name, '/' + resized_op_name,
                    resized_output_byte_size)
                shm.set_shared_memory_region(shm_ip_handle, input_list_tmp)
                shm.set_shared_memory_region(shm_shape_ip_handle,
                                             shape_input_list)
                shm.set_shared_memory_region(shm_dummy_ip_handle,
                                             dummy_input_list)
                self.triton_client_.register_system_shared_memory(
                    ip_name, '/' + ip_name, input_byte_size)
                self.triton_client_.register_system_shared_memory(
                    shape_ip_name, '/' + shape_ip_name, shape_input_byte_size)
                self.triton_client_.register_system_shared_memory(
                    dummy_ip_name, '/' + dummy_ip_name, dummy_input_byte_size)
                self.triton_client_.register_system_shared_memory(
                    shape_op_name, '/' + shape_op_name, shape_output_byte_size)
                self.triton_client_.register_system_shared_memory(
                    op_name, '/' + op_name, output_byte_size)
                self.triton_client_.register_system_shared_memory(
                    resized_op_name, '/' + resized_op_name,
                    resized_output_byte_size)

                shm_region_handles.append(
                    (ip_name, input_byte_size, shm_ip_handle))
                shm_region_handles.append(
                    (shape_ip_name, shape_input_byte_size, shm_shape_ip_handle))
                shm_region_handles.append(
                    (dummy_ip_name, dummy_input_byte_size, shm_dummy_ip_handle))
                shm_region_handles.append(
                    (shape_op_name, shape_output_byte_size,
                     shm_shape_op_handle))
                shm_region_handles.append(
                    (op_name, output_byte_size, shm_op_handle))
                shm_region_handles.append(
                    (resized_op_name, resized_output_byte_size,
                     shm_resized_op_handle))
            return shm_region_handles
        else:
            return []

    def cleanup_shm_regions(self, shm_handles):
        # Make sure unregister is before shared memory destruction
        if _test_system_shared_memory:
            self.triton_client_.unregister_system_shared_memory()
        if _test_cuda_shared_memory:
            self.triton_client_.unregister_cuda_shared_memory()
        for shm_tmp_handle in shm_handles:
            if _test_system_shared_memory:
                shm.destroy_shared_memory_region(shm_tmp_handle[2])
            elif _test_cuda_shared_memory:
                cudashm.destroy_shared_memory_region(shm_tmp_handle[2])

    def check_sequence(self,
                       trial,
                       model_name,
                       input_dtype,
                       correlation_id,
                       sequence_thresholds,
                       values,
                       expected_result,
                       protocol,
                       batch_size=1,
                       sequence_name="<unknown>",
                       tensor_shape=(1,)):
        """Perform sequence of inferences. The 'values' holds a list of
        tuples, one for each inference with format:

        (flag_str, value, (ls_ms, gt_ms), (pre_delay_ms, post_delay_ms)

        """
        if (("savedmodel" not in trial) and ("graphdef" not in trial) and
            ("custom" not in trial) and ("onnx" not in trial) and
            ("libtorch" not in trial) and ("plan" not in trial) and
            ("python" not in trial)):
            self.assertFalse(True, "unknown trial type: " + trial)

        # Can only send the request exactly once since it is a
        # sequence model with state, so can have only a single config.
        configs = []
        if protocol == "http":
            configs.append((f"{_tritonserver_ipaddr}:8000", "http", False))
        if protocol == "grpc":
            configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", False))
        if protocol == "streaming":
            configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", True))

        self.assertFalse(
            _test_system_shared_memory and _test_cuda_shared_memory,
            "Cannot set both System and CUDA shared memory flags to 1")

        self.assertEqual(len(configs), 1)

        full_shape = tensor_shape if "nobatch" in trial else (
            batch_size,) + tensor_shape

        # create and register shared memory output region in advance,
        # knowing that this function will not be called concurrently.
        if _test_system_shared_memory or _test_cuda_shared_memory:
            self.triton_client_.unregister_system_shared_memory()
            self.triton_client_.unregister_cuda_shared_memory()
            output_byte_size = 512
            if _test_system_shared_memory:
                shm_op_handle = shm.create_shared_memory_region(
                    "output_data", "/output", output_byte_size)
                self.triton_client_.register_system_shared_memory(
                    "output_data", "/output", output_byte_size)
            elif _test_cuda_shared_memory:
                shm_op_handle = cudashm.create_shared_memory_region(
                    "output_data", output_byte_size, 0)
                self.triton_client_.register_cuda_shared_memory(
                    "output_data", cudashm.get_raw_handle(shm_op_handle), 0,
                    output_byte_size)
            shm_ip_handles = []

        for config in configs:
            client_utils = grpcclient if config[1] == "grpc" else httpclient

            triton_client = client_utils.InferenceServerClient(config[0],
                                                               verbose=True)
            if config[2]:
                user_data = UserData()
                triton_client.start_stream(
                    partial(completion_callback, user_data))
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))

                INPUT = "INPUT__0" if trial.startswith("libtorch") else "INPUT"
                OUTPUT = "OUTPUT__0" if trial.startswith(
                    "libtorch") else "OUTPUT"
                for flag_str, value, thresholds, delay_ms in values:
                    if _test_valgrind or _test_jetson:
                        if delay_ms is not None:
                            delay_ms[0] = max(_valgrind_delay_ms, delay_ms[0])
                            delay_ms[1] = max(_valgrind_delay_ms, delay_ms[1])
                        else:
                            delay_ms = (_valgrind_delay_ms, _valgrind_delay_ms)

                    if delay_ms is not None:
                        time.sleep(delay_ms[0] / 1000.0)

                    seq_start = False
                    seq_end = False
                    if flag_str is not None:
                        seq_start = ("start" in flag_str)
                        seq_end = ("end" in flag_str)

                    # Construct request IOs
                    inputs = []
                    outputs = []
                    inputs.append(
                        client_utils.InferInput(
                            INPUT, full_shape, np_to_triton_dtype(input_dtype)))
                    outputs.append(client_utils.InferRequestedOutput(OUTPUT))
                    if input_dtype == np.object_:
                        in0 = np.full(full_shape, value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)],
                                        dtype=object)
                        in0 = in0n.reshape(full_shape)
                    else:
                        in0 = np.full(full_shape, value, dtype=input_dtype)

                    # create input shared memory and copy input data values into it
                    if _test_system_shared_memory or _test_cuda_shared_memory:
                        if input_dtype == np.object_:
                            input_list_tmp = iu.serialize_byte_tensor_list(
                                [in0])
                            input_byte_size = sum([
                                serialized_byte_size(i0)
                                for i0 in input_list_tmp
                            ])
                        else:
                            input_list_tmp = [in0]
                            input_byte_size = sum(
                                [i0.nbytes for i0 in input_list_tmp])
                        ip_name = "ip{}".format(len(shm_ip_handles))
                        if _test_system_shared_memory:
                            shm_ip_handles.append(
                                shm.create_shared_memory_region(
                                    ip_name, "/" + ip_name, input_byte_size))
                            shm.set_shared_memory_region(
                                shm_ip_handles[-1], input_list_tmp)
                            triton_client.register_system_shared_memory(
                                ip_name, "/" + ip_name, input_byte_size)
                        elif _test_cuda_shared_memory:
                            shm_ip_handles.append(
                                cudashm.create_shared_memory_region(
                                    ip_name, input_byte_size, 0))
                            cudashm.set_shared_memory_region(
                                shm_ip_handles[-1], input_list_tmp)
                            triton_client.register_cuda_shared_memory(
                                ip_name,
                                cudashm.get_raw_handle(shm_ip_handles[-1]), 0,
                                input_byte_size)

                        inputs[0].set_shared_memory(ip_name, input_byte_size)
                        outputs[0].set_shared_memory("output_data",
                                                     output_byte_size)
                    else:
                        inputs[0].set_data_from_numpy(in0)

                    start_ms = int(round(time.time() * 1000))

                    if config[2]:
                        triton_client.async_stream_infer(
                            model_name,
                            inputs,
                            outputs=outputs,
                            sequence_id=correlation_id,
                            sequence_start=seq_start,
                            sequence_end=seq_end)
                        (results, error) = user_data._completed_requests.get()
                        if error is not None:
                            raise error
                    else:
                        results = triton_client.infer(
                            model_name,
                            inputs,
                            outputs=outputs,
                            sequence_id=correlation_id,
                            sequence_start=seq_start,
                            sequence_end=seq_end)

                    end_ms = int(round(time.time() * 1000))

                    # Get value of "OUTPUT", for shared memory, need to get it via
                    # shared memory utils
                    if (not _test_system_shared_memory) and (
                            not _test_cuda_shared_memory):
                        out = results.as_numpy(OUTPUT)
                    else:
                        output = results.get_output(OUTPUT)
                        if config[1] == "http":
                            output_shape = output["shape"]
                        else:
                            output_shape = output.shape
                        output_type = input_dtype
                        if _test_system_shared_memory:
                            out = shm.get_contents_as_numpy(
                                shm_op_handle, output_type, output_shape)
                        else:
                            out = cudashm.get_contents_as_numpy(
                                shm_op_handle, output_type, output_shape)
                    result = out[0] if "nobatch" in trial else out[0][0]
                    print("{}: {}".format(sequence_name, result))

                    if thresholds is not None:
                        lt_ms = thresholds[0]
                        gt_ms = thresholds[1]
                        if lt_ms is not None:
                            self.assertTrue((end_ms - start_ms) < lt_ms,
                                            "expected less than " + str(lt_ms) +
                                            "ms response time, got " +
                                            str(end_ms - start_ms) + " ms")
                        if gt_ms is not None:
                            self.assertTrue(
                                (end_ms - start_ms) > gt_ms,
                                "expected greater than " + str(gt_ms) +
                                "ms response time, got " +
                                str(end_ms - start_ms) + " ms")
                    if delay_ms is not None:
                        time.sleep(delay_ms[1] / 1000.0)

                seq_end_ms = int(round(time.time() * 1000))

                if input_dtype == np.object_:
                    self.assertEqual(int(result), expected_result)
                else:
                    self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        if _test_jetson:
                            lt_ms *= _jetson_slowdown_factor
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " +
                                        str(lt_ms) + "ms response time, got " +
                                        str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " +
                                        str(gt_ms) + "ms response time, got " +
                                        str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                self.add_deferred_exception(ex)
            if config[2]:
                triton_client.stop_stream()

        if _test_system_shared_memory or _test_cuda_shared_memory:
            self.triton_client_.unregister_system_shared_memory()
            self.triton_client_.unregister_cuda_shared_memory()
            destroy_func = shm.destroy_shared_memory_region if _test_system_shared_memory else cudashm.destroy_shared_memory_region
            destroy_func(shm_op_handle)
            for shm_ip_handle in shm_ip_handles:
                destroy_func(shm_ip_handle)

    def check_sequence_async(self,
                             trial,
                             model_name,
                             input_dtype,
                             correlation_id,
                             sequence_thresholds,
                             values,
                             expected_result,
                             shm_region_handles,
                             batch_size=1,
                             sequence_name="<unknown>",
                             tensor_shape=(1,)):
        """Perform sequence of inferences using stream async run.
        The 'values' holds a list of tuples, one for each inference with format:

        (flag_str, value, pre_delay_ms)

        """
        if (("savedmodel" not in trial) and ("graphdef" not in trial) and
            ("custom" not in trial) and ("onnx" not in trial) and
            ("libtorch" not in trial) and ("plan" not in trial) and
            ("python" not in trial)):
            self.assertFalse(True, "unknown trial type: " + trial)

        self.assertFalse(
            _test_system_shared_memory and _test_cuda_shared_memory,
            "Cannot set both System and CUDA shared memory flags to 1")

        full_shape = tensor_shape if "nobatch" in trial else (
            batch_size,) + tensor_shape

        client_utils = grpcclient
        triton_client = client_utils.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001", verbose=True)
        user_data = UserData()
        triton_client.start_stream(partial(completion_callback, user_data))
        # Execute the sequence of inference...
        try:
            seq_start_ms = int(round(time.time() * 1000))

            INPUT = "INPUT__0" if trial.startswith("libtorch") else "INPUT"
            OUTPUT = "OUTPUT__0" if trial.startswith("libtorch") else "OUTPUT"
            sent_count = 0
            for flag_str, value, pre_delay_ms in values:
                seq_start = False
                seq_end = False
                if flag_str is not None:
                    seq_start = ("start" in flag_str)
                    seq_end = ("end" in flag_str)

                # Construct request IOs
                inputs = []
                outputs = []
                inputs.append(
                    client_utils.InferInput(INPUT, full_shape,
                                            np_to_triton_dtype(input_dtype)))
                outputs.append(client_utils.InferRequestedOutput(OUTPUT))

                if not (_test_system_shared_memory or _test_cuda_shared_memory):
                    if input_dtype == np.object_:
                        in0 = np.full(full_shape, value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)],
                                        dtype=object)
                        in0 = in0n.reshape(full_shape)
                    else:
                        in0 = np.full(full_shape, value, dtype=input_dtype)
                    inputs[0].set_data_from_numpy(in0)
                else:
                    offset = 2 * sent_count
                    inputs[0].set_shared_memory(shm_region_handles[offset][0],
                                                shm_region_handles[offset][1])
                    outputs[0].set_shared_memory(
                        shm_region_handles[offset + 1][0],
                        shm_region_handles[offset + 1][1])

                if pre_delay_ms is not None:
                    time.sleep(pre_delay_ms / 1000.0)

                triton_client.async_stream_infer(model_name,
                                                 inputs,
                                                 outputs=outputs,
                                                 sequence_id=correlation_id,
                                                 sequence_start=seq_start,
                                                 sequence_end=seq_end)
                sent_count += 1

            # Wait for the results in the order sent
            result = None
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                if error is not None:
                    raise error
                # Get value of "OUTPUT", for shared memory, need to get it via
                # shared memory utils
                if (not _test_system_shared_memory) and (
                        not _test_cuda_shared_memory):
                    out = results.as_numpy(OUTPUT)
                else:
                    output = results.get_output(OUTPUT)
                    offset = 2 * processed_count + 1
                    output_shape = output.shape
                    output_type = input_dtype
                    if _test_system_shared_memory:
                        out = shm.get_contents_as_numpy(
                            shm_region_handles[offset][2], output_type,
                            output_shape)
                    else:
                        out = cudashm.get_contents_as_numpy(
                            shm_region_handles[offset][2], output_type,
                            output_shape)
                result = out[0] if "nobatch" in trial else out[0][0]
                print("{}: {}".format(sequence_name, result))
                processed_count += 1

            seq_end_ms = int(round(time.time() * 1000))

            if input_dtype == np.object_:
                self.assertEqual(int(result), expected_result)
            else:
                self.assertEqual(result, expected_result)

            if sequence_thresholds is not None:
                lt_ms = sequence_thresholds[0]
                gt_ms = sequence_thresholds[1]
                if lt_ms is not None:
                    if _test_jetson:
                        lt_ms *= _jetson_slowdown_factor
                    self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                    "sequence expected less than " +
                                    str(lt_ms) + "ms response time, got " +
                                    str(seq_end_ms - seq_start_ms) + " ms")
                if gt_ms is not None:
                    self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                    "sequence expected greater than " +
                                    str(gt_ms) + "ms response time, got " +
                                    str(seq_end_ms - seq_start_ms) + " ms")
        except Exception as ex:
            self.add_deferred_exception(ex)
        triton_client.stop_stream()

    # This sequence util only sends inference via streaming scenario
    def check_sequence_shape_tensor_io(self,
                                       model_name,
                                       input_dtype,
                                       correlation_id,
                                       sequence_thresholds,
                                       values,
                                       expected_result,
                                       shm_region_handles,
                                       using_dynamic_batcher=False,
                                       sequence_name="<unknown>"):
        """Perform sequence of inferences using async run. The 'values' holds
        a list of tuples, one for each inference with format:

        (flag_str, shape_value, value, pre_delay_ms)

        """
        tensor_shape = (1, 1)
        # shape tensor is 1-D tensor that doesn't contain batch size as first value
        shape_tensor_shape = (1,)
        self.assertFalse(_test_cuda_shared_memory,
                         "Shape tensors does not support CUDA shared memory")

        client_utils = grpcclient
        triton_client = client_utils.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001", verbose=True)
        user_data = UserData()
        triton_client.start_stream(partial(completion_callback, user_data))
        # Execute the sequence of inference...
        try:
            seq_start_ms = int(round(time.time() * 1000))

            sent_count = 0
            shape_values = list()
            for flag_str, shape_value, value, pre_delay_ms in values:
                seq_start = False
                seq_end = False
                if flag_str is not None:
                    seq_start = ("start" in flag_str)
                    seq_end = ("end" in flag_str)

                # Construct request IOs
                inputs = []
                outputs = []
                # input order: input, shape(, dummy)
                inputs.append(
                    client_utils.InferInput(
                        "INPUT", tensor_shape,
                        np_to_triton_dtype(np.int32 if using_dynamic_batcher
                                           else input_dtype)))
                inputs.append(
                    client_utils.InferInput("SHAPE_INPUT", shape_tensor_shape,
                                            np_to_triton_dtype(np.int32)))
                if using_dynamic_batcher:
                    inputs.append(
                        client_utils.InferInput(
                            "DUMMY_INPUT", tensor_shape,
                            np_to_triton_dtype(input_dtype)))
                # output order: shape, output, resized
                outputs.append(
                    client_utils.InferRequestedOutput("SHAPE_OUTPUT"))
                outputs.append(client_utils.InferRequestedOutput("OUTPUT"))
                outputs.append(
                    client_utils.InferRequestedOutput("RESIZED_OUTPUT"))

                # Set IO values
                shape_values.append(
                    np.full(shape_tensor_shape, shape_value, dtype=np.int32))
                if not _test_system_shared_memory:
                    if using_dynamic_batcher:
                        if input_dtype == np.object_:
                            dummy_in0 = np.full(tensor_shape,
                                                value,
                                                dtype=np.int32)
                            dummy_in0n = np.array(
                                [str(x) for x in in0.reshape(dummy_in0.size)],
                                dtype=object)
                            dummy_in0 = dummy_in0n.reshape(tensor_shape)
                        else:
                            dummy_in0 = np.full(tensor_shape,
                                                value,
                                                dtype=input_dtype)
                        in0 = np.full(tensor_shape, value, dtype=np.int32)
                    else:
                        if input_dtype == np.object_:
                            in0 = np.full(tensor_shape, value, dtype=np.int32)
                            in0n = np.array(
                                [str(x) for x in in0.reshape(in0.size)],
                                dtype=object)
                            in0 = in0n.reshape(tensor_shape)
                        else:
                            in0 = np.full(tensor_shape,
                                          value,
                                          dtype=input_dtype)

                    inputs[0].set_data_from_numpy(in0)
                    inputs[1].set_data_from_numpy(shape_values[-1])
                    if using_dynamic_batcher:
                        inputs[2].set_data_from_numpy(dummy_in0)
                else:
                    if using_dynamic_batcher:
                        input_offset = 6 * sent_count
                        output_offset = 6 * sent_count + 3
                    else:
                        input_offset = 5 * sent_count
                        output_offset = 5 * sent_count + 2
                    for i in range(len(inputs)):
                        inputs[i].set_shared_memory(
                            shm_region_handles[input_offset + i][0],
                            shm_region_handles[input_offset + i][1])
                    for i in range(len(outputs)):
                        outputs[i].set_shared_memory(
                            shm_region_handles[output_offset + i][0],
                            shm_region_handles[output_offset + i][1])

                if pre_delay_ms is not None:
                    time.sleep(pre_delay_ms / 1000.0)

                triton_client.async_stream_infer(model_name,
                                                 inputs,
                                                 outputs=outputs,
                                                 sequence_id=correlation_id,
                                                 sequence_start=seq_start,
                                                 sequence_end=seq_end)

                sent_count += 1

            # Wait for the results in the order sent
            result = None
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                if error is not None:
                    raise error
                # Get value of "OUTPUT", for shared memory, need to get it via
                # shared memory utils
                if (not _test_system_shared_memory):
                    out = results.as_numpy("OUTPUT")
                else:
                    output = results.get_output("OUTPUT")
                    output_offset = 6 * processed_count + 4 if using_dynamic_batcher else 5 * processed_count + 3
                    output_shape = output.shape
                    output_type = np.int32 if using_dynamic_batcher else np.float32
                    out = shm.get_contents_as_numpy(
                        shm_region_handles[output_offset][2], output_type,
                        output_shape)
                result = out[0][0]

                # Validate the (debatched) shape of the resized output matches
                # with the shape input values
                resized_shape = results.get_output("RESIZED_OUTPUT").shape[1:]
                self.assertTrue(
                    np.array_equal(resized_shape,
                                   shape_values[processed_count]),
                    "{}, {}, slot {}, expected: {}, got {}".format(
                        model_name, "RESIZED_OUTPUT", processed_count,
                        shape_values[processed_count], resized_shape))
                print("{}: {}".format(sequence_name, result))
                processed_count += 1

            seq_end_ms = int(round(time.time() * 1000))

            if input_dtype == np.object_:
                self.assertEqual(int(result), expected_result)
            else:
                self.assertEqual(result, expected_result)

            if sequence_thresholds is not None:
                lt_ms = sequence_thresholds[0]
                gt_ms = sequence_thresholds[1]
                if lt_ms is not None:
                    if _test_jetson:
                        lt_ms *= _jetson_slowdown_factor
                    self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                    "sequence expected less than " +
                                    str(lt_ms) + "ms response time, got " +
                                    str(seq_end_ms - seq_start_ms) + " ms")
                if gt_ms is not None:
                    self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                    "sequence expected greater than " +
                                    str(gt_ms) + "ms response time, got " +
                                    str(seq_end_ms - seq_start_ms) + " ms")
        except Exception as ex:
            self.add_deferred_exception(ex)
        triton_client.stop_stream()

    def check_setup(self, model_name):
        # Make sure test.sh set up the correct batcher settings
        config = self.triton_client_.get_model_config(model_name).config
        # Skip the sequence batching check on ensemble model
        if config.platform != "ensemble":
            bconfig = config.sequence_batching
            self.assertEqual(bconfig.max_sequence_idle_microseconds,
                             _max_sequence_idle_ms * 1000)  # 5 secs

    def check_status(self, model_name, batch_exec, exec_cnt, infer_cnt):
        stats = self.triton_client_.get_inference_statistics(model_name, "1")
        self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
        self.assertEqual(stats.model_stats[0].name, model_name,
                         "expect model stats for model {}".format(model_name))
        self.assertEqual(
            stats.model_stats[0].version, "1",
            "expect model stats for model {} version 1".format(model_name))

        if batch_exec is not None:
            batch_stats = stats.model_stats[0].batch_stats
            print(batch_stats)
            self.assertEqual(
                len(batch_stats), len(batch_exec),
                "expected {} different batch-sizes, got {}".format(
                    len(batch_exec), len(batch_stats)))

            for batch_stat in batch_stats:
                bs = batch_stat.batch_size
                bc = batch_stat.compute_infer.count
                self.assertTrue(
                    bs in batch_exec,
                    "did not find expected batch-size {}".format(bs))
                # Get count from one of the stats
                self.assertEqual(
                    bc, batch_exec[bs],
                    "expected model-execution-count {} for batch size {}, got {}"
                    .format(batch_exec[bs], bs, bc))

        actual_exec_cnt = stats.model_stats[0].execution_count
        self.assertEqual(
            actual_exec_cnt, exec_cnt,
            "expected model-exec-count {}, got {}".format(
                exec_cnt, actual_exec_cnt))

        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt, infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt))
