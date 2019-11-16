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

_trials = ("custom", "savedmodel", "graphdef", "netdef", "plan", "onnx", "libtorch")
_protocols = ("http", "grpc")
_max_sequence_idle_ms = 5000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = None

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_run()
def completion_callback(user_data, infer_ctx, request_id):
    user_data._completed_requests.put(request_id)


class DynaSequenceBatcherTest(unittest.TestCase):
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

    def precreate_register_regions(self, value_list, dtype, i, batch_size=1):
        if _test_system_shared_memory or _test_cuda_shared_memory:
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=True)
            shm_region_handles = []
            for j, value in enumerate(value_list):
                # create data
                input_list = list()
                for b in range(batch_size):
                    if dtype == np.object:
                        in0 = np.full((1,), value, dtype=np.int32)
                        in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                        in0 = in0n.reshape((1,))
                    else:
                        in0 = np.full((1,), value, dtype=dtype)
                    input_list.append(in0)

                input_list_tmp = iu._prepend_string_size(input_list) if (dtype == np.object) else input_list
                input_byte_size = sum([i0.nbytes for i0 in input_list_tmp])
                output_byte_size = np.dtype(dtype).itemsize + 2

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
                       protocol, batch_size=1, sequence_name="<unknown>"):
        """Perform sequence of inferences. The 'values' holds a list of
        tuples, one for each inference with format:

        (flag_str, value, (ls_ms, gt_ms), (pre_delay_ms, post_delay_ms)

        """
        if (("savedmodel" in trial) or ("graphdef" in trial) or
            ("netdef" in trial) or ("custom" in trial) or
            ("onnx" in trial) or ("libtorch" in trial) or
	        ("plan" in trial)):
            tensor_shape = (1,)
        else:
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
            shared_memory_ctx = SharedMemoryControlContext("localhost:8000",  ProtocolType.HTTP, verbose=True)
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
                    if "libtorch" not in trial:
                        results = ctx.run(
                            { "INPUT" : input_info }, { "OUTPUT" : output_info},
                            batch_size=batch_size, flags=flags)
                        OUTPUT = "OUTPUT"
                    else:
                        results = ctx.run(
                            { "INPUT__0" : input_info }, { "OUTPUT__0" : output_info},
                            batch_size=batch_size, flags=flags)
                        OUTPUT = "OUTPUT__0"

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
                             protocol, shm_region_handles, batch_size=1, sequence_name="<unknown>"):
        """Perform sequence of inferences using async run. The 'values' holds
        a list of tuples, one for each inference with format:

        (flag_str, value, pre_delay_ms)

        """
        if (("savedmodel" in trial) or ("graphdef" in trial) or
            ("netdef" in trial) or ("custom" in trial) or
            ("onnx" in trial) or ("libtorch" in trial) or
            ("plan" in trial)):
            tensor_shape = (1,)
        else:
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

                    if "libtorch" not in trial:
                        ctx.async_run(partial(completion_callback, user_data),
                            { 'INPUT' :input_info }, { 'OUTPUT' :output_info },
                               batch_size=batch_size, flags=flags)
                        OUTPUT = "OUTPUT"
                    else:
                        ctx.async_run(partial(completion_callback, user_data),
                            { 'INPUT__0' :input_info }, { 'OUTPUT__0' :output_info },
                               batch_size=batch_size, flags=flags)
                        OUTPUT = "OUTPUT__0"
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

    def check_setup(self, model_name):
        # Make sure test.sh set up the correct batcher settings
        ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
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

        self.assertEqual(vs[1].model_execution_count, exec_cnt,
                         "expected model-execution-count " + str(exec_cnt) + ", got " +
                         str(vs[1].model_execution_count))
        self.assertEqual(vs[1].model_inference_count, infer_cnt,
                         "expected model-inference-count " + str(infer_cnt) + ", got " +
                         str(vs[1].model_inference_count))

    def get_datatype(self, trial):
        return np.int32

    def get_expected_result(self, expected_result, corrid, value, trial, flag_str=None):
        # Adjust the expected_result for models that
        # couldn't implement the full accumulator. See
        # qa/common/gen_qa_dyna_sequence_models.py for more
        # information.
        if "custom" not in trial:
            expected_result = value
            if flag_str is not None:
                if "start" in flag_str:
                    expected_result += 1
                if "end" in flag_str:
                    expected_result += corrid
        return expected_result

    def test_simple_sequence(self):
        # Send one sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                self.clear_deferred_exceptions()
                try:
                    dtype = self.get_datatype(trial)
                    model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    corrid = 52
                    self.check_sequence(trial, model_name, dtype, corrid,
                                        (4000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         (None, 2, None, None),
                                         (None, 3, None, None),
                                         (None, 4, None, None),
                                         (None, 5, None, None),
                                         (None, 6, None, None),
                                         (None, 7, None, None),
                                         (None, 8, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(45 + corrid, corrid, 9, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), 9 * (idx + 1), 9 * (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_length1_sequence(self):
        # Send a length-1 sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                self.clear_deferred_exceptions()
                try:
                    dtype = self.get_datatype(trial)
                    model_name = tu.get_dyna_sequence_model_name(trial, dtype)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    corrid = 99
                    self.check_sequence(trial, model_name, dtype, corrid,
                                        (4000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start,end", 42, None, None),),
                                        self.get_expected_result(42 + corrid, corrid, 42,
                                                                 trial, "start,end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), (idx + 1), (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_sequence(self):
        # Send four sequences in parallel and make sure they get
        # completely batched into batch-size 4 inferences.
        for trial in _trials:
            self.clear_deferred_exceptions()
            dtype = self.get_datatype(trial)
            precreated_shm0_handles = self.precreate_register_regions((1,2,3), dtype, 0)
            precreated_shm1_handles = self.precreate_register_regions((11,12,13), dtype, 1)
            precreated_shm2_handles = self.precreate_register_regions((111,112,113), dtype, 2)
            precreated_shm3_handles = self.precreate_register_regions((1111,1112,1113), dtype, 3)
            try:
                model_name = tu.get_dyna_sequence_model_name(trial, dtype)
                protocol = "streaming"

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for all sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 11)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                corrids = [ 1001, 1002, 1003, 1004 ]
                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[0],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           ("end", 3, None)),
                          self.get_expected_result(4 + corrids[0], corrids[0], 3, trial, "end"),
                          protocol, precreated_shm0_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[0])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[1],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(36 + corrids[1], corrids[1], 13, trial, "end"),
                          protocol, precreated_shm1_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[1])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[2],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(336 + corrids[2], corrids[2], 113, trial, "end"),
                          protocol, precreated_shm2_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[2])}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, dtype, corrids[3],
                          (None, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336 + corrids[3], corrids[3], 1113, trial, "end"),
                          protocol, precreated_shm3_handles),
                    kwargs={'sequence_name' : "{}_{}_{}".format(
                      self._testMethodName, protocol, corrids[3])}))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), 3, 11)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if _test_system_shared_memory or _test_cuda_shared_memory:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)


if __name__ == '__main__':
    unittest.main()
