# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import json
import os
import queue
import shutil
import unittest

import numpy
from tritonserver import _c as triton_bindings


# Callback functions used in inference pipeline
# 'user_object' is a per-request counter of how many times the
# callback is invoked
def g_alloc_fn(
    allocator, tensor_name, byte_size, memory_type, memory_type_id, user_object
):
    if "alloc" not in user_object:
        user_object["alloc"] = 0
    user_object["alloc"] += 1
    buffer = numpy.empty(byte_size, numpy.byte)
    return (buffer.ctypes.data, buffer, triton_bindings.TRITONSERVER_MemoryType.CPU, 0)


def g_release_fn(
    allocator, buffer, buffer_user_object, byte_size, memory_type, memory_type_id
):
    # No-op, buffer ('buffer_user_object') will be garbage collected
    # only sanity check that the objects are expected
    if (not isinstance(buffer_user_object, numpy.ndarray)) or (
        buffer_user_object.ctypes.data != buffer
    ):
        raise Exception("Misaligned parameters in allocator release callback")
    pass


def g_start_fn(allocator, user_object):
    if "start" not in user_object:
        user_object["start"] = 0
    user_object["start"] += 1
    pass


def g_query_fn(
    allocator, user_object, tensor_name, byte_size, memory_type, memory_type_id
):
    if "query" not in user_object:
        user_object["query"] = 0
    user_object["query"] += 1
    return (triton_bindings.TRITONSERVER_MemoryType.CPU, 0)


def g_buffer_fn(
    allocator, tensor_name, buffer_attribute, user_object, buffer_user_object
):
    if "buffer" not in user_object:
        user_object["buffer"] = 0
    user_object["buffer"] += 1
    buffer_attribute.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
    buffer_attribute.memory_type_id = 0
    buffer_attribute.byte_size = buffer_user_object.size
    return buffer_attribute


def g_timestamp_fn(trace, activity, timestamp_ns, user_object):
    if trace.id not in user_object:
        user_object[trace.id] = []
    # not owning trace, so must read property out
    trace_log = {
        "id": trace.id,
        "parent_id": trace.parent_id,
        "model_name": trace.model_name,
        "model_version": trace.model_version,
        "request_id": trace.request_id,
        "activity": activity,
        "timestamp": timestamp_ns,
    }
    user_object[trace.id].append(trace_log)


def g_tensor_fn(
    trace,
    activity,
    tensor_name,
    data_type,
    buffer,
    byte_size,
    shape,
    memory_type,
    memory_type_id,
    user_object,
):
    if trace.id not in user_object:
        user_object[trace.id] = []

    # not owning trace, so must read property out
    trace_log = {
        "id": trace.id,
        "parent_id": trace.parent_id,
        "model_name": trace.model_name,
        "model_version": trace.model_version,
        "request_id": trace.request_id,
        "activity": activity,
        "tensor": {
            "name": tensor_name,
            "data_type": data_type,
            # skip 'buffer'
            "byte_size": byte_size,
            "shape": shape,
            "memory_type": memory_type,
            "memory_type_id": memory_type_id,
        },
    }
    user_object[trace.id].append(trace_log)


def g_trace_release_fn(trace, user_object):
    # sanity check that 'trace' has been tracked, the object
    # will be released on garbage collection
    if trace.id not in user_object:
        raise Exception("Releasing unseen trace")
    user_object["signal_queue"].put("TRACE_RELEASED")


def g_response_fn(response, flags, user_object):
    user_object.put((flags, response))


def g_request_fn(request, flags, user_object):
    if flags != 1:
        raise Exception("Unexpected request release flag")
    # counter of "inflight" requests
    user_object.put(request)


# Python model file string to fastly deploy test model, depends on
# 'TRITONSERVER_Server' operates properly to load model with content passed
# through the load API.
g_python_addsub = b'''
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input0 = {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        input1 = {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]}
        output0 = {"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        output1 = {"name": "OUTPUT1", "data_type": "TYPE_FP32", "dims": [4]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input0)
        auto_complete_model_config.add_input(input1)
        auto_complete_model_config.add_output(output0)
        auto_complete_model_config.add_output(output1)

        # [WARNING] Specify specific dynamic batching field by knowing
        # the implementation detail
        auto_complete_model_config.set_dynamic_batching()
        auto_complete_model_config._model_config["dynamic_batching"]["priority_levels"] = 20
        auto_complete_model_config._model_config["dynamic_batching"]["default_priority_level"] = 10

        return auto_complete_model_config

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

    def execute(self, requests):
        """This function is called on inference request."""

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            out_0, out_1 = (
                in_0.as_numpy() + in_1.as_numpy(),
                in_0.as_numpy() - in_1.as_numpy(),
            )

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1.astype(output1_dtype))
            responses.append(pb_utils.InferenceResponse([out_tensor_0, out_tensor_1]))
        return responses
'''


# ======================================= Test cases ===========================
class BindingTest(unittest.TestCase):
    def setUp(self):
        self._test_model_repo = os.path.join(os.getcwd(), "binding_test_repo")
        # clear model repository that may be created for testing.
        if os.path.exists(self._test_model_repo):
            shutil.rmtree(self._test_model_repo)
        os.makedirs(self._test_model_repo)
        self._model_name = "addsub"
        self._version = "1"
        self._file_name = "model.py"

    def tearDown(self):
        gc.collect()
        # clear model repository that may be created for testing.
        if os.path.exists(self._test_model_repo):
            shutil.rmtree(self._test_model_repo)

    # helper functions
    def _to_pyobject(self, triton_message):
        return json.loads(triton_message.serialize_to_json())

    # prepare a model repository with "addsub" model
    def _create_model_repository(self):
        os.makedirs(
            os.path.join(self._test_model_repo, self._model_name, self._version)
        )
        with open(
            os.path.join(
                self._test_model_repo, self._model_name, self._version, self._file_name
            ),
            "wb",
        ) as f:
            f.write(g_python_addsub)

    # create a Triton instance with POLL mode on repository prepared by
    # '_create_model_repository'
    def _start_polling_server(self):
        # prepare model repository
        self._create_model_repository()

        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path(self._test_model_repo)
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.POLL
        )
        # enable "auto-complete" to skip providing config.pbtxt
        options.set_strict_model_config(False)
        options.set_server_id("testing_server")
        # [FIXME] Need to fix coupling of response and server
        options.set_exit_timeout(5)
        return triton_bindings.TRITONSERVER_Server(options)

    def _prepare_inference_request(self, server):
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            g_alloc_fn, g_release_fn, g_start_fn
        )
        allocator.set_buffer_attributes_function(g_buffer_fn)
        allocator.set_query_function(g_query_fn)

        request_counter = queue.Queue()
        response_queue = queue.Queue()
        allocator_counter = {}
        request = triton_bindings.TRITONSERVER_InferenceRequest(
            server, self._model_name, -1
        )
        request.id = "req_0"
        request.set_release_callback(g_request_fn, request_counter)
        request.set_response_callback(
            allocator, allocator_counter, g_response_fn, response_queue
        )

        input = numpy.ones([4], dtype=numpy.float32)
        input_buffer = input.ctypes.data
        ba = triton_bindings.TRITONSERVER_BufferAttributes()
        ba.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        ba.memory_type_id = 0
        ba.byte_size = input.itemsize * input.size

        request.add_input(
            "INPUT0", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        request.add_input(
            "INPUT1", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        request.append_input_data_with_buffer_attributes("INPUT0", input_buffer, ba)
        request.append_input_data_with_buffer_attributes("INPUT1", input_buffer, ba)

        return request, allocator, response_queue, request_counter

    def test_exceptions(self):
        ex_list = [
            triton_bindings.UnknownError,
            triton_bindings.InternalError,
            triton_bindings.NotFoundError,
            triton_bindings.InvalidArgumentError,
            triton_bindings.UnavailableError,
            triton_bindings.UnsupportedError,
            triton_bindings.AlreadyExistsError,
        ]
        for ex_type in ex_list:
            with self.assertRaises(triton_bindings.TritonError) as ctx:
                raise ex_type("Error message")
            self.assertTrue(isinstance(ctx.exception, ex_type))
            self.assertEqual(str(ctx.exception), "Error message")

    def test_data_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_DataType.INVALID, "<invalid>", 0),
            (triton_bindings.TRITONSERVER_DataType.BOOL, "BOOL", 1),
            (triton_bindings.TRITONSERVER_DataType.UINT8, "UINT8", 1),
            (triton_bindings.TRITONSERVER_DataType.UINT16, "UINT16", 2),
            (triton_bindings.TRITONSERVER_DataType.UINT32, "UINT32", 4),
            (triton_bindings.TRITONSERVER_DataType.UINT64, "UINT64", 8),
            (triton_bindings.TRITONSERVER_DataType.INT8, "INT8", 1),
            (triton_bindings.TRITONSERVER_DataType.INT16, "INT16", 2),
            (triton_bindings.TRITONSERVER_DataType.INT32, "INT32", 4),
            (triton_bindings.TRITONSERVER_DataType.INT64, "INT64", 8),
            (triton_bindings.TRITONSERVER_DataType.FP16, "FP16", 2),
            (triton_bindings.TRITONSERVER_DataType.FP32, "FP32", 4),
            (triton_bindings.TRITONSERVER_DataType.FP64, "FP64", 8),
            (triton_bindings.TRITONSERVER_DataType.BYTES, "BYTES", 0),
            (triton_bindings.TRITONSERVER_DataType.BF16, "BF16", 2),
        ]

        for t, t_str, t_size in t_list:
            self.assertEqual(triton_bindings.TRITONSERVER_DataTypeString(t), t_str)
            self.assertEqual(triton_bindings.TRITONSERVER_StringToDataType(t_str), t)
            self.assertEqual(triton_bindings.TRITONSERVER_DataTypeByteSize(t), t_size)

    def test_memory_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_MemoryType.CPU, "CPU"),
            (triton_bindings.TRITONSERVER_MemoryType.CPU_PINNED, "CPU_PINNED"),
            (triton_bindings.TRITONSERVER_MemoryType.GPU, "GPU"),
        ]
        for t, t_str in t_list:
            self.assertEqual(triton_bindings.TRITONSERVER_MemoryTypeString(t), t_str)

    def test_parameter_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_ParameterType.STRING, "STRING"),
            (triton_bindings.TRITONSERVER_ParameterType.INT, "INT"),
            (triton_bindings.TRITONSERVER_ParameterType.BOOL, "BOOL"),
            (triton_bindings.TRITONSERVER_ParameterType.BYTES, "BYTES"),
        ]
        for t, t_str in t_list:
            self.assertEqual(triton_bindings.TRITONSERVER_ParameterTypeString(t), t_str)

    def test_parameter(self):
        # C API doesn't provide additional API for parameter, can only test
        # New/Delete unless we mock the implementation to expose more info.
        str_param = triton_bindings.TRITONSERVER_Parameter("str_key", "str_value")
        int_param = triton_bindings.TRITONSERVER_Parameter("int_key", 123)
        bool_param = triton_bindings.TRITONSERVER_Parameter("bool_key", True)
        # bytes parameter doesn't own the buffer
        b = bytes("abc", "utf-8")
        bytes_param = triton_bindings.TRITONSERVER_Parameter("bytes_key", b)
        del str_param
        del int_param
        del bool_param
        del bytes_param
        gc.collect()

    def test_instance_kind(self):
        t_list = [
            (triton_bindings.TRITONSERVER_InstanceGroupKind.AUTO, "AUTO"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.CPU, "CPU"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.GPU, "GPU"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.MODEL, "MODEL"),
        ]
        for t, t_str in t_list:
            self.assertEqual(
                triton_bindings.TRITONSERVER_InstanceGroupKindString(t), t_str
            )

    def test_log(self):
        # This test depends on 'TRITONSERVER_ServerOptions' operates properly
        # to modify log settings.

        # Direct Triton to log message into a file so that the log may be
        # retrieved on the Python side. Otherwise the log will be default
        # on stderr and Python utils can not redirect the pipe on Triton side.
        log_file = "triton_binding_test_log_output.txt"
        default_format_regex = r"[0-9][0-9][0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]"
        iso8601_format_regex = r"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z"
        try:
            options = triton_bindings.TRITONSERVER_ServerOptions()
            # Enable subset of log levels
            options.set_log_file(log_file)
            options.set_log_info(True)
            options.set_log_warn(False)
            options.set_log_error(True)
            options.set_log_verbose(0)
            options.set_log_format(triton_bindings.TRITONSERVER_LogFormat.DEFAULT)
            for ll, enabled in [
                (triton_bindings.TRITONSERVER_LogLevel.INFO, True),
                (triton_bindings.TRITONSERVER_LogLevel.WARN, False),
                (triton_bindings.TRITONSERVER_LogLevel.ERROR, True),
                (triton_bindings.TRITONSERVER_LogLevel.VERBOSE, False),
            ]:
                self.assertEqual(triton_bindings.TRITONSERVER_LogIsEnabled(ll), enabled)
            # Write message to each of the log level
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.INFO,
                "filename",
                123,
                "info_message",
            )
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.WARN,
                "filename",
                456,
                "warn_message",
            )
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.ERROR,
                "filename",
                789,
                "error_message",
            )
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.VERBOSE,
                "filename",
                147,
                "verbose_message",
            )
            with open(log_file, "r") as f:
                log = f.read()
                # Check level
                self.assertRegex(log, r"filename:123.*info_message")
                self.assertNotRegex(log, r"filename:456.*warn_message")
                self.assertRegex(log, r"filename:789.*error_message")
                self.assertNotRegex(log, r"filename:147.*verbose_message")
                # Check format "MMDD hh:mm:ss.ssssss".
                self.assertRegex(log, default_format_regex)
                # sanity check that there is no log with other format "YYYY-MM-DDThh:mm:ssZ L"
                self.assertNotRegex(log, iso8601_format_regex)
            # Test different format
            options.set_log_format(triton_bindings.TRITONSERVER_LogFormat.ISO8601)
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.INFO, "fn", 258, "info_message"
            )
            with open(log_file, "r") as f:
                log = f.read()
                self.assertRegex(log, r"fn:258.*info_message")
                self.assertRegex(log, iso8601_format_regex)
        finally:
            # Must make sure the log settings are reset as the logger is unique
            # within the process
            options.set_log_file("")
            options.set_log_info(False)
            options.set_log_warn(False)
            options.set_log_error(False)
            options.set_log_verbose(0)
            options.set_log_format(triton_bindings.TRITONSERVER_LogFormat.DEFAULT)
            os.remove(log_file)

    def test_buffer_attributes(self):
        expected_memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU_PINNED
        expected_memory_type_id = 4
        expected_byte_size = 1024
        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type_id = expected_memory_type_id
        self.assertEqual(buffer_attributes.memory_type_id, expected_memory_type_id)
        buffer_attributes.memory_type = expected_memory_type
        self.assertEqual(buffer_attributes.memory_type, expected_memory_type)
        buffer_attributes.byte_size = expected_byte_size
        self.assertEqual(buffer_attributes.byte_size, expected_byte_size)
        # cuda_ipc_handle is supposed to be cudaIpcMemHandle_t, must initialize buffer
        # of that size to avoid segfault. The handle getter/setter is different from other
        # attributes that different pointers may be returned from the getter, but the byte
        # content pointed by the pointer should be the same
        import ctypes
        from array import array

        handle_byte_size = 64
        mock_handle = array("b", [i for i in range(handle_byte_size)])
        buffer_attributes.cuda_ipc_handle = mock_handle.buffer_info()[0]
        res_arr = (ctypes.c_char * handle_byte_size).from_address(
            buffer_attributes.cuda_ipc_handle
        )
        for i in range(handle_byte_size):
            self.assertEqual(int.from_bytes(res_arr[i], "big"), mock_handle[i])

    def test_allocator(self):
        def alloc_fn(
            allocator, tensor_name, byte_size, memory_type, memory_type_id, user_object
        ):
            return (123, None, triton_bindings.TRITONSERVER_MemoryType.GPU, 1)

        def release_fn(
            allocator,
            buffer,
            buffer_user_object,
            byte_size,
            memory_type,
            memory_type_id,
        ):
            pass

        def start_fn(allocator, user_object):
            pass

        def query_fn(
            allocator, user_object, tensor_name, byte_size, memory_type, memory_type_id
        ):
            return (triton_bindings.TRITONSERVER_MemoryType.GPU, 1)

        def buffer_fn(
            allocator, tensor_name, buffer_attribute, user_object, buffer_user_object
        ):
            return buffer_attribute

        # allocator without start_fn
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(alloc_fn, release_fn)
        del allocator
        gc.collect()

        # allocator with start_fn
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            alloc_fn, release_fn, start_fn
        )
        allocator.set_buffer_attributes_function(buffer_fn)
        allocator.set_query_function(query_fn)

    def test_message(self):
        expected_dict = {"key_0": [1, 2, "3"], "key_1": {"nested_key": "nested_value"}}
        message = triton_bindings.TRITONSERVER_Message(json.dumps(expected_dict))
        self.assertEqual(expected_dict, json.loads(message.serialize_to_json()))

    def test_metrics(self):
        # This test depends on 'TRITONSERVER_Server' operates properly
        # to access metrics.

        # Create server in EXPLICIT mode so we don't need to ensure
        # a model repository is proper repository
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path(self._test_model_repo)
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT
        )
        server = triton_bindings.TRITONSERVER_Server(options)
        metrics = server.metrics()
        # Check one of the metrics is reported
        self.assertTrue(
            "nv_cpu_memory_used_bytes"
            in metrics.formatted(triton_bindings.TRITONSERVER_MetricFormat.PROMETHEUS)
        )

    def test_trace_enum(self):
        t_list = [
            (triton_bindings.TRITONSERVER_InferenceTraceLevel.DISABLED, "DISABLED"),
            (triton_bindings.TRITONSERVER_InferenceTraceLevel.MIN, "MIN"),
            (triton_bindings.TRITONSERVER_InferenceTraceLevel.MAX, "MAX"),
            (triton_bindings.TRITONSERVER_InferenceTraceLevel.TIMESTAMPS, "TIMESTAMPS"),
            (triton_bindings.TRITONSERVER_InferenceTraceLevel.TENSORS, "TENSORS"),
        ]
        for t, t_str in t_list:
            self.assertEqual(
                triton_bindings.TRITONSERVER_InferenceTraceLevelString(t), t_str
            )
        # bit-wise operation
        level = int(triton_bindings.TRITONSERVER_InferenceTraceLevel.TIMESTAMPS) | int(
            triton_bindings.TRITONSERVER_InferenceTraceLevel.TENSORS
        )
        self.assertNotEqual(
            level & int(triton_bindings.TRITONSERVER_InferenceTraceLevel.TIMESTAMPS), 0
        )
        self.assertNotEqual(
            level & int(triton_bindings.TRITONSERVER_InferenceTraceLevel.TENSORS), 0
        )

        t_list = [
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.REQUEST_START,
                "REQUEST_START",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.QUEUE_START,
                "QUEUE_START",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_START,
                "COMPUTE_START",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_INPUT_END,
                "COMPUTE_INPUT_END",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_OUTPUT_START,
                "COMPUTE_OUTPUT_START",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_END,
                "COMPUTE_END",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.REQUEST_END,
                "REQUEST_END",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_QUEUE_INPUT,
                "TENSOR_QUEUE_INPUT",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_BACKEND_INPUT,
                "TENSOR_BACKEND_INPUT",
            ),
            (
                triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_BACKEND_OUTPUT,
                "TENSOR_BACKEND_OUTPUT",
            ),
        ]
        for t, t_str in t_list:
            self.assertEqual(
                triton_bindings.TRITONSERVER_InferenceTraceActivityString(t), t_str
            )

    def test_trace(self):
        # This test depends on 'test_infer_async' test to capture
        # the trace
        level = int(triton_bindings.TRITONSERVER_InferenceTraceLevel.TIMESTAMPS) | int(
            triton_bindings.TRITONSERVER_InferenceTraceLevel.TENSORS
        )
        trace_dict = {"signal_queue": queue.Queue()}
        trace = triton_bindings.TRITONSERVER_InferenceTrace(
            level, 123, g_timestamp_fn, g_tensor_fn, g_trace_release_fn, trace_dict
        )
        # [FIXME] get a copy of trace id due to potential issue of 'trace'
        # lifecycle
        trace_id = trace.id

        # Send and wait for inference, not care about result.
        server = self._start_polling_server()
        (
            request,
            allocator,
            response_queue,
            request_counter,
        ) = self._prepare_inference_request(server)
        server.infer_async(request, trace)

        # [FIXME] WAR due to trace lifecycle is tied to response in Triton core,
        # trace reference should drop on response send..
        res = response_queue.get(block=True, timeout=10)
        del res
        gc.collect()

        _ = trace_dict["signal_queue"].get(block=True, timeout=10)

        # check 'trace_dict'
        self.assertTrue(trace_id in trace_dict)

        # check activity are logged correctly,
        # value of 0 indicate it is timestamp trace,
        # non-zero is tensor trace and the value is how many times this
        # particular activity should be logged
        expected_activities = {
            # timestamp
            triton_bindings.TRITONSERVER_InferenceTraceActivity.REQUEST_START: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.QUEUE_START: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_START: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_INPUT_END: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_OUTPUT_START: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.COMPUTE_END: 0,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.REQUEST_END: 0,
            # not timestamp
            triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_QUEUE_INPUT: 2,
            # TENSOR_BACKEND_INPUT never get called with in Triton core
            # triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_BACKEND_INPUT : 2,
            triton_bindings.TRITONSERVER_InferenceTraceActivity.TENSOR_BACKEND_OUTPUT: 2,
        }
        for tl in trace_dict[trace_id]:
            # basic check
            self.assertEqual(tl["id"], trace_id)
            self.assertEqual(tl["parent_id"], 123)
            self.assertEqual(tl["model_name"], self._model_name)
            self.assertEqual(tl["model_version"], 1)
            self.assertEqual(tl["request_id"], "req_0")
            self.assertTrue(tl["activity"] in expected_activities)
            if expected_activities[tl["activity"]] == 0:
                self.assertTrue("timestamp" in tl)
            else:
                self.assertTrue("tensor" in tl)
                expected_activities[tl["activity"]] -= 1
            if expected_activities[tl["activity"]] == 0:
                del expected_activities[tl["activity"]]
        # check if dict is empty to ensure the activity are logged in correct
        # amount.
        self.assertFalse(bool(expected_activities))
        request_counter.get()

    def test_options(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()

        # Generic
        options.set_server_id("server_id")
        options.set_min_supported_compute_capability(7.0)
        options.set_exit_on_error(False)
        options.set_strict_readiness(False)
        options.set_exit_timeout(30)

        # Models
        options.set_model_repository_path("model_repo_0")
        options.set_model_repository_path("model_repo_1")
        for m in [
            triton_bindings.TRITONSERVER_ModelControlMode.NONE,
            triton_bindings.TRITONSERVER_ModelControlMode.POLL,
            triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT,
        ]:
            options.set_model_control_mode(m)
        options.set_startup_model("*")
        options.set_strict_model_config(True)
        options.set_model_load_thread_count(2)
        options.set_model_namespacing(True)
        # Only support Kind GPU for now
        options.set_model_load_device_limit(
            triton_bindings.TRITONSERVER_InstanceGroupKind.GPU, 0, 0.5
        )
        for k in [
            triton_bindings.TRITONSERVER_InstanceGroupKind.AUTO,
            triton_bindings.TRITONSERVER_InstanceGroupKind.CPU,
            triton_bindings.TRITONSERVER_InstanceGroupKind.MODEL,
        ]:
            with self.assertRaises(triton_bindings.TritonError) as context:
                options.set_model_load_device_limit(k, 0, 0)
            self.assertTrue("not supported" in str(context.exception))

        # Backend
        options.set_backend_directory("backend_dir_0")
        options.set_backend_directory("backend_dir_1")
        options.set_backend_config("backend_name", "setting", "value")

        # Rate limiter
        for r in [
            triton_bindings.TRITONSERVER_RateLimitMode.OFF,
            triton_bindings.TRITONSERVER_RateLimitMode.EXEC_COUNT,
        ]:
            options.set_rate_limiter_mode(r)
        options.add_rate_limiter_resource("shared_resource", 4, -1)
        options.add_rate_limiter_resource("device_resource", 1, 0)
        # memory pools
        options.set_pinned_memory_pool_byte_size(1024)
        options.set_cuda_memory_pool_byte_size(0, 2048)
        # cache
        options.set_response_cache_byte_size(4096)
        options.set_cache_config(
            "cache_name", json.dumps({"config_0": "value_0", "config_1": "value_1"})
        )
        options.set_cache_directory("cache_dir_0")
        options.set_cache_directory("cache_dir_1")
        # Log
        try:
            options.set_log_file("some_file")
            options.set_log_info(True)
            options.set_log_warn(True)
            options.set_log_error(True)
            options.set_log_verbose(2)
            for f in [
                triton_bindings.TRITONSERVER_LogFormat.DEFAULT,
                triton_bindings.TRITONSERVER_LogFormat.ISO8601,
            ]:
                options.set_log_format(f)
        finally:
            # Must make sure the log settings are reset as the logger is unique
            # within the process
            options.set_log_file("")
            options.set_log_info(False)
            options.set_log_warn(False)
            options.set_log_error(False)
            options.set_log_verbose(0)
            options.set_log_format(triton_bindings.TRITONSERVER_LogFormat.DEFAULT)

        # Metrics
        options.set_gpu_metrics(True)
        options.set_cpu_metrics(True)
        options.set_metrics_interval(5)
        options.set_metrics_config("metrics_group", "setting", "value")

        # Misc..
        with self.assertRaises(triton_bindings.TritonError) as context:
            options.set_host_policy("policy_name", "setting", "value")
        self.assertTrue("Unsupported host policy setting" in str(context.exception))
        options.set_repo_agent_directory("repo_agent_dir_0")
        options.set_repo_agent_directory("repo_agent_dir_1")
        options.set_buffer_manager_thread_count(4)

    def test_server(self):
        server = self._start_polling_server()
        # is_live
        self.assertTrue(server.is_live())
        # is_ready
        self.assertTrue(server.is_ready())
        # model_is_ready
        self.assertTrue(server.model_is_ready(self._model_name, -1))
        # model_batch_properties
        expected_batch_properties = (
            int(triton_bindings.TRITONSERVER_ModelBatchFlag.UNKNOWN),
            0,
        )
        self.assertEqual(
            server.model_batch_properties(self._model_name, -1),
            expected_batch_properties,
        )
        # model_transaction_properties
        expected_transaction_policy = (
            int(triton_bindings.TRITONSERVER_ModelTxnPropertyFlag.ONE_TO_ONE),
            0,
        )
        self.assertEqual(
            server.model_transaction_properties(self._model_name, -1),
            expected_transaction_policy,
        )
        # metadata
        server_meta_data = self._to_pyobject(server.metadata())
        self.assertTrue("name" in server_meta_data)
        self.assertEqual(server_meta_data["name"], "testing_server")
        # model_metadata
        model_meta_data = self._to_pyobject(server.model_metadata(self._model_name, -1))
        self.assertTrue("name" in model_meta_data)
        self.assertEqual(model_meta_data["name"], self._model_name)
        # model_statistics
        model_statistics = self._to_pyobject(
            server.model_statistics(self._model_name, -1)
        )
        self.assertTrue("model_stats" in model_statistics)
        # model_config
        model_config = self._to_pyobject(server.model_config(self._model_name, -1, 1))
        self.assertTrue("input" in model_config)
        # model_index
        model_index = self._to_pyobject(server.model_index(0))
        self.assertEqual(model_index[0]["name"], self._model_name)
        # metrics (see test_metrics)
        # infer_async (see test_infer_async)

    def test_request(self):
        # This test depends on 'TRITONSERVER_Server' operates properly to initialize
        # the request
        server = self._start_polling_server()

        with self.assertRaises(triton_bindings.NotFoundError) as ctx:
            _ = triton_bindings.TRITONSERVER_InferenceRequest(
                server, "not_existing_model", -1
            )
        self.assertTrue("unknown model" in str(ctx.exception))

        expected_request_id = "request"
        expected_flags = int(
            triton_bindings.TRITONSERVER_RequestFlag.SEQUENCE_START
        ) | int(triton_bindings.TRITONSERVER_RequestFlag.SEQUENCE_END)
        expected_correlation_id = 2
        expected_correlation_id_string = "123"
        expected_priority = 19
        # larger value than model max priority level,
        # will be set to default (10, see 'g_python_addsub' for config detail)
        expected_priority_uint64 = 67
        expected_timeout_microseconds = 222

        request = triton_bindings.TRITONSERVER_InferenceRequest(server, "addsub", -1)

        # request metadata
        request.id = expected_request_id
        self.assertEqual(request.id, expected_request_id)
        request.flags = expected_flags
        self.assertEqual(request.flags, expected_flags)
        request.correlation_id = expected_correlation_id
        self.assertEqual(request.correlation_id, expected_correlation_id)
        request.correlation_id_string = expected_correlation_id_string
        self.assertEqual(request.correlation_id_string, expected_correlation_id_string)
        # Expect error from retrieving correlation id in a wrong type,
        # wrap in lambda function to avoid early evaluation that raises
        # exception before assert
        self.assertRaises(triton_bindings.TritonError, lambda: request.correlation_id)
        request.priority = expected_priority
        self.assertEqual(request.priority, expected_priority)
        request.priority_uint64 = expected_priority_uint64
        self.assertEqual(request.priority_uint64, 10)
        request.timeout_microseconds = expected_timeout_microseconds
        self.assertEqual(request.timeout_microseconds, expected_timeout_microseconds)

        request.set_string_parameter("str_key", "str_val")
        request.set_int_parameter("int_key", 567)
        request.set_bool_parameter("bool_key", False)

        # I/O
        input = numpy.ones([2, 3], dtype=numpy.float32)
        buffer = input.ctypes.data
        ba = triton_bindings.TRITONSERVER_BufferAttributes()
        ba.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        ba.memory_type_id = 0
        ba.byte_size = input.itemsize * input.size

        request.add_input(
            "INPUT0", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        self.assertRaises(triton_bindings.TritonError, request.remove_input, "INPUT2")
        # raw input assumes single input
        self.assertRaises(triton_bindings.TritonError, request.add_raw_input, "INPUT1")
        request.remove_input("INPUT0")
        request.add_raw_input("INPUT1")
        request.remove_all_inputs()
        # all inputs are removed, all 'append' functions should raise exceptions
        aid_args = ["INPUT0", buffer, ba.byte_size, ba.memory_type, ba.memory_type_id]
        self.assertRaises(
            triton_bindings.TritonError, request.append_input_data, *aid_args
        )
        self.assertRaises(
            triton_bindings.TritonError,
            request.append_input_data_with_host_policy,
            *aid_args,
            "host_policy_name"
        )
        self.assertRaises(
            triton_bindings.TritonError,
            request.append_input_data_with_buffer_attributes,
            "INPUT0",
            buffer,
            ba,
        )
        self.assertRaises(
            triton_bindings.TritonError, request.remove_all_input_data, "INPUT0"
        )
        # Add back input
        request.add_input(
            "INPUT0", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        request.append_input_data(*aid_args)
        request.remove_all_input_data("INPUT0")

        request.add_requested_output("OUTPUT0")
        request.remove_requested_output("OUTPUT1")
        request.remove_all_requested_outputs()

    def test_infer_async(self):
        # start server
        server = self._start_polling_server()

        # prepare for infer
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            g_alloc_fn, g_release_fn, g_start_fn
        )
        allocator.set_buffer_attributes_function(g_buffer_fn)
        allocator.set_query_function(g_query_fn)

        request_counter = queue.Queue()
        response_queue = queue.Queue()
        allocator_counter = {}
        request = triton_bindings.TRITONSERVER_InferenceRequest(
            server, self._model_name, -1
        )
        request.id = "req_0"
        request.set_release_callback(g_request_fn, request_counter)
        request.set_response_callback(
            allocator, allocator_counter, g_response_fn, response_queue
        )

        input = numpy.ones([4], dtype=numpy.float32)
        input_buffer = input.ctypes.data
        ba = triton_bindings.TRITONSERVER_BufferAttributes()
        ba.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        ba.memory_type_id = 0
        ba.byte_size = input.itemsize * input.size

        request.add_input(
            "INPUT0", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        request.add_input(
            "INPUT1", triton_bindings.TRITONSERVER_DataType.FP32, input.shape
        )
        request.append_input_data_with_buffer_attributes("INPUT0", input_buffer, ba)
        request.append_input_data_with_buffer_attributes("INPUT1", input_buffer, ba)

        # non-blocking, wait on response complete
        server.infer_async(request)

        # Expect every response to be returned in 10 seconds
        flags, res = response_queue.get(block=True, timeout=10)
        self.assertEqual(
            flags, int(triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL)
        )
        # expect no error
        res.throw_if_response_error()
        # version will be actual model version
        self.assertEqual(res.model, (self._model_name, 1))
        self.assertEqual(res.id, request.id)
        self.assertEqual(res.parameter_count, 0)
        # out of range access
        self.assertRaises(triton_bindings.TritonError, res.parameter, 0)

        # read output tensor
        self.assertEqual(res.output_count, 2)
        for out, expected_name, expected_data in [
            (res.output(0), "OUTPUT0", input + input),
            (res.output(1), "OUTPUT1", input - input),
        ]:
            (
                name,
                data_type,
                shape,
                out_buffer,
                byte_size,
                memory_type,
                memory_type_id,
                numpy_buffer,
            ) = out
            self.assertEqual(name, expected_name)
            self.assertEqual(data_type, triton_bindings.TRITONSERVER_DataType.FP32)
            self.assertEqual(shape, expected_data.shape)
            self.assertEqual(out_buffer, numpy_buffer.ctypes.data)
            # buffer attribute used for input doesn't necessarily to
            # match output buffer attributes, this is just knowing the detail.
            self.assertEqual(byte_size, ba.byte_size)
            self.assertEqual(memory_type, ba.memory_type)
            self.assertEqual(memory_type_id, ba.memory_type_id)
            self.assertTrue(
                numpy.allclose(
                    numpy_buffer.view(dtype=expected_data.dtype).reshape(shape),
                    expected_data,
                )
            )

        # label (no label so empty)
        self.assertEqual(len(res.output_classification_label(0, 1)), 0)
        # [FIXME] keep alive behavior is not established between response
        # and server, so must explicitly handle the destruction order for now.
        del res

        # sanity check on user objects
        self.assertEqual(allocator_counter["start"], 1)
        self.assertEqual(allocator_counter["alloc"], 2)
        # Knowing implementation detail that the backend doesn't use query API
        self.assertTrue("query" not in allocator_counter)
        self.assertEqual(allocator_counter["buffer"], 2)
        # Expect request to be released in 10 seconds
        request = request_counter.get(block=True, timeout=10)

    def test_server_explicit(self):
        self._create_model_repository()
        # explicit : load with params
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path(self._test_model_repo)
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT
        )
        options.set_strict_model_config(False)
        server = triton_bindings.TRITONSERVER_Server(options)
        load_file_params = [
            triton_bindings.TRITONSERVER_Parameter("config", r"{}"),
            triton_bindings.TRITONSERVER_Parameter(
                "file:" + os.path.join(self._version, self._file_name), g_python_addsub
            ),
        ]
        server.load_model_with_parameters("wired_addsub", load_file_params)
        self.assertTrue(server.model_is_ready("wired_addsub", -1))

        # Model Repository
        self.assertFalse(server.model_is_ready(self._model_name, -1))
        # unregister
        server.unregister_model_repository(self._test_model_repo)
        self.assertRaises(
            triton_bindings.TritonError, server.load_model, self._model_name
        )
        # register
        server.register_model_repository(self._test_model_repo, [])
        server.load_model(self._model_name)
        self.assertTrue(server.model_is_ready(self._model_name, -1))

        # unload
        server.unload_model("wired_addsub")
        self.assertFalse(server.model_is_ready("wired_addsub", -1))
        server.unload_model_and_dependents(self._model_name)
        self.assertFalse(server.model_is_ready(self._model_name, -1))

    def test_custom_metric(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path(self._test_model_repo)
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT
        )
        server = triton_bindings.TRITONSERVER_Server(options)

        # create custom metric
        mf = triton_bindings.TRITONSERVER_MetricFamily(
            triton_bindings.TRITONSERVER_MetricKind.COUNTER,
            "custom_metric_familiy",
            "custom metric example",
        )
        m = triton_bindings.TRITONSERVER_Metric(mf, [])
        m.increment(2)
        self.assertEqual(m.kind, triton_bindings.TRITONSERVER_MetricKind.COUNTER)
        self.assertEqual(m.value, 2)
        # can't use 'set_value' due to wrong kind
        self.assertRaises(triton_bindings.TritonError, m.set_value, 5)

        # Check custom metric is reported
        metrics = server.metrics()
        self.assertTrue(
            "custom_metric_familiy"
            in metrics.formatted(triton_bindings.TRITONSERVER_MetricFormat.PROMETHEUS)
        )


if __name__ == "__main__":
    unittest.main()
