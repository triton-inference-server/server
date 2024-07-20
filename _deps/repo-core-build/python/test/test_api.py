# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import copy
import json
import os
import queue
import shutil
import time
import unittest

import numpy
import pytest
import tritonserver

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch

    if not torch.cuda.is_available():
        torch = None
except ImportError:
    torch = None

module_directory = os.path.split(os.path.abspath(__file__))[0]
test_model_directory = os.path.abspath(
    os.path.join(module_directory, "test_api_models")
)
test_logs_directory = os.path.abspath(os.path.join(module_directory, "test_api_logs"))

shutil.rmtree(test_logs_directory, ignore_errors=True)

os.makedirs(test_logs_directory)

server_options = tritonserver.Options(
    server_id="TestServer",
    model_repository=test_model_directory,
    log_verbose=6,
    log_error=True,
    log_warn=True,
    log_info=True,
    exit_on_error=True,
    strict_model_config=False,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    exit_timeout=10,
)


class ModelTests(unittest.TestCase):
    def setup_method(self, method):
        self._server_options = copy.copy(server_options)
        self._server_options.log_file = os.path.join(
            test_logs_directory, method.__name__ + ".server.log"
        )

    def test_create_request(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        request = server.models()["test"].create_request()

        request = tritonserver.InferenceRequest(server.model("test"))


class AllocatorTests(unittest.TestCase):
    class MockMemoryAllocator(tritonserver.MemoryAllocator):
        def __init__(self):
            pass

        def allocate(self, *args, **kwargs):
            raise Exception("foo")

    def setup_method(self, method):
        self._server_options = copy.copy(server_options)
        self._server_options.log_file = os.path.join(
            test_logs_directory, method.__name__ + ".server.log"
        )

    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_memory_fallback_to_cpu(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        allocator = tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU]

        del tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU]

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {
                            "decoupled": {"string_value": "False"},
                            "request_gpu_memory": {"string_value": "True"},
                        },
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
        ):
            self.assertEqual(
                response.outputs["fp16_output"].memory_type, tritonserver.MemoryType.CPU
            )
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            self.assertEqual(fp16_input[0][0], fp16_output[0][0])

        tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU] = allocator

    def test_memory_allocator_exception(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        with self.assertRaises(tritonserver.InternalError):
            for response in server.model("test").infer(
                inputs={
                    "string_input": tritonserver.Tensor.from_string_array([["hello"]])
                },
                output_memory_type="gpu",
                output_memory_allocator=AllocatorTests.MockMemoryAllocator(),
            ):
                pass

    def test_unsupported_memory_type(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        if tritonserver.MemoryType.GPU in tritonserver.default_memory_allocators:
            allocator = tritonserver.default_memory_allocators[
                tritonserver.MemoryType.GPU
            ]

            del tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU]
        else:
            allocator = None

        with self.assertRaises(tritonserver.InvalidArgumentError):
            for response in server.model("test").infer(
                inputs={
                    "string_input": tritonserver.Tensor.from_string_array([["hello"]])
                },
                output_memory_type="gpu",
            ):
                pass

        if allocator is not None:
            tritonserver.default_memory_allocators[
                tritonserver.MemoryType.GPU
            ] = allocator

    @pytest.mark.skipif(torch is None, reason="Skipping test, torch not installed")
    def test_allocate_on_cpu_and_reshape(self):
        allocator = tritonserver.default_memory_allocators[tritonserver.MemoryType.CPU]

        memory_buffer = allocator.allocate(
            memory_type=tritonserver.MemoryType.CPU, memory_type_id=0, size=200
        )

        cpu_array = memory_buffer.owner

        self.assertEqual(memory_buffer.size, 200)

        fp32_size = int(memory_buffer.size / 4)

        tensor = tritonserver.Tensor(
            tritonserver.DataType.FP32, shape=[fp32_size], memory_buffer=memory_buffer
        )

        cpu_fp32_array = numpy.from_dlpack(tensor)
        self.assertEqual(cpu_array.ctypes.data, cpu_fp32_array.ctypes.data)
        self.assertEqual(cpu_fp32_array.dtype, numpy.float32)
        self.assertEqual(cpu_fp32_array.nbytes, 200)

    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    @pytest.mark.skipif(torch is None, reason="Skipping test, torch not installed")
    def test_allocate_on_gpu_and_reshape(self):
        if cupy is None:
            return

        allocator = tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU]

        memory_buffer = allocator.allocate(
            memory_type=tritonserver.MemoryType.GPU, memory_type_id=0, size=200
        )

        gpu_array = memory_buffer.owner

        gpu_array = cupy.empty([10, 20], dtype=cupy.uint8)
        memory_buffer = tritonserver.MemoryBuffer.from_dlpack(gpu_array)

        self.assertEqual(memory_buffer.size, 200)

        fp32_size = int(memory_buffer.size / 4)

        tensor = tritonserver.Tensor(
            tritonserver.DataType.FP32, shape=[fp32_size], memory_buffer=memory_buffer
        )

        gpu_fp32_array = cupy.from_dlpack(tensor)
        self.assertEqual(
            gpu_array.__cuda_array_interface__["data"][0],
            gpu_fp32_array.__cuda_array_interface__["data"][0],
        )
        self.assertEqual(gpu_fp32_array.dtype, cupy.float32)
        self.assertEqual(gpu_fp32_array.nbytes, 200)

        torch_fp32_tensor = torch.from_dlpack(tensor)
        self.assertEqual(torch_fp32_tensor.dtype, torch.float32)
        self.assertEqual(
            torch_fp32_tensor.data_ptr(), gpu_array.__cuda_array_interface__["data"][0]
        )
        self.assertEqual(torch_fp32_tensor.nbytes, 200)


class TensorTests(unittest.TestCase):
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_cpu_to_gpu(self):
        if cupy is None:
            return
        cpu_array = numpy.random.rand(1, 3, 100, 100).astype(numpy.float32)
        cpu_tensor = tritonserver.Tensor.from_dlpack(cpu_array)
        gpu_tensor = cpu_tensor.to_device("gpu:0")
        gpu_array = cupy.from_dlpack(gpu_tensor)

        self.assertEqual(gpu_array.device, cupy.cuda.Device(0))

        numpy.testing.assert_array_equal(cpu_array, gpu_array.get())

        memory_buffer = tritonserver.MemoryBuffer.from_dlpack(gpu_array)

        self.assertEqual(
            gpu_array.__cuda_array_interface__["data"][0], memory_buffer.data_ptr
        )

    @pytest.mark.skipif(
        torch is None, reason="Skipping gpu memory, torch not installed"
    )
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_gpu_tensor_from_dl_pack(self):
        if cupy is None or torch is None:
            return
        cupy_array = cupy.ones([100]).astype(cupy.float64)
        tensor = tritonserver.Tensor.from_dlpack(cupy_array)
        torch_tensor = torch.from_dlpack(cupy_array)

        self.assertEqual(torch_tensor.data_ptr(), tensor.data_ptr)
        self.assertEqual(torch_tensor.nbytes, tensor.size)
        self.assertEqual(torch_tensor.__dlpack_device__(), tensor.__dlpack_device__())

    @pytest.mark.skipif(torch is None, reason="Skipping test, torch not installed")
    def test_tensor_from_numpy(self):
        cpu_array = numpy.random.rand(1, 3, 100, 100).astype(numpy.float32)
        tensor = tritonserver.Tensor.from_dlpack(cpu_array)
        torch_tensor = torch.from_dlpack(tensor)
        numpy.testing.assert_array_equal(torch_tensor.numpy(), cpu_array)
        self.assertEqual(torch_tensor.data_ptr(), cpu_array.ctypes.data)


class ServerTests(unittest.TestCase):
    def setup_method(self, method):
        self._server_options = copy.copy(server_options)
        self._server_options.log_file = os.path.join(
            test_logs_directory, method.__name__ + ".server.log"
        )

    def test_not_started(self):
        server = tritonserver.Server()
        with self.assertRaises(tritonserver.InvalidArgumentError):
            server.ready()

    def test_invalid_option_type(self):
        server = tritonserver.Server(server_id=1)
        with self.assertRaises(TypeError):
            server.start()

        server = tritonserver.Server(model_repository=1)
        with self.assertRaises(TypeError):
            server.start()

    def test_invalid_repo(self):
        with self.assertRaises(tritonserver.InternalError):
            tritonserver.Server(model_repository="foo").start()

    def test_ready(self):
        server = tritonserver.Server(self._server_options).start()
        self.assertTrue(server.ready())

    @pytest.mark.xfail(
        tritonserver.__version__ <= "2.48.0",
        reason="Known issue on stop: Exit timeout expired. Exiting immediately",
        raises=tritonserver.InternalError,
    )
    def test_stop(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            numpy.testing.assert_array_equal(fp16_input, fp16_output)

        server.stop()

    def test_model_repository_not_specified(self):
        with self.assertRaises(tritonserver.InvalidArgumentError):
            tritonserver.Server(model_repository=None).start()


class InferenceTests(unittest.TestCase):
    def setup_method(self, method):
        self._server_options = copy.copy(server_options)
        self._server_options.log_file = os.path.join(
            test_logs_directory, method.__name__ + ".server.log"
        )

    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_gpu_output(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="gpu",
        ):
            fp16_output = cupy.from_dlpack(response.outputs["fp16_output"])
            self.assertEqual(fp16_input[0][0], fp16_output[0][0])

        for response in server.model("test").infer(
            inputs={"string_input": [["hello"]]},
            output_memory_type="gpu",
        ):
            text_output = response.outputs["string_output"].to_string_array()
            self.assertEqual(text_output[0][0], "hello")

        for response in server.model("test").infer(
            inputs={"string_input": tritonserver.Tensor.from_string_array([["hello"]])},
            output_memory_type="gpu",
        ):
            text_output = response.outputs["string_output"].to_string_array()
            text_output = response.outputs["string_output"].to_string_array()
            self.assertEqual(text_output[0][0], "hello")

    def test_basic_inference(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        inputs = {
            "fp16_input": numpy.random.rand(1, 100).astype(dtype=numpy.float16),
            "bool_input": numpy.random.rand(1, 100).astype(dtype=numpy.bool_),
        }

        for response in server.model("test").infer(
            inputs=inputs,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            for input_name, input_value in inputs.items():
                output_value = response.outputs[input_name.replace("input", "output")]
                output_value = numpy.from_dlpack(output_value)
                numpy.testing.assert_array_equal(input_value, output_value)

        # test normal bool
        inputs = {"bool_input": [[True, False, False, True]]}

        for response in server.model("test").infer(
            inputs=inputs,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            for input_name, input_value in inputs.items():
                output_value = numpy.from_dlpack(
                    response.outputs[input_name.replace("input", "output")]
                )
                numpy.testing.assert_array_equal(input_value, output_value)

    def test_parameters(self):
        server = tritonserver.Server(self._server_options).start(wait_until_ready=True)

        self.assertTrue(server.ready())

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        input_parameters = {
            "int_parameter": 0,
            "float_parameter": 0.5,
            "bool_parameter": False,
            "string_parameter": "test",
        }
        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            parameters=input_parameters,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            numpy.testing.assert_array_equal(fp16_input, fp16_output)
            output_parameters = json.loads(
                response.outputs["output_parameters"].to_string_array()[0]
            )
            assert input_parameters == output_parameters

        with self.assertRaises(tritonserver.InvalidArgumentError):
            input_parameters = {
                "invalid": {"test": 1},
            }

            server.model("test").infer(
                inputs={"fp16_input": fp16_input},
                parameters=input_parameters,
                output_memory_type="cpu",
                raise_on_error=True,
            )

        with self.assertRaises(tritonserver.InvalidArgumentError):
            input_parameters = {
                "invalid": None,
            }

            server.model("test").infer(
                inputs={"fp16_input": fp16_input},
                parameters=input_parameters,
                output_memory_type="cpu",
                raise_on_error=True,
            )
