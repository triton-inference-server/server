#!/usr/bin/env python3

# Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import unittest
from pathlib import Path

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
import util
from parameterized import parameterized
from tritonclient import utils


class SharedMemoryTest(tu.TestResultCollector):
    # Constant members
    shared_memory_test_client_log = Path(os.getcwd()) / "client.log"
    model_dir_path = Path(os.getcwd()) / "models"
    model_source_path = Path(os.getcwd()).parents[0] / "python_models/add_sub/model.py"
    model_config_source_path = (
        Path(os.getcwd()).parents[0] / "python_models/add_sub/config.pbtxt"
    )

    # Custom setup method to allow passing of parameters
    def _setUp(self, protocol, log_file_path):
        self._tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
        self._test_windows = bool(int(os.environ.get("TEST_WINDOWS", 0)))
        self._shm_key_prefix = "/" if not self._test_windows else "Global\\"
        self._timeout = os.environ.get("SERVER_TIMEOUT", 120)
        self._protocol = protocol
        self._test_passed = False
        self._original_stdout, self._original_stderr = None, None
        self._log_file_path = log_file_path

        if self._protocol == "http":
            self._url = self._tritonserver_ipaddr + ":8000"
            self._triton_client = httpclient.InferenceServerClient(
                self._url, verbose=True
            )
        else:
            self._url = self._tritonserver_ipaddr + ":8001"
            self._triton_client = grpcclient.InferenceServerClient(
                self._url, verbose=True
            )
        self._build_model_repo()
        self._build_server_args()
        self._shared_memory_test_server_log = open(log_file_path, "w")
        self._server_process = util.run_server(
            server_executable=self._server_executable,
            launch_command=self._launch_command,
            log_file=self._shared_memory_test_server_log,
        )
        util.wait_for_server_ready(
            self._server_process, self._triton_client, self._timeout
        )
        # Redirect console output to client log and begin test
        self._client_log = open(SharedMemoryTest.shared_memory_test_client_log, "a")
        self._original_stdout, self._original_stderr = util.stream_to_log(
            self._client_log
        )

    def _build_server_args(self):
        if self._test_windows:
            backend_dir = "C:\\opt\\tritonserver\\backends"
            model_dir = "C:\\opt\\tritonserver\\qa\\L0_shared_memory\\models"
            self._server_executable = "C:\\opt\\tritonserver\\bin\\tritonserver.exe"
        else:
            triton_dir = os.environ.get("TRITON_DIR", "/opt/tritonserver")
            backend_dir = os.environ.get("BACKEND_DIR", f"{triton_dir}/backends")
            model_dir = os.environ.get("MODELDIR", (os.getcwd() + "/models"))
            self._server_executable = os.environ.get(
                "SERVER", f"{triton_dir}/bin/tritonserver"
            )

        extra_args = f"--backend-directory={backend_dir}"
        self._launch_command = f"{self._server_executable} --model-repository={model_dir} --log-verbose 1 {extra_args}"

    def _build_model_repo(self, model_name="simple", model_version=1):
        util.create_model_dir(
            SharedMemoryTest.model_dir_path,
            model_name,
            model_version,
            SharedMemoryTest.model_source_path,
            SharedMemoryTest.model_config_source_path,
        )
        test_model_config = (
            SharedMemoryTest.model_dir_path / model_name / "config.pbtxt"
        )
        util.replace_config_attribute(test_model_config, "TYPE_FP32", "TYPE_INT32")
        util.add_config_attribute(test_model_config, "max_batch_size: 8")

    def tearDown(self):
        util.kill_server(self._server_process)
        # Restore stdout / stderr so we can print to console and see server
        # output in CI even after logs expire. Print test result to client
        # before doing so for legibility.
        if not self._test_passed:
            print("*\n*\n*\nTest Failed\n*\n*\n*\n")
        util.stream_to_console(self._original_stdout, self._original_stderr)
        self._shared_memory_test_server_log.close()
        self._client_log.close()
        # Print server log to console on failure
        if not self._test_passed:
            with open(self._log_file_path, "r") as f:
                print(f.read())
            print("*\n*\n*\nEnd of Server Output\n*\n*\n*\n")

    def _configure_sever(self):
        shm_ip0_handle = shm.create_shared_memory_region(
            "input0_data", (self._shm_key_prefix + "input0_data"), 64
        )
        shm_ip1_handle = shm.create_shared_memory_region(
            "input1_data", (self._shm_key_prefix + "input1_data"), 64
        )
        shm_op0_handle = shm.create_shared_memory_region(
            "output0_data", (self._shm_key_prefix + "output0_data"), 64
        )
        shm_op1_handle = shm.create_shared_memory_region(
            "output1_data", (self._shm_key_prefix + "output1_data"), 64
        )
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        shm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        shm.set_shared_memory_region(shm_ip1_handle, [input1_data])

        self._triton_client.register_system_shared_memory(
            "input0_data", (self._shm_key_prefix + "input0_data"), 64
        )
        self._triton_client.register_system_shared_memory(
            "input1_data", (self._shm_key_prefix + "input1_data"), 64
        )
        self._triton_client.register_system_shared_memory(
            "output0_data", (self._shm_key_prefix + "output0_data"), 64
        )
        self._triton_client.register_system_shared_memory(
            "output1_data", (self._shm_key_prefix + "output1_data"), 64
        )
        return [shm_ip0_handle, shm_ip1_handle, shm_op0_handle, shm_op1_handle]

    def _cleanup_server(self, shm_handles):
        for shm_handle in shm_handles:
            shm.destroy_shared_memory_region(shm_handle)

    def _basic_inference(
        self,
        shm_ip0_handle,
        shm_ip1_handle,
        shm_op0_handle,
        shm_op1_handle,
        error_msg,
        big_shm_name="",
        big_shm_size=64,
        shm_output_offset=0,
    ):
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        inputs = []
        outputs = []
        if self._protocol == "http":
            inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
            inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))
            outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))
            outputs.append(
                httpclient.InferRequestedOutput("OUTPUT1", binary_data=False)
            )
        else:
            inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))
            inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT1"))

        inputs[0].set_shared_memory("input0_data", 64)

        if type(shm_ip1_handle) == np.array:
            inputs[1].set_data_from_numpy(input0_data, binary_data=True)
        elif big_shm_name != "":
            inputs[1].set_shared_memory(big_shm_name, big_shm_size)
        else:
            inputs[1].set_shared_memory("input1_data", 64)

        outputs[0].set_shared_memory("output0_data", 64, offset=shm_output_offset)
        outputs[1].set_shared_memory("output1_data", 64, offset=shm_output_offset)

        try:
            results = self._triton_client.infer(
                "simple", inputs, model_version="", outputs=outputs
            )
            output = results.get_output("OUTPUT0")
            if self._protocol == "http":
                output_datatype = output["datatype"]
                output_shape = output["shape"]
            else:
                output_datatype = output.datatype
                output_shape = output.shape
            output_dtype = utils.triton_to_np_dtype(output_datatype)
            output_data = shm.get_contents_as_numpy(
                shm_op0_handle, output_dtype, output_shape
            )
            self.assertTrue(
                (output_data[0] == (input0_data + input1_data)).all(),
                "Model output does not match expected output",
            )
        except Exception as ex:
            error_msg.append(str(ex))

    @parameterized.expand([("grpc"), ("http")])
    def test_invalid_create_shm(self, protocol):
        # Raises error since tried to create invalid system shared memory region
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_invalid_create_shm.{protocol}.server.log",
        )
        print(f"*\n*\n*\nStarting Test:test_invalid_create_shm.{protocol}\n*\n*\n*\n")
        try:
            shm.create_shared_memory_region(
                "dummy_data", (self._shm_key_prefix + "dummy_data"), -1
            )
        except Exception as ex:
            if self._test_windows:
                self.assertEqual(str(ex), "unable to create file mapping")
            else:
                self.assertEqual(str(ex), "unable to initialize the size")
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_invalid_registration(self, protocol):
        # Attempt to register non-existent shared memory region
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_invalid_registration.{protocol}.server.log",
        )
        print(f"*\n*\n*\nStarting Test:test_invalid_registration.{protocol}\n*\n*\n*\n")

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        shm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        try:
            self._triton_client.register_system_shared_memory(
                "dummy_data", (self._shm_key_prefix + "wrong_key"), 8
            )
        except Exception as ex:
            self.assertIn("Unable to open shared memory region", str(ex))
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_valid_create_set_register(self, protocol):
        # Create a valid system shared memory region, fill data in it and register
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_valid_create_set_register.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_valid_create_set_register.{protocol}\n*\n*\n*\n"
        )

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        shm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        self._triton_client.register_system_shared_memory(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_different_name_same_key(self, protocol):
        # Create a valid system shared memory region, fill data in it and register
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_different_name_same_key.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_different_name_same_key.{protocol}\n*\n*\n*\n"
        )

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy", (self._shm_key_prefix + "dummy_data"), 8
        )
        shm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        self._triton_client.register_system_shared_memory(
            "dummy", (self._shm_key_prefix + "dummy_data"), 8
        )
        try:
            self._triton_client.register_system_shared_memory(
                "dummy2", (self._shm_key_prefix + "dummy_data"), 8
            )
        except Exception as ex:
            self.assertIn("registering an active shared memory key", str(ex))
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_unregister_before_register(self, protocol):
        # Create a valid system shared memory region and unregister before register
        self._setUp(
            protocol,
            Path(os.getcwd())
            / f"test_unregister_before_register.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_unregister_before_register.{protocol}\n*\n*\n*\n"
        )

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        self._triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_unregister_after_register(self, protocol):
        # Create a valid system shared memory region and unregister after register
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_unregister_after_register.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_unregister_after_register.{protocol}\n*\n*\n*\n"
        )

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        self._triton_client.register_system_shared_memory(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        self._triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_reregister_after_register(self, protocol):
        # Create a valid system shared memory region and unregister after register
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_reregister_after_register.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_reregister_after_register.{protocol}\n*\n*\n*\n"
        )

        shm_op0_handle = shm.create_shared_memory_region(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        self._triton_client.register_system_shared_memory(
            "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
        )
        try:
            self._triton_client.register_system_shared_memory(
                "dummy_data", (self._shm_key_prefix + "dummy_data"), 8
            )
        except Exception as ex:
            self.assertIn(
                "shared memory region 'dummy_data' already in manager", str(ex)
            )
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        shm.destroy_shared_memory_region(shm_op0_handle)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_unregister_after_inference(self, protocol):
        # Unregister after inference
        self._setUp(
            protocol,
            Path(os.getcwd())
            / f"test_unregister_after_inference.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_unregister_after_inference.{protocol}\n*\n*\n*\n"
        )

        error_msg = []
        shm_handles = self._configure_sever()
        self._basic_inference(
            shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))

        self._triton_client.unregister_system_shared_memory("output0_data")
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 3)
        else:
            self.assertEqual(len(shm_status.regions), 3)
        self._cleanup_server(shm_handles)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_register_after_inference(self, protocol):
        # Register after inference
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_register_after_inference.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_register_after_inference.{protocol}\n*\n*\n*\n"
        )

        error_msg = []
        shm_handles = self._configure_sever()

        self._basic_inference(
            shm_handles[0], shm_handles[1], shm_handles[2], shm_handles[3], error_msg
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = shm.create_shared_memory_region(
            "input2_data", (self._shm_key_prefix + "input2_data"), 64
        )
        self._triton_client.register_system_shared_memory(
            "input2_data", (self._shm_key_prefix + "input2_data"), 64
        )
        shm_status = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(shm_status), 5)
        else:
            self.assertEqual(len(shm_status.regions), 5)
        shm_handles.append(shm_ip2_handle)
        self._cleanup_server(shm_handles)
        self._test_passed = True

    # FIXME: Only works with graphdef models.
    # @parameterized.expand([("grpc"),("http")])
    # def test_too_big_shm(self, protocol):
    #     # Shared memory input region larger than needed - Throws error
    #     error_msg = []
    #     shm_handles = self._configure_sever()
    #     shm_ip2_handle = shm.create_shared_memory_region(
    #         "input2_data", (self._shm_key_prefix + "input2_data"), 128
    #     )

    #     self._triton_client.register_system_shared_memory("input2_data", (self._shm_key_prefix + "input2_data"), 128)
    #     shm_status = self._triton_client.get_system_shared_memory_status()
    #     self._basic_inference(
    #         shm_handles[0],
    #         shm_ip2_handle,
    #         shm_handles[2],
    #         shm_handles[3],
    #         error_msg,
    #         "input2_data",
    #         128,
    #     )
    #     if len(error_msg) > 0:
    #         self.assertTrue(
    #             "unexpected total byte size 128 for input 'INPUT1', expecting 64"
    #             in error_msg[-1]
    #         )
    #     shm_handles.append(shm_ip2_handle)
    #     self._cleanup_server(shm_handles)

    @parameterized.expand([("grpc"), ("http")])
    def test_mixed_raw_shm(self, protocol):
        # Mix of shared memory and RAW inputs
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_mixed_raw_shm.{protocol}.server.log",
        )
        print(f"*\n*\n*\nStarting Test:test_mixed_raw_shm.{protocol}\n*\n*\n*\n")

        error_msg = []
        shm_handles = self._configure_sever()
        input1_data = np.ones(shape=16, dtype=np.int32)
        self._basic_inference(
            shm_handles[0], [input1_data], shm_handles[2], shm_handles[3], error_msg
        )
        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_server(shm_handles)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_unregisterall(self, protocol):
        # Unregister all shared memory blocks
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_unregisterall.{protocol}.server.log",
        )
        print(f"*\n*\n*\nStarting Test:test_unregisterall.{protocol}\n*\n*\n*\n")

        shm_handles = self._configure_sever()

        status_before = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(status_before), 4)
        else:
            self.assertEqual(len(status_before.regions), 4)
        self._triton_client.unregister_system_shared_memory()
        status_after = self._triton_client.get_system_shared_memory_status()
        if self._protocol == "http":
            self.assertEqual(len(status_after), 0)
        else:
            self.assertEqual(len(status_after.regions), 0)
        self._cleanup_server(shm_handles)
        self._test_passed = True

    @parameterized.expand([("grpc"), ("http")])
    def test_infer_offset_out_of_bound(self, protocol):
        # Shared memory offset outside output region - Throws error
        self._setUp(
            protocol,
            Path(os.getcwd()) / f"test_infer_offset_out_of_bound.{protocol}.server.log",
        )
        print(
            f"*\n*\n*\nStarting Test:test_infer_offset_out_of_bound.{protocol}\n*\n*\n*\n"
        )

        error_msg = []
        shm_handles = self._configure_sever()
        if self._protocol == "http":
            # -32 when placed in an int64 signed type, to get a negative offset
            # by overflowing
            offset = 2**64 - 32
        else:
            # gRPC will throw an error if > 2**63 - 1, so instead test for
            # exceeding shm region size by 1 byte, given its size is 64 bytes
            offset = 64
        self._basic_inference(
            shm_handles[0],
            shm_handles[1],
            shm_handles[2],
            shm_handles[3],
            error_msg,
            shm_output_offset=offset,
        )
        self.assertEqual(len(error_msg), 1)
        self.assertIn("Invalid offset for shared memory region", error_msg[0])
        self._cleanup_server(shm_handles)
        self._test_passed = True


if __name__ == "__main__":
    unittest.main()
