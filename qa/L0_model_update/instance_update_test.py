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

import unittest
import time
import concurrent.futures
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from models.model_init_del.util import (get_count, reset_count, set_delay,
                                        update_instance_group,
                                        update_model_file, enable_batching,
                                        disable_batching)


class TestInstanceUpdate(unittest.TestCase):

    def setUp(self):
        # Reset counters
        reset_count("initialize")
        reset_count("finalize")
        # Reset batching
        disable_batching()
        # Reset delays
        set_delay("initialize", 0)
        set_delay("infer", 0)
        # Initialize client
        self.__triton = grpcclient.InferenceServerClient("localhost:8001",
                                                         verbose=True)

    def __get_inputs(self, shape=(2,)):
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def test_add_rm_add(self):
        # Load model
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 3)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((8,)))
        # Add 1 instance
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 4)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Remove 1 instance
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 4)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((4,)))
        # Add 1 instance
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 5)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((1,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 5)
        self.assertEqual(get_count("finalize"), 5)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((5,)))

    def test_rm_add_rm(self):
        # Load model
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((4,)))
        # Remove 1 instance
        update_instance_group("{\ncount: 1\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Add 1 instance
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 3)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((3,)))
        # Remove 1 instance
        update_instance_group("{\ncount: 1\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 3)
        self.assertEqual(get_count("finalize"), 2)
        self.__triton.infer("model_init_del", self.__get_inputs((8,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 3)
        self.assertEqual(get_count("finalize"), 3)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((12,)))

    def test_gpu_cpu_instance_mix(self):
        # Load model
        update_instance_group(
            "{\ncount: 2\nkind: KIND_CPU\n},\n{\ncount: 1\nkind: KIND_GPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 3)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Add 2 GPU instance and remove 1 CPU instance
        update_instance_group(
            "{\ncount: 1\nkind: KIND_CPU\n},\n{\ncount: 3\nkind: KIND_GPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 5)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((1,)))
        # Shuffle the instances
        update_instance_group(
            "{\ncount: 3\nkind: KIND_GPU\n},\n{\ncount: 1\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 5)
        self.assertEqual(get_count("finalize"), 1)
        self.__triton.infer("model_init_del", self.__get_inputs((4,)))
        # Remove 1 GPU instance and add 1 CPU instance
        update_instance_group(
            "{\ncount: 2\nkind: KIND_GPU\n},\n{\ncount: 2\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 6)
        self.assertEqual(get_count("finalize"), 2)
        self.__triton.infer("model_init_del", self.__get_inputs((1,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 6)
        self.assertEqual(get_count("finalize"), 6)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((2,)))

    def test_invalid_config(self):
        # Load model
        update_instance_group("{\ncount: 8\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 8)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((16,)))
        # Invalid config
        update_instance_group("--- invalid config ---")
        with self.assertRaises(InferenceServerException):
            self.__triton.load_model("model_init_del")
        # Correct config
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 8)
        self.assertEqual(get_count("finalize"), 4)
        self.__triton.infer("model_init_del", self.__get_inputs((9,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 8)
        self.assertEqual(get_count("finalize"), 8)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((8,)))

    def test_model_file_update(self):
        # Load model
        update_instance_group("{\ncount: 5\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 5)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((1,)))
        # Update instance and model file
        update_instance_group("{\ncount: 6\nkind: KIND_CPU\n}")
        update_model_file()
        self.__triton.load_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 11)
        self.assertEqual(get_count("finalize"), 5)
        self.__triton.infer("model_init_del", self.__get_inputs((3,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 11)
        self.assertEqual(get_count("finalize"), 11)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((4,)))

    def test_more_than_instance_update(self):
        # Load model
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 4)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Update batching and instance
        enable_batching()
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 6)
        self.assertEqual(get_count("finalize"), 4)
        self.__triton.infer("model_init_del", self.__get_inputs((1, 1)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 6)
        self.assertEqual(get_count("finalize"), 6)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((1, 4)))

    def test_update_while_inferencing(self):
        # Load model
        update_instance_group("{\ncount: 1\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 1)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Add 1 instance while inferencing
        set_delay("infer", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            infer_start_time = time.time()
            infer_thread = pool.submit(self.__triton.infer, "model_init_del",
                                       self.__get_inputs((1,)))
            time.sleep(2)  # make sure inference has started
            update_start_time = time.time()
            update_thread = pool.submit(self.__triton.load_model,
                                        "model_init_del")
            update_thread.result()
            update_end_time = time.time()
            infer_thread.result()
            infer_end_time = time.time()
        infer_time = infer_end_time - infer_start_time
        update_time = update_end_time - update_start_time
        # Adding a new instance does not depend on existing instances, so the
        # ongoing inference should not block the update.
        self.assertGreaterEqual(infer_time, 10.0, "Invalid infer time")
        self.assertLess(update_time, 5.0, "Update blocked by infer")
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((8,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 2)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((4,)))

    def test_infer_while_updating(self):
        # Load model
        update_instance_group("{\ncount: 1\nkind: KIND_CPU\n}")
        self.__triton.load_model("model_init_del")
        self.assertEqual(get_count("initialize"), 1)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((2,)))
        # Infer while adding 1 instance
        set_delay("initialize", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            update_start_time = time.time()
            update_thread = pool.submit(self.__triton.load_model,
                                        "model_init_del")
            time.sleep(2)  # make sure update has started
            infer_start_time = time.time()
            infer_thread = pool.submit(self.__triton.infer, "model_init_del",
                                       self.__get_inputs((1,)))
            infer_thread.result()
            infer_end_time = time.time()
            update_thread.result()
            update_end_time = time.time()
        update_time = update_end_time - update_start_time
        infer_time = infer_end_time - infer_start_time
        # Waiting on new instance creation should not block inference on
        # existing instances.
        self.assertGreaterEqual(update_time, 10.0, "Invalid update time")
        self.assertLess(infer_time, 5.0, "Infer blocked by update")
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 0)
        self.__triton.infer("model_init_del", self.__get_inputs((8,)))
        # Unload model
        self.__triton.unload_model("model_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_count("initialize"), 2)
        self.assertEqual(get_count("finalize"), 2)
        with self.assertRaises(InferenceServerException):
            self.__triton.infer("model_init_del", self.__get_inputs((4,)))


if __name__ == "__main__":
    unittest.main()
