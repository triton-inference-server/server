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
from pathlib import Path
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from models.instance_init_del.util import get_initialize_count, reset_initialize_count, get_finalize_count, reset_finalize_count, update_instance_group


class TestInstanceUpdate(unittest.TestCase):

    def test_add_rm_add(self):
        # Initialize client
        triton = grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=True)
        # Prepare infer data
        inputs = [grpcclient.InferInput("INPUT0", (8,), "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape=(8,), dtype=np.float32))
        # Reset counters
        reset_initialize_count()
        reset_finalize_count()
        # Load model
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 3)
        self.assertEqual(get_finalize_count(), 0)
        triton.infer("instance_init_del", inputs)
        # Add 1 instance
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 4)
        self.assertEqual(get_finalize_count(), 0)
        triton.infer("instance_init_del", inputs)
        # Remove 1 instance
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 4)
        self.assertEqual(get_finalize_count(), 1)
        triton.infer("instance_init_del", inputs)
        # Add 1 instance
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 5)
        self.assertEqual(get_finalize_count(), 1)
        triton.infer("instance_init_del", inputs)
        # Unload model
        triton.unload_model("instance_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_initialize_count(), 5)
        self.assertEqual(get_finalize_count(), 5)
        with self.assertRaises(InferenceServerException):
            triton.infer("instance_init_del", inputs)

    def test_rm_add_rm(self):
        # Initialize client
        triton = grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=True)
        # Prepare infer data
        inputs = [grpcclient.InferInput("INPUT0", (4,), "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape=(4,), dtype=np.float32))
        # Reset counters
        reset_initialize_count()
        reset_finalize_count()
        # Load model
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 2)
        self.assertEqual(get_finalize_count(), 0)
        triton.infer("instance_init_del", inputs)
        # Remove 1 instance
        update_instance_group("{\ncount: 1\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 2)
        self.assertEqual(get_finalize_count(), 1)
        triton.infer("instance_init_del", inputs)
        # Add 1 instance
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 3)
        self.assertEqual(get_finalize_count(), 1)
        triton.infer("instance_init_del", inputs)
        # Unload model
        triton.unload_model("instance_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_initialize_count(), 3)
        self.assertEqual(get_finalize_count(), 3)
        with self.assertRaises(InferenceServerException):
            triton.infer("instance_init_del", inputs)

    def test_invalid_config(self):
        # Initialize client
        triton = grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=True)
        # Prepare infer data
        inputs = [grpcclient.InferInput("INPUT0", (16,), "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape=(16,), dtype=np.float32))
        # Reset counters
        reset_initialize_count()
        reset_finalize_count()
        # Load model
        update_instance_group("{\ncount: 8\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 8)
        self.assertEqual(get_finalize_count(), 0)
        triton.infer("instance_init_del", inputs)
        # Invalid config
        update_instance_group("--- invalid config ---")
        with self.assertRaises(InferenceServerException):
            triton.load_model("instance_init_del")
        # Correct config
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 8)
        self.assertEqual(get_finalize_count(), 4)
        triton.infer("instance_init_del", inputs)
        # Unload model
        triton.unload_model("instance_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_initialize_count(), 8)
        self.assertEqual(get_finalize_count(), 8)
        with self.assertRaises(InferenceServerException):
            triton.infer("instance_init_del", inputs)

    def test_model_file_update(self):
        # Initialize client
        triton = grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=True)
        # Prepare infer data
        inputs = [grpcclient.InferInput("INPUT0", (1,), "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape=(1,), dtype=np.float32))
        # Reset counters
        reset_initialize_count()
        reset_finalize_count()
        # Load model
        update_instance_group("{\ncount: 5\nkind: KIND_CPU\n}")
        triton.load_model("instance_init_del")
        self.assertEqual(get_initialize_count(), 5)
        self.assertEqual(get_finalize_count(), 0)
        triton.infer("instance_init_del", inputs)
        # Update instance and model file
        update_instance_group("{\ncount: 6\nkind: KIND_CPU\n}")
        Path("models/instance_init_del/1/model.py").touch()
        triton.load_model("instance_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_initialize_count(), 11)
        self.assertEqual(get_finalize_count(), 5)
        triton.infer("instance_init_del", inputs)
        # Unload model
        triton.unload_model("instance_init_del")
        time.sleep(10)  # wait for unload to complete
        self.assertEqual(get_initialize_count(), 11)
        self.assertEqual(get_finalize_count(), 11)
        with self.assertRaises(InferenceServerException):
            triton.infer("instance_init_del", inputs)


if __name__ == "__main__":
    unittest.main()
