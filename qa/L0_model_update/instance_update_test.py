#!/usr/bin/env python3

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

import concurrent.futures
import json
import os
import random
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from models.model_init_del.util import (
    disable_batching,
    enable_batching,
    get_count,
    reset_count,
    set_delay,
    update_instance_group,
    update_model_file,
    update_sequence_batching,
)
from tritonclient.utils import InferenceServerException


class TestInstanceUpdate(unittest.TestCase):
    _model_name = "model_init_del"

    def setUp(self):
        # Reset counters
        reset_count("initialize")
        reset_count("finalize")
        # Reset batching
        disable_batching()
        # Reset delays
        set_delay("initialize", 0)
        set_delay("infer", 0)
        # Reset sequence batching
        update_sequence_batching("")
        # Initialize client
        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def tearDown(self):
        # Check if the test passed for this test case that is tearing down
        r = self._outcome.result
        passed = all(self != test_case for test_case, _ in r.errors + r.failures)
        if passed:
            # Do nothing if passed
            return
        # Best effort to reset the model state for the next test case
        self._triton.unload_model(self._model_name)
        time.sleep(30)  # time for instances to finish unloading

    def _get_inputs(self, batching=False):
        self.assertIsInstance(batching, bool)
        if batching:
            shape = [random.randint(1, 2), random.randint(1, 16)]
        else:
            shape = [random.randint(1, 16)]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _infer(self, batching=False):
        self._triton.infer(self._model_name, self._get_inputs(batching))

    def _concurrent_infer(self, concurrency=4, batching=False):
        pool = concurrent.futures.ThreadPoolExecutor()
        stop = [False]

        def repeat_infer():
            while not stop[0]:
                self._infer(batching)

        infer_threads = [pool.submit(repeat_infer) for i in range(concurrency)]

        def stop_infer():
            stop[0] = True
            [t.result() for t in infer_threads]
            pool.shutdown()

        return stop_infer

    def _check_count(self, kind, expected_count, poll=False):
        self.assertIsInstance(poll, bool)
        if poll:
            timeout = 30  # seconds
            poll_interval = 0.1  # seconds
            max_retry = timeout / poll_interval
            num_retry = 0
            while num_retry < max_retry and get_count(kind) < expected_count:
                time.sleep(poll_interval)
                num_retry += 1
        self.assertEqual(get_count(kind), expected_count)

    def _load_model(self, instance_count, instance_config="", batching=False):
        # Set batching
        enable_batching() if batching else disable_batching()
        # Load model
        self._update_instance_count(
            instance_count, 0, instance_config, batching=batching
        )

    def _update_instance_count(
        self,
        add_count,
        del_count,
        instance_config="",
        wait_for_finalize=False,
        batching=False,
    ):
        self.assertIsInstance(add_count, int)
        self.assertGreaterEqual(add_count, 0)
        self.assertIsInstance(del_count, int)
        self.assertGreaterEqual(del_count, 0)
        self.assertIsInstance(instance_config, str)
        prev_initialize_count = get_count("initialize")
        prev_finalize_count = get_count("finalize")
        new_initialize_count = prev_initialize_count + add_count
        new_finalize_count = prev_finalize_count + del_count
        if len(instance_config) == 0:
            prev_count = prev_initialize_count - prev_finalize_count
            new_count = prev_count + add_count - del_count
            instance_config = "{\ncount: " + str(new_count) + "\nkind: KIND_CPU\n}"
        update_instance_group(instance_config)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", new_initialize_count)
        self._check_count("finalize", new_finalize_count, wait_for_finalize)
        self._infer(batching)

    def _unload_model(self, batching=False):
        prev_initialize_count = get_count("initialize")
        self._triton.unload_model(self._model_name)
        self._check_count("initialize", prev_initialize_count)
        self._check_count("finalize", prev_initialize_count, True)
        with self.assertRaises(InferenceServerException):
            self._infer(batching)

    # Test add -> remove -> add an instance without batching
    def test_add_rm_add_instance_no_batching(self):
        self._load_model(3, batching=False)
        stop = self._concurrent_infer(batching=False)
        self._update_instance_count(1, 0, batching=False)  # add
        self._update_instance_count(0, 1, batching=False)  # remove
        self._update_instance_count(1, 0, batching=False)  # add
        stop()
        self._unload_model(batching=False)

    # Test add -> remove -> add an instance with batching
    def test_add_rm_add_instance_with_batching(self):
        self._load_model(4, batching=True)
        stop = self._concurrent_infer(batching=True)
        self._update_instance_count(1, 0, batching=True)  # add
        self._update_instance_count(0, 1, batching=True)  # remove
        self._update_instance_count(1, 0, batching=True)  # add
        stop()
        self._unload_model(batching=True)

    # Test remove -> add -> remove an instance without batching
    def test_rm_add_rm_instance_no_batching(self):
        self._load_model(2, batching=False)
        stop = self._concurrent_infer(batching=False)
        self._update_instance_count(0, 1, batching=False)  # remove
        self._update_instance_count(1, 0, batching=False)  # add
        self._update_instance_count(0, 1, batching=False)  # remove
        stop()
        self._unload_model(batching=False)

    # Test remove -> add -> remove an instance with batching
    def test_rm_add_rm_instance_with_batching(self):
        self._load_model(3, batching=True)
        stop = self._concurrent_infer(batching=True)
        self._update_instance_count(0, 1, batching=True)  # remove
        self._update_instance_count(1, 0, batching=True)  # add
        self._update_instance_count(0, 1, batching=True)  # remove
        stop()
        self._unload_model(batching=True)

    # Test reduce instance count to zero
    def test_rm_instance_to_zero(self):
        self._load_model(1)
        # Setting instance group count to 0 will be overwritten to 1, so no
        # instances should be created or removed.
        self._update_instance_count(0, 0, "{\ncount: 0\nkind: KIND_CPU\n}")
        self._unload_model()

    # Test add/remove multiple CPU instances at a time
    def test_cpu_instance_update(self):
        self._load_model(8)
        self._update_instance_count(0, 4)  # remove 4 instances
        self._update_instance_count(0, 3)  # remove 3 instances
        self._update_instance_count(0, 0)  # no change
        time.sleep(0.1)  # larger the gap for config.pbtxt timestamp to update
        self._update_instance_count(2, 0)  # add 2 instances
        self._update_instance_count(5, 0)  # add 5 instances
        self._unload_model()

    # Test add/remove multiple GPU instances at a time
    def test_gpu_instance_update(self):
        self._load_model(6, "{\ncount: 6\nkind: KIND_GPU\n}")
        self._update_instance_count(0, 2, "{\ncount: 4\nkind: KIND_GPU\n}")
        self._update_instance_count(3, 0, "{\ncount: 7\nkind: KIND_GPU\n}")
        self._unload_model()

    # Test add/remove multiple CPU/GPU instances at a time
    def test_gpu_cpu_instance_update(self):
        # Load model with 1 GPU instance and 2 CPU instance
        self._load_model(
            3, "{\ncount: 2\nkind: KIND_CPU\n},\n{\ncount: 1\nkind: KIND_GPU\n}"
        )
        # Add 2 GPU instance and remove 1 CPU instance
        self._update_instance_count(
            2, 1, "{\ncount: 1\nkind: KIND_CPU\n},\n{\ncount: 3\nkind: KIND_GPU\n}"
        )
        # Shuffle the instances
        self._update_instance_count(
            0, 0, "{\ncount: 3\nkind: KIND_GPU\n},\n{\ncount: 1\nkind: KIND_CPU\n}"
        )
        time.sleep(0.1)  # larger the gap for config.pbtxt timestamp to update
        # Remove 1 GPU instance and add 1 CPU instance
        self._update_instance_count(
            1, 1, "{\ncount: 2\nkind: KIND_GPU\n},\n{\ncount: 2\nkind: KIND_CPU\n}"
        )
        # Unload model
        self._unload_model()

    # Test model instance name update
    def test_instance_name_update(self):
        # Load 3 instances with 2 different names
        self._load_model(
            3,
            '{\nname: "old_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "old_2"\ncount: 2\nkind: KIND_GPU\n}',
        )
        # Change the instance names
        self._update_instance_count(
            0,
            0,
            '{\nname: "new_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "new_2"\ncount: 2\nkind: KIND_GPU\n}',
        )
        # Unload model
        self._unload_model()

    # Test instance signature grouping
    def test_instance_signature(self):
        # Load 2 GPU instances and 3 CPU instances
        self._load_model(
            5,
            '{\nname: "GPU_group"\ncount: 2\nkind: KIND_GPU\n},\n{\nname: "CPU_group"\ncount: 3\nkind: KIND_CPU\n}',
        )
        # Flatten the instances representation
        self._update_instance_count(
            0,
            0,
            '{\nname: "CPU_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_2_3"\ncount: 2\nkind: KIND_CPU\n},\n{\nname: "GPU_1"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "GPU_2"\ncount: 1\nkind: KIND_GPU\n}',
        )
        time.sleep(0.1)  # larger the gap for config.pbtxt timestamp to update
        # Consolidate different representations
        self._update_instance_count(
            0,
            0,
            '{\nname: "CPU_group"\ncount: 3\nkind: KIND_CPU\n},\n{\nname: "GPU_group"\ncount: 2\nkind: KIND_GPU\n}',
        )
        time.sleep(0.1)  # larger the gap for config.pbtxt timestamp to update
        # Flatten the instances representation
        self._update_instance_count(
            0,
            0,
            '{\nname: "GPU_1"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "GPU_2"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "CPU_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_2"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_3"\ncount: 1\nkind: KIND_CPU\n}',
        )
        # Unload model
        self._unload_model()

    # Test instance update with invalid instance group config
    def test_invalid_config(self):
        # Load model with 8 instances
        self._load_model(8)
        # Set invalid config
        update_instance_group("--- invalid config ---")
        with self.assertRaises(InferenceServerException):
            self._triton.load_model("model_init_del")
        # Correct config by reducing instances to 4
        self._update_instance_count(0, 4)
        # Unload model
        self._unload_model()

    # Test instance update with model file changed
    def test_model_file_update(self):
        self._load_model(5)
        update_model_file()
        self._update_instance_count(
            6, 5, "{\ncount: 6\nkind: KIND_CPU\n}", wait_for_finalize=True
        )
        self._unload_model()

    # Test instance update with non instance config changed in config.pbtxt
    def test_non_instance_config_update(self):
        self._load_model(4, batching=False)
        enable_batching()
        self._update_instance_count(
            2,
            4,
            "{\ncount: 2\nkind: KIND_CPU\n}",
            wait_for_finalize=True,
            batching=True,
        )
        self._unload_model(batching=True)

    # Test passing new instance config via load API
    def test_load_api_with_config(self):
        # Load model with 1 instance
        self._load_model(1)
        # Get the model config from Triton
        config = self._triton.get_model_config(self._model_name, as_json=True)
        self.assertIn("config", config)
        self.assertIsInstance(config["config"], dict)
        config = config["config"]
        self.assertIn("instance_group", config)
        self.assertIsInstance(config["instance_group"], list)
        self.assertEqual(len(config["instance_group"]), 1)
        self.assertIn("count", config["instance_group"][0])
        self.assertIsInstance(config["instance_group"][0]["count"], int)
        # Add an extra instance into the model config
        config["instance_group"][0]["count"] += 1
        self.assertEqual(config["instance_group"][0]["count"], 2)
        # Load the extra instance via the load API
        self._triton.load_model(self._model_name, config=json.dumps(config))
        self._check_count("initialize", 2)  # 2 instances in total
        self._check_count("finalize", 0)  # no instance is removed
        self._infer()
        # Unload model
        self._unload_model()

    # Test instance update with an ongoing inference
    def test_update_while_inferencing(self):
        # Load model with 1 instance
        self._load_model(1)
        # Add 1 instance while inferencing
        set_delay("infer", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            infer_start_time = time.time()
            infer_thread = pool.submit(self._infer)
            time.sleep(2)  # make sure inference has started
            update_start_time = time.time()
            update_thread = pool.submit(self._triton.load_model, self._model_name)
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
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        self._infer()
        # Unload model
        self._unload_model()

    # Test inference with an ongoing instance update
    def test_infer_while_updating(self):
        # Load model with 1 instance
        self._load_model(1)
        # Infer while adding 1 instance
        set_delay("initialize", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            update_start_time = time.time()
            update_thread = pool.submit(self._triton.load_model, self._model_name)
            time.sleep(2)  # make sure update has started
            infer_start_time = time.time()
            infer_thread = pool.submit(self._infer)
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
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        self._infer()
        # Unload model
        self._unload_model()

    # Test instance resource requirement increase
    @unittest.skipUnless(
        "execution_count" in os.environ["RATE_LIMIT_MODE"],
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_increase(self):
        # Load model
        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 2\n}\n]\n}\n}',
        )
        # Increase resource requirement
        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 8\n}\n]\n}\n}',
        )
        # Check the model is not blocked from infer due to the default resource
        # possibly not updated to the larger resource requirement.
        infer_count = 8
        infer_complete = [False for i in range(infer_count)]

        def infer():
            for i in range(infer_count):
                self._infer()
                infer_complete[i] = True

        with concurrent.futures.ThreadPoolExecutor() as pool:
            infer_thread = pool.submit(infer)
            time.sleep(infer_count / 2)  # each infer should take < 0.5 seconds
            self.assertNotIn(False, infer_complete, "Infer possibly stuck")
            infer_thread.result()
        # Unload model
        self._unload_model()

    # Test instance resource requirement increase above explicit resource
    @unittest.skipUnless(
        os.environ["RATE_LIMIT_MODE"] == "execution_count_with_explicit_resource",
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_increase_above_explicit(self):
        # Load model
        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 2\n}\n]\n}\n}',
        )
        # Increase resource requirement
        with self.assertRaises(InferenceServerException):
            self._update_instance_count(
                0,
                0,
                '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 32\n}\n]\n}\n}',
            )
        # Correct the resource requirement to match the explicit resource
        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 10\n}\n]\n}\n}',
        )
        # Unload model
        self._unload_model()

    # Test instance resource requirement decrease
    @unittest.skipUnless(
        "execution_count" in os.environ["RATE_LIMIT_MODE"],
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_decrease(self):
        # Load model
        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 4\n}\n]\n}\n}',
        )
        # Decrease resource requirement
        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 3\n}\n]\n}\n}',
        )
        # Unload model
        self._unload_model()
        # The resource count of 3 is unique across this entire test, so check
        # the server output to make sure it is printed, which ensures the
        # max resource is actually decreased.
        time.sleep(1)  # make sure the log file is updated
        log_path = os.path.join(
            os.environ["MODEL_LOG_DIR"],
            "instance_update_test.rate_limit_"
            + os.environ["RATE_LIMIT_MODE"]
            + ".server.log",
        )
        with open(log_path, mode="r", encoding="utf-8", errors="strict") as f:
            if os.environ["RATE_LIMIT_MODE"] == "execution_count":
                # Make sure the previous max resource limit of 4 is reduced to 3
                # when no explicit limit is set.
                self.assertIn("Resource: R1\\t Count: 3", f.read())
            else:
                # Make sure the max resource limit is never set to 3 when
                # explicit limit of 10 is set.
                self.assertNotIn("Resource: R1\\t Count: 3", f.read())

    _direct_sequence_batching_str = (
        "direct { }\nmax_sequence_idle_microseconds: 8000000"
    )
    _oldest_sequence_batching_str = (
        "oldest { max_candidate_sequences: 4 }\nmax_sequence_idle_microseconds: 8000000"
    )

    # Test instance update for direct scheduler without any ongoing sequences
    def test_direct_scheduler_update_no_ongoing_sequences(self):
        self._test_scheduler_update_no_ongoing_sequences(
            self._direct_sequence_batching_str
        )

    # Test instance update for direct scheduler with any ongoing sequences
    def test_direct_scheduler_update_with_ongoing_sequences(self):
        self._test_scheduler_update_with_ongoing_sequences(
            self._direct_sequence_batching_str
        )

    # Test instance update for oldest scheduler without ongoing sequences
    def test_oldest_scheduler_update_no_ongoing_sequences(self):
        self._test_scheduler_update_no_ongoing_sequences(
            self._oldest_sequence_batching_str
        )

    # Test instance update for oldest scheduler with ongoing sequences
    def test_oldest_scheduler_update_with_ongoing_sequences(self):
        self._test_scheduler_update_with_ongoing_sequences(
            self._oldest_sequence_batching_str
        )

    # Helper function for testing the success of sequence instance updates
    # without any ongoing sequences.
    def _test_scheduler_update_no_ongoing_sequences(self, sequence_batching_str):
        # Load model
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        update_sequence_batching(sequence_batching_str)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        # Basic sequence inference
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=1)
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )
        # Add 2 instances without in-flight sequence
        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 0)
        # Basic sequence inference
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )
        # Remove 1 instance without in-flight sequence
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 1, poll=True)
        # Basic sequence inference
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )
        # Unload model
        self._triton.unload_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 4, poll=True)

    # Helper function for testing if ongoing sequences may continue to infer on
    # the same instance after the instance processing the sequence is removed
    # from an instance update, which the removed instance will live until the
    # sequences end.
    def _test_scheduler_update_with_ongoing_sequences(self, sequence_batching_str):
        # Load model
        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        update_sequence_batching(sequence_batching_str)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 3)
        self._check_count("finalize", 0)
        # Start sequence 1 and 2 on CPU instances
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=2, sequence_start=True
        )
        # Remove all 3 CPU and add 1 GPU instance with in-flight sequences. Both
        # in-flight sequences are assigned to any 2 CPU instances, so exactly 1
        # CPU instance can be removed immediately.
        update_instance_group("{\ncount: 1\nkind: KIND_GPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)  # 3 CPU + 1 GPU
        self._check_count("finalize", 1, poll=True)  # 1 CPU
        # Sequence 1 and 2 may continue to infer
        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=1)
        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=2)
        self._check_count("finalize", 1)  # check 2 CPU instances not removed
        # Start sequence 3 on GPU instance
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=3, sequence_start=True
        )
        self._check_count("finalize", 1)  # check 2 CPU instances not removed
        # End sequence 1 and 2 will remove the 2 CPU instances
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=2, sequence_end=True
        )
        self._check_count("finalize", 3, poll=True)  # 3 CPU
        # End sequence 3
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=3, sequence_end=True
        )
        # Unload model
        self._triton.unload_model(self._model_name)
        self._check_count("initialize", 4)  # 3 CPU + 1 GPU
        self._check_count("finalize", 4, poll=True)  # 3 CPU + 1 GPU


if __name__ == "__main__":
    unittest.main()
