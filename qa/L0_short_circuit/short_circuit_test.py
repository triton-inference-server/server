# Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
import threading
import unittest
import json
import numpy as np
import test_util as tu
import tritonclient.http as httpclient
from google.protobuf import text_format, json_format
import tritonclient.grpc.model_config_pb2 as mc


class ShortCircuitTest(tu.TestResultCollector):
    def setUp(self):
        self._triton_client = httpclient.InferenceServerClient("localhost:8000")
    
    # def tearDown(self):


    def read_entire_file(self, filename):
        file = open(filename, mode="r")
        data = file.read()
        file.close()
        return data

    # timeout_limit is in seconds
    def wait_or_timeout_model_load(self, model_name, model_version, timeout_limit=10):
        timeout = 0;
        while (self._triton_client.is_model_ready(model_name=model_name, model_version=model_version) == False and timeout < timeout_limit):
            print("Model is not ready, sleeping [{}/{}]".format(timeout, timeout_limit))
            time.sleep(1)
            timeout += 1


    # Really should be more like
    # 1. update config with starting instance group
    # 2. Call load_model 
    # 3. Call get_config
    # 4. Edit the config to modify the insance_group
    # 5. Call load_model with the new config
    # 6. Call get_config
    # 7. Assert the instance_group is has the correct configuration
    def run_test(self, backend_name, model_name, model_version, start_count, new_instance_groups, end_count):
        client = self._triton_client
        
        client.load_model(model_name=model_name)
        self.wait_or_timeout_model_load(model_name, model_version)

        config = client.get_model_config(model_name=model_name, model_version=model_version)
        self.assertEqual(config["instance_group"][0]["count"], start_count)

        config["instance_group"] = new_instance_groups

        client.load_model(model_name=model_name, config=json.dumps(config))
        time.sleep(2)

        config = client.get_model_config(model_name=model_name, model_version=model_version)
        self.assertEqual(config["instance_group"][0]["count"], end_count)

    def set_instance_group_proto(self, filename, instance_group):
        file = open(filename, "r+")
        config = text_format.Parse(file.read(), mc.ModelConfig())
        file.truncate(0)
        file.seek(0)
        print("config before addition: ")
        print(config)
        text_format.Merge(instance_group, config)
        print("config after addition: ")
        print(config)
        file.write(text_format.MessageToString(config))
        file.close()
        
    def set_instance_group_json(self, filename):
        pass

    def test_increase_instance_count(self):
        start_instance_group =           \
            'name: "increase_count"      \
            instance_group {             \
                name: "increase_count",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/increase_count/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count/config.pbtxt')

        self.run_test("tensorflow", "increase_count", "1", 1, end_instance_group, 2)

    def test_decrease_instance_count(self):
        start_instance_group =           \
            'name: "decrease_count"      \
            instance_group {             \
                name: "decrease_count",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/decrease_count/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "decrease_count",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/decrease_count/config.pbtxt')

        self.run_test("tensorflow", "decrease_count", "1", 2, end_instance_group, 1)

    def test_decrease_instance_count_past_zero(self):
        start_instance_group =           \
            'name: "decrease_count_past_zero"      \
            instance_group {             \
                name: "decrease_count_past_zero",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/decrease_count_past_zero/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "decrease_count_past_zero",  \
                "count": -1,               \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/decrease_count_past_zero/config.pbtxt')

        self.run_test("tensorflow", "decrease_count_past_zero", "1", 2, end_instance_group, 1)
    
    def test_increase_instance_count_no_config(self):
        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count_no_config",  \
                "count": 2,               \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count_no_config/config.pbtxt')

        self.run_test("tensorflow", "increase_count_no_config", "1", 1, end_instance_group, 2)
        

if __name__ == '__main__':
    unittest.main()
