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
# import time
# import threading
import unittest
import json
# import numpy as np
# import test_util as tu
import tritonclient.http as httpclient
# from google.protobuf import text_format, json_format
# import tritonclient.grpc.model_config_pb2 as mc
from short_circuit_util import ShortCircuitUtil as scutil

class ShortCircuitTest(scutil):
    # def setUp(self):
    #     self._triton_client = httpclient.InferenceServerClient("localhost:8000")

    def test_different_field(self):
        start_instance_group =           \
            'name: "different_field"      \
            instance_group {             \
                name: "different_field",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/different_field/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "different_field",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU",         \
                "rate_limiter": {      \
                    "resources": [{            \
                        "name": "resource_1",  \
                        "global": true,        \
                        "count": 1           \
                    }],                     \
                    "priority": 1          \
                }                         \
            }]                             \
        ')
        self.set_instance_group_json('models/different_field/config.pbtxt')

        self.run_test_single("different_field", "1", 1, end_instance_group, 1)


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

        self.run_test_single("increase_count", "1", 1, end_instance_group, 2)

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

        self.run_test_single("decrease_count", "1", 2, end_instance_group, 1)

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

        self.run_test_single("decrease_count_past_zero", "1", 2, end_instance_group, 1)
    
    def test_increase_instance_count_no_config(self):
        # if the directory doesn't exist then we don't run the test
        if not os.path.isdir('models/increase_count_no_config'):
            return    

        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count_no_config",  \
                "count": 2,               \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count_no_config/config.pbtxt')

        self.run_test_single("increase_count_no_config", "1", 1, end_instance_group, 2)
        
    def test_increase_instance_count_some_multiple(self):
        start_instance_group =           \
            'name: "increase_count_some_multiple"      \
            instance_group {             \
                name: "increase_count_some_multiple_1",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "increase_count_some_multiple_2",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/increase_count_some_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count_some_multiple_1",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "increase_count_some_multiple_2",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count_some_multiple/config.pbtxt')

        self.run_test_multiple("increase_count_some_multiple", "1", [1, 1], end_instance_group, [1,2])

    def test_increase_instance_count_all_multiple(self):
        start_instance_group =           \
            'name: "increase_count_all_multiple"      \
            instance_group {             \
                name: "increase_count_all_multiple_1",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "increase_count_all_multiple_2",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/increase_count_all_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count_all_multiple_1",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "increase_count_all_multiple_2",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count_all_multiple/config.pbtxt')

        self.run_test_multiple("increase_count_all_multiple", "1", [1, 1], end_instance_group, [2,2])

    def test_increase_instance_count_rearrange_multiple(self):
        start_instance_group =           \
            'name: "increase_count_rearrange_multiple"      \
            instance_group {             \
                name: "increase_count_rearrange_multiple_1",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "increase_count_rearrange_multiple_2",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/increase_count_rearrange_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "increase_count_rearrange_multiple_2",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "increase_count_rearrange_multiple_1",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/increase_count_rearrange_multiple/config.pbtxt')

    def test_rearrange_multiple(self):
        start_instance_group =           \
            'name: "rearrange_multiple"      \
            instance_group {             \
                name: "rearrange_multiple_1",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "rearrange_multiple_2",  \
                count: 1,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/rearrange_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "rearrange_multiple_2",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "rearrange_multiple_1",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/rearrange_multiple/config.pbtxt')

        self.run_test_multiple("rearrange_multiple", "1", [1, 1], end_instance_group, [1,1])

    def test_decrease_instance_count_some_multiple(self):
        start_instance_group =           \
            'name: "decrease_count_some_multiple"      \
            instance_group {             \
                name: "decrease_count_some_multiple_1",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "decrease_count_some_multiple_2",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/decrease_count_some_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "decrease_count_some_multiple_1",  \
                "count": 2,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "decrease_count_some_multiple_2",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/decrease_count_some_multiple/config.pbtxt')

        self.run_test_multiple("decrease_count_some_multiple", "1", [2, 2], end_instance_group, [2,1])

    def test_decrease_instance_count_all_multiple(self):
        start_instance_group =           \
            'name: "decrease_count_all_multiple"      \
            instance_group {             \
                name: "decrease_count_all_multiple_1",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }                           \
            instance_group {             \
                name: "decrease_count_all_multiple_2",  \
                count: 2,                \
                gpus: 0                  \
                kind: KIND_GPU,          \
            }'
        self.set_instance_group_proto('models/decrease_count_all_multiple/config.pbtxt', start_instance_group)

        end_instance_group = json.loads(
            '[{                            \
                "name": "decrease_count_all_multiple_1",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            },                              \
            {                            \
                "name": "decrease_count_all_multiple_2",  \
                "count": 1,                \
                "gpus": 0,                 \
                "kind": "KIND_GPU"         \
            }]                             \
        ')
        self.set_instance_group_json('models/decrease_count_all_multiple/config.pbtxt')

        self.run_test_multiple( "decrease_count_all_multiple", "1", [2, 2], end_instance_group, [1, 1])

if __name__ == '__main__':
    unittest.main()
