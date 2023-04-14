#!/usr/bin/python

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import sys
import unittest
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import json
from google.protobuf import json_format
import test_util as tu


# Similar set up as dynamic batcher tests
class TraceEndpointTest(tu.TestResultCollector):

    def tearDown(self):
        # Clear all trace settings to initial state.
        # Note that the tearDown function uses HTTP client so the pass/fail
        # of the HTTP trace setting test cases should be checked to make sure
        # tearDown() is properly executed and not affecting start state of
        # other test cases
        clear_settings = {
            "trace_file": None,
            "trace_level": None,
            "trace_rate": None,
            "trace_count": None,
            "log_frequency": None
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        triton_client.update_trace_settings(model_name="simple",
                                            settings=clear_settings)
        triton_client.update_trace_settings(model_name=None,
                                            settings=clear_settings)

    def check_server_initial_state(self):
        # Helper function to make sure the trace setting is properly
        # initialized / reset before actually running the test case.
        # Note that this function uses HTTP client so the pass/fail of
        # the HTTP trace setting test cases should be checked to make sure
        # the initial state is checked properly before running other test cases.
        initial_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(initial_settings,
                         triton_client.get_trace_settings(model_name="simple"))
        self.assertEqual(initial_settings, triton_client.get_trace_settings())

    def test_http_get_settings(self):
        # Model trace settings will be the same as global trace settings since
        # no update has been made.
        initial_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(initial_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected initial model trace settings")
        self.assertEqual(initial_settings, triton_client.get_trace_settings(),
                         "Unexpected initial global settings")

    def test_grpc_get_settings(self):
        # Model trace settings will be the same as global trace settings since
        # no update has been made.
        initial_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["global_unittest.log"]
                    },
                    "trace_level": {
                        "value": ["TIMESTAMPS"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["0"]
                    },
                }
            }), initial_settings)

        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        self.assertEqual(initial_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected initial model trace settings")
        self.assertEqual(initial_settings, triton_client.get_trace_settings(),
                         "Unexpected initial global settings")

    def test_http_update_settings(self):
        # Update model and global trace settings in order,
        # and expect the global trace settings will only reflect to
        # the model setting fields that haven't been specified.
        self.check_server_initial_state()

        expected_first_model_settings = {
            "trace_file": "model.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }
        expected_second_model_settings = {
            "trace_file": "model.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }
        expected_global_settings = {
            "trace_file": "another.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }

        model_update_settings = {"trace_file": "model.log"}
        global_update_settings = {
            "trace_file": "another.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"]
        }

        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(
            expected_first_model_settings,
            triton_client.update_trace_settings(model_name="simple",
                                                settings=model_update_settings),
            "Unexpected updated model trace settings")
        # Note that 'trace_level' may be mismatch due to the order of
        # the levels listed, currently we assume the order is the same
        # for simplicity. But the order shouldn't be enforced and this checking
        # needs to be improved when this kind of failure is reported
        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(
                settings=global_update_settings),
            "Unexpected updated global settings")
        self.assertEqual(expected_second_model_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected model trace settings after global update")

    def test_grpc_update_settings(self):
        # Update model and global trace settings in order,
        # and expect the global trace settings will only reflect to
        # the model setting fields that haven't been specified.
        self.check_server_initial_state()

        expected_first_model_settings = grpcclient.service_pb2.TraceSettingResponse(
        )
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["model.log"]
                    },
                    "trace_level": {
                        "value": ["TIMESTAMPS"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["0"]
                    },
                }
            }), expected_first_model_settings)

        expected_second_model_settings = grpcclient.service_pb2.TraceSettingResponse(
        )
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["model.log"]
                    },
                    "trace_level": {
                        "value": ["TIMESTAMPS", "TENSORS"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["0"]
                    },
                }
            }), expected_second_model_settings)

        expected_global_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["another.log"]
                    },
                    "trace_level": {
                        "value": ["TIMESTAMPS", "TENSORS"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["0"]
                    },
                }
            }), expected_global_settings)

        model_update_settings = {"trace_file": "model.log"}
        global_update_settings = {
            "trace_file": "another.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"]
        }

        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        self.assertEqual(
            expected_first_model_settings,
            triton_client.update_trace_settings(model_name="simple",
                                                settings=model_update_settings),
            "Unexpected updated model trace settings")
        # Note that 'trace_level' may be mismatch due to the order of
        # the levels listed, currently we assume the order is the same
        # for simplicity. But the order shouldn't be enforced and this checking
        # needs to be improved when this kind of failure is reported
        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(
                settings=global_update_settings),
            "Unexpected updated global settings")
        self.assertEqual(expected_second_model_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected model trace settings after global update")

    def test_http_clear_settings(self):
        # Clear global and model trace settings in order,
        # and expect the default / global trace settings are
        # propagated properly.
        self.check_server_initial_state()

        # First set up the model / global trace setting that:
        # model 'simple' has 'trace_rate' and 'log_frequency' specified
        # global has 'trace_level', 'trace_count' and 'trace_rate' specified
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        triton_client.update_trace_settings(model_name="simple",
                                            settings={
                                                "trace_rate": "12",
                                                "log_frequency": "34"
                                            })
        triton_client.update_trace_settings(settings={
            "trace_rate": "56",
            "trace_count": "78",
            "trace_level": ["OFF"]
        })

        expected_global_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0"
        }
        expected_first_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "12",
            "trace_count": "-1",
            "log_frequency": "34"
        }
        expected_second_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "34"
        }
        global_clear_settings = {"trace_rate": None, "trace_count": None}
        model_clear_settings = {"trace_rate": None, "trace_level": None}

        # Clear global
        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_clear_settings),
            "Unexpected cleared global trace settings")
        self.assertEqual(expected_first_model_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected model trace settings after global clear")
        self.assertEqual(
            expected_second_model_settings,
            triton_client.update_trace_settings(model_name="simple",
                                                settings=model_clear_settings),
            "Unexpected model trace settings after model clear")
        self.assertEqual(expected_global_settings,
                         triton_client.get_trace_settings(),
                         "Unexpected global trace settings after model clear")

    def test_grpc_clear_settings(self):
        # Clear global and model trace settings in order,
        # and expect the default / global trace settings are
        # propagated properly.
        self.check_server_initial_state()

        # First set up the model / global trace setting that:
        # model 'simple' has 'trace_rate' and 'log_frequency' specified
        # global has 'trace_level', 'trace_count' and 'trace_rate' specified
        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        triton_client.update_trace_settings(model_name="simple",
                                            settings={
                                                "trace_rate": "12",
                                                "log_frequency": "34"
                                            })
        triton_client.update_trace_settings(settings={
            "trace_rate": "56",
            "trace_count": "78",
            "trace_level": ["OFF"]
        })

        expected_global_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["global_unittest.log"]
                    },
                    "trace_level": {
                        "value": ["OFF"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["0"]
                    },
                }
            }), expected_global_settings)
        expected_first_model_settings = grpcclient.service_pb2.TraceSettingResponse(
        )
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["global_unittest.log"]
                    },
                    "trace_level": {
                        "value": ["OFF"]
                    },
                    "trace_rate": {
                        "value": ["12"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["34"]
                    },
                }
            }), expected_first_model_settings)
        expected_second_model_settings = grpcclient.service_pb2.TraceSettingResponse(
        )
        json_format.Parse(
            json.dumps({
                "settings": {
                    "trace_file": {
                        "value": ["global_unittest.log"]
                    },
                    "trace_level": {
                        "value": ["OFF"]
                    },
                    "trace_rate": {
                        "value": ["1"]
                    },
                    "trace_count": {
                        "value": ["-1"]
                    },
                    "log_frequency": {
                        "value": ["34"]
                    },
                }
            }), expected_second_model_settings)

        global_clear_settings = {"trace_rate": None, "trace_count": None}
        model_clear_settings = {"trace_rate": None, "trace_level": None}

        # Clear global
        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_clear_settings),
            "Unexpected cleared global trace settings")
        self.assertEqual(expected_first_model_settings,
                         triton_client.get_trace_settings(model_name="simple"),
                         "Unexpected model trace settings after global clear")
        self.assertEqual(
            expected_second_model_settings,
            triton_client.update_trace_settings(model_name="simple",
                                                settings=model_clear_settings),
            "Unexpected model trace settings after model clear")
        self.assertEqual(expected_global_settings,
                         triton_client.get_trace_settings(),
                         "Unexpected global trace settings after model clear")


if __name__ == '__main__':
    unittest.main()
