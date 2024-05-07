#!/usr/bin/python

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import sys
import unittest

import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from google.protobuf import json_format
from tritonclient.utils import InferenceServerException


# Similar set up as dynamic batcher tests
class LogEndpointTest(tu.TestResultCollector):
    def tearDown(self):
        # Clear all log settings to initial state.
        # Note that the tearDown function uses HTTP client so the pass/fail
        # of the HTTP log setting test cases should be checked to make sure
        # tearDown() is properly executed and not affecting start state of
        # other test cases
        clear_settings = {
            "log_info": True,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        triton_client.update_log_settings(settings=clear_settings)

    def check_server_initial_state(self):
        # Helper function to make sure the log setting is properly
        # initialized / reset before actually running the test case.
        # Note that this function uses HTTP client so the pass/fail of
        # the HTTP log setting test cases should be checked to make sure
        # the initial state is checked properly before running other test cases.
        initial_settings = {
            "log_file": "",
            "log_info": True,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(initial_settings, triton_client.get_log_settings())

    def test_http_get_settings(self):
        # Log settings will be the same as default settings since
        # no update has been made.
        initial_settings = {
            "log_file": "",
            "log_info": True,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(
            initial_settings,
            triton_client.get_log_settings(),
            "Unexpected initial log settings",
        )

    def test_grpc_get_settings(self):
        # Log settings will be the same as default settings since
        # no update has been made.
        initial_settings = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": True},
                        "log_warning": {"boolParam": True},
                        "log_error": {"boolParam": True},
                        "log_verbose_level": {"uint32Param": 0},
                        "log_format": {"stringParam": "default"},
                    }
                }
            ),
            initial_settings,
        )
        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        self.assertEqual(
            initial_settings,
            triton_client.get_log_settings(),
            "Unexpected initial log settings",
        )

    def test_http_update_settings(self):
        # Update each possible log configuration
        # field and check that they are reflected
        # by the server
        self.check_server_initial_state()

        log_settings_1 = {
            "log_file": "log_file.log",
            "log_info": True,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_1 = {
            "error": "log file location can not be updated through network protocol"
        }

        log_settings_2 = {
            "log_info": False,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_2 = log_settings_2.copy()
        expected_log_settings_2["log_file"] = ""

        log_settings_3 = {
            "log_info": False,
            "log_warning": False,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_3 = log_settings_3.copy()
        expected_log_settings_3["log_file"] = ""

        log_settings_4 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_4 = log_settings_4.copy()
        expected_log_settings_4["log_file"] = ""

        log_settings_5 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 1,
            "log_format": "default",
        }
        expected_log_settings_5 = log_settings_5.copy()
        expected_log_settings_5["log_file"] = ""

        log_settings_6 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 1,
            "log_format": "ISO8601",
        }
        expected_log_settings_6 = log_settings_6.copy()
        expected_log_settings_6["log_file"] = ""

        triton_client = httpclient.InferenceServerClient("localhost:8000")
        with self.assertRaisesRegex(
            InferenceServerException, expected_log_settings_1["error"]
        ) as e:
            triton_client.update_log_settings(settings=log_settings_1)
        self.assertEqual(
            expected_log_settings_2,
            triton_client.update_log_settings(settings=log_settings_2),
            "Unexpected updated log settings",
        )
        self.assertEqual(
            expected_log_settings_3,
            triton_client.update_log_settings(settings=log_settings_3),
            "Unexpected updated log settings",
        )
        self.assertEqual(
            expected_log_settings_4,
            triton_client.update_log_settings(settings=log_settings_4),
            "Unexpected updated log settings",
        )
        self.assertEqual(
            expected_log_settings_5,
            triton_client.update_log_settings(settings=log_settings_5),
            "Unexpected updated log settings",
        )
        self.assertEqual(
            expected_log_settings_6,
            triton_client.update_log_settings(settings=log_settings_6),
            "Unexpected updated log settings",
        )

    def test_grpc_update_settings(self):
        # Update each possible log configuration
        # field and check that they are reflected
        # by the server
        self.check_server_initial_state()
        triton_client = grpcclient.InferenceServerClient("localhost:8001")

        log_settings_1 = {
            "log_file": "log_file.log",
            "log_info": True,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_1 = (
            "log file location can not be updated through network protocol"
        )

        with self.assertRaisesRegex(
            InferenceServerException, expected_log_settings_1
        ) as e:
            triton_client.update_log_settings(settings=log_settings_1)

        log_settings_2 = {
            "log_info": False,
            "log_warning": True,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_2 = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": False},
                        "log_warning": {"boolParam": True},
                        "log_error": {"boolParam": True},
                        "log_verbose_level": {"uint32Param": 0},
                        "log_format": {"stringParam": "default"},
                    }
                }
            ),
            expected_log_settings_2,
        )

        self.assertEqual(
            expected_log_settings_2,
            triton_client.update_log_settings(settings=log_settings_2),
            "Unexpected updated log settings",
        )

        log_settings_3 = {
            "log_info": False,
            "log_warning": False,
            "log_error": True,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_3 = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": False},
                        "log_warning": {"boolParam": False},
                        "log_error": {"boolParam": True},
                        "log_verbose_level": {"uint32Param": 0},
                        "log_format": {"stringParam": "default"},
                    }
                }
            ),
            expected_log_settings_3,
        )

        self.assertEqual(
            expected_log_settings_3,
            triton_client.update_log_settings(settings=log_settings_3),
            "Unexpected updated log settings",
        )

        log_settings_4 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 0,
            "log_format": "default",
        }
        expected_log_settings_4 = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": False},
                        "log_warning": {"boolParam": False},
                        "log_error": {"boolParam": False},
                        "log_verbose_level": {"uint32Param": 0},
                        "log_format": {"stringParam": "default"},
                    }
                }
            ),
            expected_log_settings_4,
        )

        self.assertEqual(
            expected_log_settings_4,
            triton_client.update_log_settings(settings=log_settings_4),
            "Unexpected updated log settings",
        )

        log_settings_5 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 1,
            "log_format": "default",
        }
        expected_log_settings_5 = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": False},
                        "log_warning": {"boolParam": False},
                        "log_error": {"boolParam": False},
                        "log_verbose_level": {"uint32Param": 1},
                        "log_format": {"stringParam": "default"},
                    }
                }
            ),
            expected_log_settings_5,
        )

        self.assertEqual(
            expected_log_settings_5,
            triton_client.update_log_settings(settings=log_settings_5),
            "Unexpected updated log settings",
        )

        log_settings_6 = {
            "log_info": False,
            "log_warning": False,
            "log_error": False,
            "log_verbose_level": 1,
            "log_format": "ISO8601",
        }
        expected_log_settings_6 = grpcclient.service_pb2.LogSettingsResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "log_file": {"stringParam": ""},
                        "log_info": {"boolParam": False},
                        "log_warning": {"boolParam": False},
                        "log_error": {"boolParam": False},
                        "log_verbose_level": {"uint32Param": 1},
                        "log_format": {"stringParam": "ISO8601"},
                    }
                }
            ),
            expected_log_settings_6,
        )

        self.assertEqual(
            expected_log_settings_6,
            triton_client.update_log_settings(settings=log_settings_6),
            "Unexpected updated log settings",
        )


if __name__ == "__main__":
    unittest.main()
