#!/usr/bin/env python
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

import queue
import time
import unittest

# For stream infer test
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class RestrictedProtocolTest(unittest.TestCase):
    def setUp(self):
        self.client_ = grpcclient.InferenceServerClient(url="localhost:8001")
        self.model_name_ = "simple"
        self.prefix_ = "triton-grpc-protocol-"

    # Other unspecified protocols should not be restricted
    def test_sanity(self):
        self.client_.get_inference_statistics("simple")
        self.client_.get_inference_statistics(
            "simple", headers={self.prefix_ + "infer-key": "infer-value"}
        )

    # health, infer, model repository protocols are restricted.
    # health and infer expects "triton-grpc-restricted-infer-key : infer-value" header,
    # model repository expected "triton-grpc-restricted-admin-key : admin-value".
    def test_model_repository(self):
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            self.client_.unload_model(
                self.model_name_, headers={self.prefix_ + "infer-key": "infer-value"}
            )
        # Request go through and get actual transaction error
        with self.assertRaisesRegex(
            InferenceServerException, "explicit model load / unload is not allowed"
        ):
            self.client_.unload_model(
                self.model_name_, headers={self.prefix_ + "admin-key": "admin-value"}
            )

    def test_health(self):
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            self.client_.is_server_live()
        self.client_.is_server_live({self.prefix_ + "infer-key": "infer-value"})

    def test_infer(self):
        # setup
        inputs = [
            grpcclient.InferInput("INPUT0", [1, 16], "INT32"),
            grpcclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))

        # This test only care if the request goes through
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            _ = self.client_.infer(
                model_name=self.model_name_, inputs=inputs, headers={"test": "1"}
            )
        self.client_.infer(
            model_name=self.model_name_,
            inputs=inputs,
            headers={self.prefix_ + "infer-key": "infer-value"},
        )

    def test_stream_infer(self):
        # setup
        inputs = [
            grpcclient.InferInput("INPUT0", [1, 16], "INT32"),
            grpcclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        user_data = UserData()
        # The server can't interfere with whether GRPC should create the stream,
        # server will be notified after the stream is established and only
        # until then be able to access metadata to decide whether to continue
        # the stream.
        # So on client side, it will always perceive that the stream is
        # successfully created and can only check its health at a later time.
        self.client_.start_stream(partial(callback, user_data), headers={"test": "1"})
        # wait for sufficient round-trip time
        time.sleep(1)
        with self.assertRaisesRegex(
            InferenceServerException, "The stream is no longer in valid state"
        ):
            self.client_.async_stream_infer(model_name=self.model_name_, inputs=inputs)
        # callback should record error detail
        self.assertFalse(user_data._completed_requests.empty())
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            raise user_data._completed_requests.get()

        self.assertTrue(user_data._completed_requests.empty())

        # Stop and start new stream with proper header
        self.client_.stop_stream()
        self.client_.start_stream(
            partial(callback, user_data),
            headers={self.prefix_ + "infer-key": "infer-value"},
        )
        self.client_.async_stream_infer(model_name=self.model_name_, inputs=inputs)
        # wait for response
        time.sleep(1)
        self.assertFalse(user_data._completed_requests.empty())
        self.assertNotEqual(
            type(user_data._completed_requests.get()), InferenceServerException
        )


if __name__ == "__main__":
    unittest.main()
