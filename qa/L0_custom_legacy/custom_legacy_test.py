#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from functools import partial
import numpy as np
import queue
import unittest
import test_util as tu

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


#
# The simple inference tests on  leagacy custom backend.
#
class CustomLegacyTest(tu.TestResultCollector):

    def setUp(self):
        self.model_name_ = "custom_identity_int32"
        self.input0_data_ = np.array([[10]], dtype=np.int32)

    def _prepare_request(self, protocol):
        if (protocol == "grpc"):
            self.inputs_ = []
            self.inputs_.append(grpcclient.InferInput('INPUT0', [1, 1],
                                                      "INT32"))
            self.outputs_ = []
            self.outputs_.append(grpcclient.InferRequestedOutput('OUTPUT0'))
        else:
            self.inputs_ = []
            self.inputs_.append(httpclient.InferInput('INPUT0', [1, 1],
                                                      "INT32"))
            self.outputs_ = []
            self.outputs_.append(httpclient.InferRequestedOutput('OUTPUT0'))

        self.inputs_[0].set_data_from_numpy(self.input0_data_)

    def _test_no_outputs_helper(self,
                                use_grpc=True,
                                use_http=True,
                                use_streaming=True):

        if use_grpc:
            triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True)
            self._prepare_request("grpc")
            result = triton_client.infer(model_name=self.model_name_,
                                         inputs=self.inputs_,
                                         outputs=self.outputs_,
                                         client_timeout=3)
            # The response should not contain any outputs
            self.assertEqual(result.as_numpy('OUTPUT0'), None)

        if use_http:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True, network_timeout=2.0)
            self._prepare_request("http")
            result = triton_client.infer(model_name=self.model_name_,
                                         inputs=self.inputs_,
                                         outputs=self.outputs_)
            # The response should not contain any outputs
            self.assertEqual(result.as_numpy('OUTPUT0'), None)

        if use_streaming:
            triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True)
            self._prepare_request("grpc")
            user_data = UserData()

            triton_client.stop_stream()
            triton_client.start_stream(callback=partial(callback, user_data),
                                       stream_timeout=1)
            triton_client.async_stream_infer(model_name=self.model_name_,
                                             inputs=self.inputs_,
                                             outputs=self.outputs_)
            result = user_data._completed_requests.get()
            if type(result) == InferenceServerException:
                raise result

            # The response should not contain any outputs
            self.assertEqual(result.as_numpy('OUTPUT0'), None)

    # The tests needs the identity backend to be configured with "suppress_outputs"
    # with TRUE.
    def test_no_outputs(self):
        self._test_no_outputs_helper()


if __name__ == '__main__':
    unittest.main()
