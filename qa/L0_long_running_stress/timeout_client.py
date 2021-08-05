# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import time
import socket
import test_util as tu

import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class ClientTimeoutTest(tu.TestResultCollector):

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

    def test_grpc_infer(self):
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001",
                                                         verbose=True)
        self._prepare_request("grpc")

        # The model is configured to take three seconds to send the
        # response. Expect an exception for small timeout values.
        result = triton_client.infer(model_name=self.model_name_,
                                     inputs=self.inputs_,
                                     outputs=self.outputs_,
                                     client_timeout=0.2)


if __name__ == '__main__':
    unittest.main()
