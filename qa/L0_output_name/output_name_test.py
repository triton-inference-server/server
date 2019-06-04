#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import numpy as np
import os
from builtins import range
from functools import partial
from PIL import Image

import grpc
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config

import unittest
FLAGS = None

class OutputNameValidationTest(unittest.TestCase):
    def TestGRPC(self):
        channel = grpc.insecure_channel(self.url)
        grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

        request_ = self.requestGenerator("DUMMY", FLAGS)
        # Send request
        response_ = grpc_stub.Infer(request_)
        return response_.request_status.code==3

    def requestGenerator(self, output_name, FLAGS):
        # Prepare request for Infer gRPC
        # The meta data part can be reused across requests
        request = grpc_service_pb2.InferRequest()
        request.model_name = self.model_name
        request.model_version = -1

        request.meta_data.batch_size = 1
        output_message = api_pb2.InferRequestHeader.Output()
        output_message.name = output_name
        request.meta_data.output.extend([output_message])

        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input_bytes = input0_data.tobytes()
        request.meta_data.input.add(name="INPUT0", dims=[16])

        del request.raw_input[:]
        request.raw_input.extend([input_bytes])
        return request

    def test_grpc(self):
        self.model_name = 'savedmodel_zero_1_float32'
        self.url = "localhost:8001"
        self.assertTrue(self.TestGRPC())

if __name__ == '__main__':
    unittest.main()
