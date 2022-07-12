#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import numpy as np
import os
from builtins import range
from functools import partial
from PIL import Image
import unittest
import test_util as tu

import grpc
from tritongrpcclient import grpc_service_pb2
from tritongrpcclient import grpc_service_pb2_grpc

_trials = ("graphdef", "libtorch", "onnx", "plan", "savedmodel")


class OutputNameValidationTest(tu.TestResultCollector):

    def requestGenerator(self, model_name, output_name):
        request = grpc_service_pb2.ModelInferRequest()
        request.model_name = model_name
        request.id = "output name validation"

        input = grpc_service_pb2.ModelInferRequest().InferInputTensor()
        input.name = "INPUT0"
        input.datatype = "FP32"
        input.shape.extend([1])

        request.inputs.extend([input])

        output = grpc_service_pb2.ModelInferRequest(
        ).InferRequestedOutputTensor()
        output.name = output_name
        request.outputs.extend([output])

        request.raw_input_contents.extend([bytes(4 * 'a', 'utf-8')])

        return request

    def test_grpc(self):
        channel = grpc.insecure_channel("localhost:8001")
        grpc_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)

        # Send request with invalid output name
        for trial in _trials:
            model_name = "{}_nobatch_zero_1_float32".format(trial)
            request = self.requestGenerator(model_name, "DUMMY")
            try:
                response = grpc_stub.ModelInfer(request)
                self.assertTrue(
                    False,
                    "unexpected success for unknown output " + model_name)
            except grpc.RpcError as rpc_error:
                msg = rpc_error.details()
                self.assertTrue(
                    msg.startswith(
                        "unexpected inference output 'DUMMY' for model"))


if __name__ == '__main__':
    unittest.main()
