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
import pytest
import tritonclient
import tritonclient.http as httpclient
import numpy

sys.path.append("../common")

import test_util as tu

# Similar set up as dynamic batcher tests
class LogInjectionTest(tu.TestResultCollector):

    def test_injection(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit(1)

        input_name = "'nothing_wrong'\nI0205 18:34:18.707423 1 [file.cc:123] THIS ENTRY WAS INJECTED\nI0205 18:34:18.707461 1 [http_server.cc:3570] [request id: <id_unknown>] Infer failed: [request id: <id_unknown>] input 'nothing_wrong"
        
        input_data = numpy.random.randn(1, 3).astype(numpy.float32)
        input_tensor = httpclient.InferInput(input_name, input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        result = triton_client.infer(model_name="simple", inputs=[input_tensor])
