#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Inference check for the RHEL/manylinux tutorial's PyTorch model, served by
# test.sh: add_torch (pytorch backend, KIND_GPU) computes OUTPUT__0 = INPUT__0 +
# INPUT__1.

import os

import numpy as np
import pytest
import tritonclient.http as httpclient

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

INPUT0 = np.array([1, 2, 3, 4], dtype=np.float32)
INPUT1 = np.array([10, 20, 30, 40], dtype=np.float32)


@pytest.fixture(scope="module")
def client():
    with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as c:
        yield c


def test_server_ready(client):
    assert client.is_server_ready()


def test_add_torch(client):
    assert client.is_model_ready("add_torch")
    inputs = [
        httpclient.InferInput("INPUT__0", INPUT0.shape, "FP32"),
        httpclient.InferInput("INPUT__1", INPUT1.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(INPUT0)
    inputs[1].set_data_from_numpy(INPUT1)
    result = client.infer("add_torch", inputs)
    np.testing.assert_array_equal(result.as_numpy("OUTPUT__0"), INPUT0 + INPUT1)
