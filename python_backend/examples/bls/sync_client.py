# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "bls_sync"
shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.rand(*shape).astype(np.float32)
    input1_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
        httpclient.InferInput("MODEL_NAME", [1], np_to_triton_dtype(np.object_)),
    ]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    # Will perform the inference request on the 'add_sub' model.
    inputs[2].set_data_from_numpy(np.array(["add_sub"], dtype=np.object_))

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")
    print("=========='add_sub' model result==========")
    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, output0_data
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT1 ({})".format(
            input0_data, input1_data, output1_data
        )
    )
    if not np.allclose(input0_data + input1_data, output0_data):
        print("BLS sync example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("BLS sync example error: incorrect difference")
        sys.exit(1)

    # Will perform the inference request on the pytorch model:
    inputs[2].set_data_from_numpy(np.array(["pytorch"], dtype=np.object_))
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")
    print("\n")
    print("=========='pytorch' model result==========")
    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, output0_data
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT1 ({})".format(
            input0_data, input1_data, output1_data
        )
    )
    if not np.allclose(input0_data + input1_data, output0_data):
        print("BLS sync example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("BLS sync example error: incorrect difference")
        sys.exit(1)

    # Will perform the same inference request on an undefined model. This leads
    # to an exception:
    print("\n")
    print("=========='undefined' model result==========")
    try:
        inputs[2].set_data_from_numpy(np.array(["undefined_model"], dtype=np.object_))
        _ = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    except InferenceServerException as e:
        print(e.message())

    print("PASS: BLS Sync")
    sys.exit(0)
