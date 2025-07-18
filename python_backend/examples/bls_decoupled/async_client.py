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

import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "bls_decoupled_async"
shape = [1]

with httpclient.InferenceServerClient("localhost:8000") as client:
    in_values = [4, 2, 0, 1]

    for in_value in in_values:
        input_data = np.array([in_value], dtype=np.int32)
        inputs = [
            httpclient.InferInput(
                "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [httpclient.InferRequestedOutput("SUM")]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        # output_data contains two times of the square value of the input value.
        output_data = response.as_numpy("SUM")
        print("==========model result==========")
        print(
            "Two times the square value of {} is {}\n".format(input_data, output_data)
        )

        if not np.allclose((2 * input_data * input_data), output_data):
            print(
                "BLS Decoupled Async example error: incorrect output value. Expected {}, got {}.".format(
                    (2 * input_data * input_data), output_data
                )
            )
            sys.exit(1)

    print("PASS: BLS Decoupled Async")
    sys.exit(0)
