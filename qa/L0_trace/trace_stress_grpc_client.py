#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import random
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    # 1 ms cancellation timeout
    client_timeout = 1
    url = "localhost:8001"

    try:
        triton_client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = "identity_fp32"

    # Infer
    inputs = []

    input_data = np.array(
        [random.random() for i in range(50)], dtype=np.float32
    ).reshape(1, -1)
    model_input = grpcclient.InferInput(
        name="INPUT0", datatype="FP32", shape=input_data.shape
    )
    model_input.set_data_from_numpy(input_data)
    inputs.append(model_input)

    # Define the callback function. Note the last two parameters should be
    # result and error. InferenceServerClient would povide the results of an
    # inference as grpcclient.InferResult in result. For successful
    # inference, error will be None, otherwise it will be an object of
    # tritonclientutils.InferenceServerException holding the error details
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    # list to hold the results of inference.
    user_data = []

    # Inference call
    for _ in range(1000):
        triton_client.async_infer(
            model_name=model_name,
            inputs=inputs,
            callback=partial(callback, user_data),
            client_timeout=client_timeout,
        )

    # Wait until the results are available in user_data
    time_out = 20
    while (len(user_data) == 0) and time_out > 0:
        time_out = time_out - 1
        time.sleep(1)

    print("results: ", len(user_data))
