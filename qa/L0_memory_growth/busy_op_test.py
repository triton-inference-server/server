#!/usr/bin/python

# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from builtins import range

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of model.")
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        required=True,
        help="Number of asynchronous requests to launch.",
    )

    FLAGS = parser.parse_args()

    # Run the busyop model which takes a delay as input.
    model_name = FLAGS.model

    # Create the inference context for the model. Need to set the concurrency
    # based on the number of requests so that the delivery of the async
    # requests is not blocked.
    # See the comment for more details: https://github.com/triton-inference-server/client/blob/r24.02/src/python/library/tritonclient/http/_client.py#L1501
    client = httpclient.InferenceServerClient(
        FLAGS.url, verbose=FLAGS.verbose, concurrency=FLAGS.num_requests
    )

    # Collect async requests here
    requests = []

    # Create the data for the input tensor. Creating tensor size with 5 MB.
    tensor_size = [1, 5 * 1024 * 1024]
    input_data = np.random.randn(*tensor_size).astype(np.float32)

    inputs = [
        httpclient.InferInput(
            "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    # Send requests
    for i in range(FLAGS.num_requests):
        requests.append(client.async_infer(model_name, inputs))
        print("Sent request %d" % i, flush=True)
    # wait for requests to finish
    for i in range(len(requests)):
        requests[i].get_result()
        print("Received result %d" % i, flush=True)
