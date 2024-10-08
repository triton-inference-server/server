# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pathlib

import numpy as np
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import KServeHttp


def main():
    # Constructing path to Model Repository
    model_path = f"{pathlib.Path(__file__).parent.resolve()}/example_model_repository"
    # Selecting Server Options
    server_options = tritonserver.Options(
        server_id="ExampleServer",
        model_repository=model_path,
        log_error=True,
        log_info=True,
        log_warn=True,
    )

    # Creating server instance
    server = tritonserver.Server(server_options).start(wait_until_ready=True)

    # Selecting Options for KServeHttp Frontend
    http_options = KServeHttp.Options(port=8005)

    # or http_service = KServeHttp.Server(server, http_options) & http_service.stop()
    with KServeHttp(server, http_options) as http_service:
        # The identity model returns an exact duplicate of the input data as output
        model_name = "identity"
        url = "localhost:8005"

        # Create a Triton client
        client = httpclient.InferenceServerClient(url=url)

        # Prepare input data
        input_data = np.array([["Roger Roger"]], dtype=object)

        # Create input and output objects
        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]

        # Set the data for the input tensor
        inputs[0].set_data_from_numpy(input_data)

        results = client.infer(model_name, inputs=inputs)

        # Get the output data
        output_data = results.as_numpy("OUTPUT0")

        print("--------------------- INFERENCE RESULTS ---------------------")
        print("Output data:", output_data)
        print("-------------------------------------------------------------")

    server.stop()


if __name__ == "__main__":
    main()
