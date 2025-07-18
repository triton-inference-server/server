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

import asyncio
import json

import numpy as np

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.

    This model demonstrates how to use BLS with decoupled models.

    This model has a single input and a single output. The model does not
    support batching.
      - Input 'IN' shape must be equal to [1], datatype must be INT32.
      - For each response, output 'SUM' shape must be equal to [1], datatype
        must be INT32.

    For every request, the model will send a single response that contains an
    output named 'SUM'. We will send two BLS requests to the square model and
    the 'SUM' will contain the summation of the 'OUT' response output returned
    by the square model in the two BLS requests. The input 'IN' determines how
    many responses the square model will generate.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        # This model does not support batching, so 'request_count' should
        # always be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + len(requests)
            )

        response_num = pb_utils.get_input_tensor_by_name(requests[0], "IN")

        # List of awaitables containing inflight inference responses.
        inference_response_awaits = []

        # For detailed explanation about the inputs of the repeat model, refer
        # to the example below:
        # https://github.com/triton-inference-server/python_backend/blob/r22.12/examples/decoupled/square_model.py
        # Construct two BLS requests
        for _ in range(2):
            infer_request = pb_utils.InferenceRequest(
                model_name="square_int32",
                inputs=[response_num],
                requested_output_names=["OUT"],
            )
            # Store the awaitable inside the array. We don't need
            # the inference response immediately so we do not `await`
            # here.
            inference_response_awaits.append(infer_request.async_exec(decoupled=True))

        # Wait for all the inference requests to finish. The execution
        # of the Python script will be blocked until all the awaitables
        # are resolved.
        async_responses = await asyncio.gather(*inference_response_awaits)

        # The variable that will store the sum of the responses.
        response_sum = np.array([0])

        # Iterate over the list of generators of responses returned by the BLS
        # request. This interface can support zero, one, and many inference
        # responses per request.
        for infer_responses in async_responses:
            for infer_response in infer_responses:
                # If inference response has an error, raise an exception
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_response.error().message()
                    )

                # Check for the last empty response.
                if len(infer_response.output_tensors()) > 0:
                    response_sum += pb_utils.get_output_tensor_by_name(
                        infer_response, "OUT"
                    ).as_numpy()

        response = [
            pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("SUM", response_sum)]
            )
        ]

        # Since the model is using the default mode in this example, we
        # will be returning a single response.
        return response

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
