# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import threading
import time

import numpy

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.

    This model demonstrates how to write a decoupled model where each
    request can generate 0 to many responses.

    This model has three inputs and two outputs. The model does not support
    batching.

      - Input 'IN' can have any vector shape (e.g. [4] or [12]), datatype must
      be INT32.
      - Input 'DELAY' must have the same shape as IN, datatype must be UINT32.
      - Input 'WAIT' must have shape [1] and datatype UINT32.
      - For each response, output 'OUT' must have shape [1] and datatype INT32.
      - For each response, output 'IDX' must have shape [1] and datatype UINT32.

    For a request, the model will send 'n' responses where 'n' is the number of
    elements in IN.  For the i'th response, OUT will equal the i'th element of
    IN and IDX will equal the zero-based count of this response for the request.
    For example, the first response for a request will have IDX = 0 and OUT =
    IN[0], the second will have IDX = 1 and OUT = IN[1], etc. The model will
    wait the i'th DELAY, in milliseconds, before sending the i'th response. If
    IN shape is [0] then no responses will be sent.

    After WAIT milliseconds the model will return from the execute function so
    that Triton can call execute again with another request. WAIT can be less
    than the sum of DELAY so that execute returns before all responses are sent.
    Thus, even if there is only one instance of the model, multiple requests can
    be processed at the same time, and the responses for multiple requests can
    be intermixed, depending on the values of DELAY and WAIT.
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
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Get IDX configuration
        idx_config = pb_utils.get_output_config_by_name(model_config, "IDX")

        # Convert Triton types to numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config["data_type"])
        self.idx_dtype = pb_utils.triton_string_to_numpy(idx_config["data_type"])

        # Optional parameter to specify the number of elements in the OUT tensor in each response.
        # Defaults to 1 if not provided. Example: If input 'IN' is [4] and 'output_num_elements' is set to 3,
        # then 'OUT' will be [4, 4, 4]. If 'output_num_elements' is not specified, 'OUT' will default to [4].
        parameters = self.model_config.get("parameters", {})
        self.output_num_elements = int(
            parameters.get("output_num_elements", {}).get("string_value", 1)
        )

        # To keep track of response threads so that we can delay
        # the finalizing the model until all response threads
        # have completed.
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. The request.get_response_sender() must be used to
        get an InferenceResponseSender object associated with the request.
        Use the InferenceResponseSender.send(response=<infer response object>,
        flags=<flags>) to send responses.

        In the final response sent using the response sender object, you must
        set the flags argument to TRITONSERVER_RESPONSE_COMPLETE_FINAL to
        indicate no responses will be sent for the corresponding request. If
        there is an error, you can set the error argument when creating a
        pb_utils.InferenceResponse. Setting the flags argument is optional and
        defaults to zero. When the flags argument is set to
        TRITONSERVER_RESPONSE_COMPLETE_FINAL providing the response argument is
        optional.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        None
        """

        # This model does not support batching, so 'request_count' should always
        # be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + len(requests)
            )

        in_input = pb_utils.get_input_tensor_by_name(requests[0], "IN").as_numpy()
        delay_input = pb_utils.get_input_tensor_by_name(requests[0], "DELAY").as_numpy()
        if in_input.shape != delay_input.shape:
            raise pb_utils.TritonModelException(
                f"expected IN and DELAY shape to match, got {list(in_input.shape)} and {list(delay_input.shape)}."
            )

        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(
            target=self.response_thread,
            args=(requests[0].get_response_sender(), in_input, delay_input),
        )

        # A model using decoupled transaction policy is not required to send all
        # responses for the current request before returning from the execute.
        # To demonstrate the flexibility of the decoupled API, we are running
        # response thread entirely independent of the execute thread.
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

        # Read WAIT input for wait time, then return so that Triton can call
        # execute again with another request.
        wait_input = pb_utils.get_input_tensor_by_name(requests[0], "WAIT").as_numpy()
        time.sleep(wait_input[0] / 1000)

        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another request. As
        # we are not waiting for the response thread to complete here, it is
        # possible that at any give time the model may be processing multiple
        # requests. Depending upon the request workload, this may lead to a lot
        # of requests being processed by a single model instance at a time. In
        # real-world models, the developer should be mindful of when to return
        # from execute and be willing to accept next request.
        return None

    def response_thread(self, response_sender, in_input, delay_input):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.  Iterate over input/delay pairs. Wait for DELAY
        # milliseconds and then create and send a response.

        idx_dtype = self.idx_dtype
        out_dtype = self.out_dtype

        for idx in range(in_input.size):
            in_value = in_input[idx]
            delay_value = delay_input[idx]

            time.sleep(delay_value / 1000)

            idx_output = pb_utils.Tensor("IDX", numpy.array([idx], idx_dtype))
            out_output = pb_utils.Tensor(
                "OUT",
                numpy.full((self.output_num_elements,), in_value, dtype=out_dtype),
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[idx_output, out_output]
            )
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        Here we will wait for all response threads to complete sending
        responses.
        """
        print("Finalize invoked")

        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = logging_time_sec / sleep_time_sec
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
                if cycles % cycle_to_log == 0:
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1

        print("Finalize complete...")
