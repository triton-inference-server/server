# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

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

    This model has one input and one output. The model can support batching,
    with constraint that each request must be batch-1 request, but the shapes
    described here refer to the non-batch portion of the shape.

      - Input 'IN' must have shape [1] and datatype INT32.
      - Output 'OUT' must have shape [1] and datatype INT32.

    For a request, the backend will sent 'n' responses where 'n' is the
    element in IN. For each response, OUT will equal the element of IN.
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

        # Get IN configuration
        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        # Validate the shape and data type of IN
        in_shape = in_config["dims"]
        if (len(in_shape) != 1) or (in_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'IN' to be
                [1], got {}""".format(
                    args["model_name"], in_shape
                )
            )
        if in_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'IN' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], in_config["data_type"]
                )
            )

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Validate the shape and data type of OUT
        out_shape = out_config["dims"]
        if (len(out_shape) != 1) or (out_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'OUT' to be
                [1], got {}""".format(
                    args["model_name"], out_shape
                )
            )
        if out_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'OUT' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], out_config["data_type"]
                )
            )

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

        # Visit individual request to start processing them. Note that execute
        # function is not required to wait for all the requests of the current
        # batch to be processed before returning.
        for request in requests:
            self.process_request(request)

        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another batch of
        # requests. As we are not waiting for the response thread to complete
        # here, it is possible that at any give time the model may be processing
        # multiple batches of requests. Depending upon the request workload,
        # this may lead to a lot of requests being processed by a single model
        # instance at a time. In real-world models, the developer should be
        # mindful of when to return from execute and be willing to accept next
        # request batch.
        return None

    def process_request(self, request):
        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
            ),
        )

        # A model using decoupled transaction policy is not required to send all
        # responses for the current request before returning from the execute.
        # To demonstrate the flexibility of the decoupled API, we are running
        # response thread entirely independent of the execute thread.
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.

        for idx in range(in_input[0]):
            out_output = pb_utils.Tensor("OUT", np.array([in_input[0]], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
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
