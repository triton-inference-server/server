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

import json
import queue
import time
from threading import Thread

import numpy
import triton_python_backend_utils as pb_utils


class WorkItem:
    def __init__(self, response_sender, in_input, delay_input):
        self.response_sender = response_sender
        self.in_input = in_input
        self.delay_input = delay_input


class TritonPythonModel:
    """This model launches a separate thread to handle the request from a queue. The thread is launched from
    the `initialize` function and is terminated in the `finalize` function. This is different from the repeat
    example in the Python Backend repository where a thread is launched per request and terminated after the response
    is sent.
    """

    def initialize(self, args):
        self.work_queue = queue.Queue()
        self.running = True

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

        self.sender_thread = Thread(target=self.sender_loop)
        self.sender_thread.daemon = True
        self.sender_thread.start()

    def sender_loop(self):
        while self.running:
            # Grab work from queue
            work_item = self.work_queue.get()
            if work_item.response_sender is None:
                pb_utils.log(
                    pb_utils.LogLevel.INFO,
                    "Sender thread received dummy work item. Exiting...",
                )
                self.work_queue.task_done()
                break

            response_sender = work_item.response_sender
            in_input = work_item.in_input
            delay_input = work_item.delay_input

            idx_dtype = self.idx_dtype
            out_dtype = self.out_dtype

            for idx in range(in_input.size):
                in_value = in_input[idx]
                delay_value = delay_input[idx]

                time.sleep(delay_value / 1000)

                idx_output = pb_utils.Tensor("IDX", numpy.array([idx], idx_dtype))
                out_output = pb_utils.Tensor("OUT", numpy.array([in_value], out_dtype))
                response = pb_utils.InferenceResponse(
                    output_tensors=[idx_output, out_output]
                )
                response_sender.send(response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            self.work_queue.task_done()

    def execute(self, requests):
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

        # Put work item in queue to be processed by the sender thread
        self.work_queue.put(
            WorkItem(requests[0].get_response_sender(), in_input, delay_input)
        )

        wait_input = pb_utils.get_input_tensor_by_name(requests[0], "WAIT").as_numpy()
        time.sleep(wait_input[0] / 1000)

        return None

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        Here we will wait for all response threads to complete sending
        responses.
        """
        pb_utils.log(pb_utils.LogLevel.INFO, "Finalizing model...")

        # Pass dummy work item to the queue to terminate the sender_thread
        self.work_queue.put(
            WorkItem(
                None, numpy.array([0], numpy.int32), numpy.array([0], numpy.uint32)
            )
        )

        self.running = False
        self.sender_thread.join()
