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

import asyncio

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    async def execute(self, requests):
        processed_requests = []
        async_tasks = []
        for request in requests:
            wait_secs_tensors = pb_utils.get_input_tensor_by_name(
                request, "WAIT_SECONDS"
            ).as_numpy()
            for wait_secs_tensor in wait_secs_tensors:
                wait_secs = wait_secs_tensor[0]
                if wait_secs < 0:
                    self.raise_value_error(requests)
                async_tasks.append(asyncio.create_task(asyncio.sleep(wait_secs)))
            processed_requests.append(
                {
                    "response_sender": request.get_response_sender(),
                    "batch_size": wait_secs_tensors.shape[0],
                }
            )

        # This decoupled execute should be scheduled to run in the background
        # concurrently with other instances of decoupled execute, as long as the event
        # loop is not blocked.
        await asyncio.gather(*async_tasks)

        for p_req in processed_requests:
            response_sender = p_req["response_sender"]
            batch_size = p_req["batch_size"]

            output_tensors = pb_utils.Tensor(
                "DUMMY_OUT", np.array([0 for i in range(batch_size)], np.float32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensors])
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        return None

    def raise_value_error(self, requests):
        # TODO: Model may raise exception without sending complete final
        for request in requests:
            response_sender = request.get_response_sender()
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        raise ValueError("wait_secs cannot be negative")
