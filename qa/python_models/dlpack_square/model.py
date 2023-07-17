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

import json
import threading

import numpy as np
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

numpy_to_pytorch_dtype = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

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

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        for request in requests:
            self.process_request(request)

        return None

    def process_request(self, request):
        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN"),
                self.output_dtype,
            ),
        )

        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input, output_dtype):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.

        for idx in range(in_input.as_numpy()[0]):
            if in_input.is_cpu():
                if (
                    in_input.as_numpy().dtype.type is np.bytes_
                    or in_input.as_numpy().dtype == np.object_
                ):
                    out_0 = in_input.as_numpy().astype(np.int32)
                    out_tensor = pb_utils.Tensor("OUT", out_0.astype(output_dtype))
                else:
                    in_0_pytorch = from_dlpack(in_input.to_dlpack())
                    out_0 = in_0_pytorch
                    if output_dtype == np.object_:
                        out_tensor = pb_utils.Tensor(
                            "OUT", out_0.numpy().astype(output_dtype)
                        )
                    else:
                        out_0 = out_0.type(numpy_to_pytorch_dtype[output_dtype])
                        out_tensor = pb_utils.Tensor.from_dlpack(
                            "OUT", to_dlpack(out_0)
                        )
            else:
                in_0_pytorch = from_dlpack(in_input.to_dlpack()).cuda()
                out_0 = in_0_pytorch
                out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT0", to_dlpack(out_0))

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
