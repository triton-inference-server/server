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

import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

        # Get tensor configurations for testing/validation
        self.input0_config = pb_utils.get_input_config_by_name(
            self.model_config, "INPUT0"
        )
        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0"
        )

    def validate_bf16_tensor(self, tensor, tensor_config):
        # I/O datatypes can be queried from the model config if needed
        dtype = tensor_config["data_type"]
        if dtype != "TYPE_BF16":
            raise Exception(f"Expected a BF16 tensor, but got {dtype} instead.")

        # Converting BF16 tensors to numpy is not supported, and DLPack
        # should be used instead via to_dlpack and from_dlpack.
        try:
            _ = tensor.as_numpy()
        except pb_utils.TritonModelException as e:
            expected_error = "tensor dtype is bf16 and cannot be converted to numpy"
            assert expected_error in str(e).lower()
        else:
            raise Exception("Expected BF16 conversion to numpy to fail")

    def execute(self, requests):
        """
        Identity model in Python backend with example BF16 and PyTorch usage.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")

            # Numpy does not support BF16, so use DLPack instead.
            bf16_dlpack = input_tensor.to_dlpack()

            # OPTIONAL: The tensor can be converted to other dlpack-compatible
            # frameworks like PyTorch and TensorFlow with their dlpack utilities.
            torch_tensor = torch.utils.dlpack.from_dlpack(bf16_dlpack)

            # When complete, convert back to a pb_utils.Tensor via DLPack.
            output_tensor = pb_utils.Tensor.from_dlpack(
                "OUTPUT0", torch.utils.dlpack.to_dlpack(torch_tensor)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))

            # NOTE: The following helper function is for testing and example
            # purposes only, you should remove this in practice.
            self.validate_bf16_tensor(input_tensor, self.input0_config)
            self.validate_bf16_tensor(output_tensor, self.output0_config)

        return responses
