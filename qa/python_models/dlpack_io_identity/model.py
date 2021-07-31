# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
import numpy as np
import json


class TritonPythonModel:
    """
    This Python identity model passes the DLPack tensors as is. "OUTPUT_IS_GPU"
    input controls whether the model should put the output in GPU or in CPU.
    """
    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            print('ISCPU', input0.is_cpu())
            gpu_output = pb_utils.get_input_tensor_by_name(
                request, "GPU_OUTPUT").as_numpy()

            if input0.is_cpu():
                if not gpu_output[0]:
                    output0 = pb_utils.Tensor.from_dlpack(
                        "OUTPUT0", input0.to_dlpack())
                else:
                    outptu0_pytorch = from_dlpack(input0.to_dlpack()).cuda()
                    output0 = pb_utils.Tensor.from_dlpack(
                        "OUTPUT0", to_dlpack(outptu0_pytorch))
            else:
                if gpu_output[0]:
                    output0 = pb_utils.Tensor.from_dlpack(
                        "OUTPUT0", input0.to_dlpack())
                else:
                    outptu0_pytorch = from_dlpack(input0.to_dlpack()).cpu()
                    output0 = pb_utils.Tensor.from_dlpack(
                        "OUTPUT0", to_dlpack(outptu0_pytorch))

            next_gpu_output = pb_utils.Tensor("NEXT_GPU_OUTPUT",
                                              gpu_output[1:])
            responses.append(
                pb_utils.InferenceResponse([output0, next_gpu_output]))

        return responses
