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

import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:

    def execute(self, requests):
        """Model supporting optional inputs. If the input is not provided, an
        input tensor of size 1 containing scalar 5 will be used."""
        responses = []
        for request in requests:
            input0_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input1_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            if input0_tensor is not None:
                input0_numpy = input0_tensor.as_numpy()
            else:
                input0_numpy = np.array([5], dtype=np.int32)

            if input1_tensor is not None:
                input1_numpy = input1_tensor.as_numpy()
            else:
                input1_numpy = np.array([5], dtype=np.int32)

            output0_tensor = pb_utils.Tensor("OUTPUT0",
                                             input0_numpy + input1_numpy)
            output1_tensor = pb_utils.Tensor("OUTPUT1",
                                             input0_numpy - input1_numpy)
            responses.append(
                pb_utils.InferenceResponse([output0_tensor, output1_tensor]))

        return responses
