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
import torch
import numpy as np
import json


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT1")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        self.numpy_to_pytorch_dtype = {
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

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            # If both of the tensors are in CPU, use NumPy.
            if in_0.is_cpu() and in_1.is_cpu():
                if in_0.as_numpy().dtype.type is np.bytes_ or in_0.as_numpy(
                ).dtype == np.object_:
                    out_0, out_1 = (in_0.as_numpy().astype(np.int32) - in_1.as_numpy().astype(np.int32),\
                        in_0.as_numpy().astype(np.int32) + in_1.as_numpy().astype(np.int32))
                    out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                                   out_0.astype(output0_dtype))
                    out_tensor_1 = pb_utils.Tensor("OUTPUT1",
                                                   out_1.astype(output1_dtype))
                else:
                    in_0_pytorch, in_1_pytorch = from_dlpack(
                        in_0.to_dlpack()), from_dlpack(in_1.to_dlpack())
                    out_0, out_1 = (in_0_pytorch - in_1_pytorch,
                                    in_0_pytorch + in_1_pytorch)

                    if self.output0_dtype == np.object_:
                        out_tensor_0 = pb_utils.Tensor(
                            "OUTPUT0",
                            out_0.numpy().astype(output0_dtype))
                    else:
                        out_0 = out_0.type(
                            self.numpy_to_pytorch_dtype[output0_dtype])
                        out_tensor_0 = pb_utils.Tensor.from_dlpack(
                            "OUTPUT0", to_dlpack(out_0))

                    if self.output1_dtype == np.object_:
                        out_tensor_1 = pb_utils.Tensor(
                            "OUTPUT1",
                            out_1.numpy().astype(output1_dtype))
                    else:
                        out_1 = out_1.type(
                            self.numpy_to_pytorch_dtype[output1_dtype])
                        out_tensor_1 = pb_utils.Tensor.from_dlpack(
                            "OUTPUT1", to_dlpack(out_1))

            else:
                in_0_pytorch, in_1_pytorch = from_dlpack(
                    in_0.to_dlpack()).cuda(), from_dlpack(
                        in_1.to_dlpack()).cuda()
                out_0, out_1 = (in_0_pytorch - in_1_pytorch,
                                in_0_pytorch + in_1_pytorch)
                out_tensor_0 = pb_utils.Tensor.from_dlpack(
                    "OUTPUT0", to_dlpack(out_0))
                out_tensor_1 = pb_utils.Tensor.from_dlpack(
                    "OUTPUT1", to_dlpack(out_1))

            responses.append(
                pb_utils.InferenceResponse([out_tensor_0, out_tensor_1]))

        return responses
