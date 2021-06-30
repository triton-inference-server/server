# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import json
import triton_python_backend_utils as pb_utils


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

    def execute(self, requests):
        """ This function is called on inference request.
        """

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            if in_0.as_numpy().dtype.type is np.bytes_ or in_0.as_numpy(
            ).dtype == np.object_:
                out_0, out_1 = (in_0.as_numpy().astype(np.int32) + in_1.as_numpy().astype(np.int32),\
                    in_0.as_numpy().astype(np.int32) - in_1.as_numpy().astype(np.int32))
            else:
                out_0, out_1 = (in_0.as_numpy() + in_1.as_numpy(),
                                in_0.as_numpy() - in_1.as_numpy())

            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1",
                                           out_1.astype(output1_dtype))
            responses.append(
                pb_utils.InferenceResponse([out_tensor_0, out_tensor_1]))
        return responses
