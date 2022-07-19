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
import json


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")

        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        self.accumulator = np.zeros(1)
        self.max_batch_size = model_config["max_batch_size"]

    def execute(self, requests):
        """
        This function is called on inference request.
        It is derived from "create_tf_modelfile" in 
        common/gen_qa_sequence_models.py and mantains
        a true accumulator when the max batch size is 0

        """
        output_dtype = self.output_dtype

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT").as_numpy().astype(np.int32)
            start_tensor = pb_utils.get_input_tensor_by_name(
                request, "START").as_numpy().astype(np.int32)
            ready_tensor = pb_utils.get_input_tensor_by_name(
                request, "READY").as_numpy().astype(np.int32)

            if self.max_batch_size == 0:
                tmp = np.where(np.equal(start_tensor, 1), input_tensor,
                               np.add(self.accumulator, input_tensor))
                newacc = np.where(np.equal(ready_tensor, 1), tmp,
                                  self.accumulator)
                self.accumulator = newacc
                out_tensor = pb_utils.Tensor(
                    "OUTPUT", self.accumulator.astype(output_dtype))
            else:
                tmp = np.where(
                    np.equal(ready_tensor, 1), np.add(start_tensor,
                                                      input_tensor),
                    np.zeros(np.shape(input_tensor), dtype=output_dtype))
                out_tensor = pb_utils.Tensor("OUTPUT", tmp.astype(output_dtype))

            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
