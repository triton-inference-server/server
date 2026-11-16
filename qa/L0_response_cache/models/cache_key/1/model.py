#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        self.batched = config.get("max_batch_size", 0) > 0
        self.reduction = (
            config.get("parameters", {}).get("reduction", {}).get("string_value", "sum")
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            data = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            parameters = json.loads(request.parameters())
            maximum = data.max(axis=-1, keepdims=self.batched)
            total = data.sum(axis=-1, dtype=np.int64, keepdims=self.batched)
            if self.reduction == "max":
                total = maximum.astype(np.int64)
            total = total * int(parameters.get("scale", 1))
            metadata = np.full(
                (data.shape[0], 1) if self.batched else (1,),
                json.dumps(parameters, sort_keys=True),
                dtype=object,
            )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("SUM", total),
                        pb_utils.Tensor("MAX", maximum),
                        pb_utils.Tensor("PARAMETERS", metadata),
                    ]
                )
            )
        return responses
