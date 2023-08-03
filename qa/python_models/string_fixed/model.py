# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    This model returns a constant string on every inference request.
    """

    def initialize(self, args):
        self._index = 0
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):
        # Create four different responses (empty string or fixed string) * (two
        # datatypes)
        responses = []
        for _ in requests:
            if self._index == 0:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array(["123456"], dtype=self._dtypes[0])
                )
            elif self._index == 1:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array([], dtype=self._dtypes[1])
                )
            elif self._index == 2:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array(["123456"], dtype=self._dtypes[0])
                )
            elif self._index == 3:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array([], dtype=self._dtypes[1])
                )
            self._index += 1
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
