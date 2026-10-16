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


"""Requests to the llm_ensemble QA ensemble (see conftest.py)."""

import numpy as np
import tritonclient.grpc as grpcclient

ENSEMBLE_NAME = "llm_ensemble"

# prompt_template/1/model.py prepends this to the question; the LLM API echoes
# its prompt, so the output starting with it shows both steps ran.
PREFIX = "Answer briefly. Question: "
QUESTION = "What is the capital of France?"


def ensemble_inputs(question=QUESTION, max_tokens=16, streaming=None):
    values = [
        ("QUESTION", "BYTES", np.array([question.encode()], dtype=np.object_)),
        ("max_tokens", "INT32", np.array([max_tokens], dtype=np.int32)),
    ]
    if streaming is not None:
        values.append(("streaming", "BOOL", np.array([streaming], dtype=np.bool_)))
    inputs = []
    for name, datatype, data in values:
        tensor = grpcclient.InferInput(name, [1], datatype)
        tensor.set_data_from_numpy(data)
        inputs.append(tensor)
    return inputs
