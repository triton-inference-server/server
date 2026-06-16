<<<<<<<< HEAD:docs/examples/model_repository/simple_identity/1/model.py
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
========
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
>>>>>>>> r25.03_shantanu:.github/workflows/pre-commit.yml
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

<<<<<<<< HEAD:docs/examples/model_repository/simple_identity/1/model.py
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model always returns the input that it has received."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        """This function is called on inference request."""

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", in_0.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
========
name: pre-commit

on:
  pull_request:

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v5.0.0
      with:
        fetch-depth: 2
    - name: Get modified files
      id: modified-files
      run: echo "modified_files=$(git diff --name-only -r HEAD^1 HEAD | xargs)" >> $GITHUB_OUTPUT
    - uses: actions/setup-python@v6.0.0
    - uses: pre-commit/action@v3.0.1
      with:
        extra_args: --files ${{ steps.modified-files.outputs.modified_files }}
>>>>>>>> r25.03_shantanu:.github/workflows/pre-commit.yml
