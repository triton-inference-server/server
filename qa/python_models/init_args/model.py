# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def check_init_args(args):
    expected_args = {
        'model_name':
            'init_args',
        'model_instance_name':
            'init_args_0',
        'model_instance_kind':
            'CPU',
        'model_instance_device_id':
            '0',
        'model_repository':
            '/opt/tritonserver/qa/L0_backend_python/models/init_args',
        'model_version':
            '1'
    }
    for arg in expected_args:
        if args[arg] != expected_args[arg]:
            raise pb_utils.TritonModelException(
                arg + ' does not contain correct value.')


class TritonPythonModel:

    def initialize(self, args):
        self.args = args
        check_init_args(self.args)

    def execute(self, requests):
        """
        This function counts the number of keys in the
        "initialize" args argument to make sure that they are
        correct.
        """
        keys = [
            'model_config', 'model_instance_kind', 'model_instance_name',
            'model_instance_device_id', 'model_repository', 'model_version',
            'model_name'
        ]

        correct_keys = 0
        for key in keys:
            if key in list(self.args):
                correct_keys += 1

        responses = []
        for _ in requests:
            out_args = pb_utils.Tensor(
                "OUT", np.array([correct_keys], dtype=np.float32))
            responses.append(pb_utils.InferenceResponse([out_args]))
        return responses
