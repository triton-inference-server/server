# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import os
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input = {'name': 'INPUT', 'data_type': 'TYPE_FP32', 'dims': [1]}
        output = {'name': 'OUTPUT', 'data_type': 'TYPE_FP32', 'dims': [1]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input)
        auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        import tensorflow
        self.model_config = args['model_config']
        # This is to make sure that /bin/bash is not picking up
        # the wrong shared libraries after installing Tensorflow.
        # Tensorflow uses a shared library which is common with
        # bash.
        os.system('/bin/bash --help')
        print(
            f'Python version is {sys.version_info.major}.{sys.version_info.minor}, NumPy version is {np.version.version}, and Tensorflow version is {tensorflow.__version__}',
            flush=True)

    def execute(self, requests):
        """ This function is called on inference request.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
