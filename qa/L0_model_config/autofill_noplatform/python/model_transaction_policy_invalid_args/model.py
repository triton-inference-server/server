# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input0 = {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        input1 = {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]}
        output0 = {"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        output1 = {"name": "OUTPUT1", "data_type": "TYPE_FP32", "dims": [4]}
        transaction_policy = {"invalid": "argument"}

        auto_complete_model_config.set_max_batch_size(4)
        auto_complete_model_config.set_model_transaction_policy(transaction_policy)
        auto_complete_model_config.add_input(input0)
        auto_complete_model_config.add_input(input1)
        auto_complete_model_config.add_output(output0)
        auto_complete_model_config.add_output(output1)

        return auto_complete_model_config

    def execute(self, requests):
        pass
