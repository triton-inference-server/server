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

import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import asyncio


def verify_add_sub_results(input0, input1, infer_response):
    if infer_response.has_error():
        return False

    output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
    output1 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT1')

    if (output0 is None) or (output1 is None):
        return False

    if not input0.is_cpu():
        input0 = from_dlpack(
            input0.to_dlpack()).to('cpu').cpu().detach().numpy()
    else:
        input0 = input0.as_numpy()

    if not input1.is_cpu():
        input1 = from_dlpack(
            input1.to_dlpack()).to('cpu').cpu().detach().numpy()
    else:
        input1 = input1.as_numpy()

    if not output0.is_cpu():
        output0 = from_dlpack(
            output0.to_dlpack()).to('cpu').cpu().detach().numpy()
    else:
        output0 = output0.as_numpy()

    if not output1.is_cpu():
        output1 = from_dlpack(
            output1.to_dlpack()).to('cpu').cpu().detach().numpy()
    else:
        output1 = output1.as_numpy()

    expected_output_0 = input0 + input1
    expected_output_1 = input0 - input1

    if not np.all(expected_output_0 == output0):
        print(f'For OUTPUT0 expected {expected_output_0} found {output0}')
        return False

    if not np.all(expected_output_1 == output1):
        print(f'For OUTPUT1 expected {expected_output_1} found {output1}')
        return False

    return True


def create_addsub_inference_request(gpu=False):
    if not gpu:
        input0_np = np.random.randn(16)
        input1_np = np.random.randn(16)
        input0_np = input0_np.astype(np.float32)
        input1_np = input1_np.astype(np.float32)
        input0 = pb_utils.Tensor('INPUT0', input0_np)
        input1 = pb_utils.Tensor('INPUT1', input1_np)
    else:
        input0_pytorch = torch.rand(16).to('cuda')
        input1_pytorch = torch.rand(16).to('cuda')
        input0 = pb_utils.Tensor.from_dlpack('INPUT0',
                                             to_dlpack(input0_pytorch))
        input1 = pb_utils.Tensor.from_dlpack('INPUT1',
                                             to_dlpack(input1_pytorch))

    infer_request = pb_utils.InferenceRequest(
        model_name='dlpack_add_sub',
        inputs=[input0, input1],
        requested_output_names=['OUTPUT0', 'OUTPUT1'])
    return input0, input1, infer_request


async def async_bls_add_sub():
    input0, input1, infer_request = create_addsub_inference_request()
    infer_response = await infer_request.async_exec()
    result_correct = verify_add_sub_results(input0, input1, infer_response)
    if not result_correct:
        return False

    infer_response_sync = infer_request.exec()
    result_correct = verify_add_sub_results(input0, input1, infer_response_sync)
    if not result_correct:
        return False

    return True


async def multiple_async_bls(gpu):
    infer_request_aws = []
    inputs = []
    for _ in range(10):
        input0, input1, infer_request = create_addsub_inference_request(gpu)
        inputs.append((input0, input1))
        infer_request_aws.append(infer_request.async_exec())

    infer_responses = await asyncio.gather(*infer_request_aws)
    for infer_response, input_pair in zip(infer_responses, inputs):
        if infer_response.has_error():
            print('Async BLS failed:',
                  infer_response.error().message(),
                  flush=True)
            return False

        result_correct = verify_add_sub_results(input_pair[0], input_pair[1],
                                                infer_response)
        if not result_correct:
            return False

    return True


class TritonPythonModel:

    async def execute(self, requests):
        responses = []
        for _ in requests:
            test1 = await multiple_async_bls(gpu=True)
            test2 = await multiple_async_bls(gpu=False)
            test3 = await async_bls_add_sub()

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor('OUTPUT0', np.array([test1 & test2 &
                                                         test3]))
                ]))

        return responses
