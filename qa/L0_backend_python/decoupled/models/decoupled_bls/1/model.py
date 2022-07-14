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
import json
import threading
import time
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import sys


class TritonPythonModel:
    """ This model sends an error message with the first request.
    """

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(args['model_name']))

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Convert Triton types to numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(
            out_config['data_type'])

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """ This function is called on inference request.
        """

        # Only generate the error for the first request
        for i, request in enumerate(requests):
            request_input = pb_utils.get_input_tensor_by_name(request, 'IN')

            # Sync BLS request
            infer_request = pb_utils.InferenceRequest(
                model_name='identity_fp32',
                requested_output_names=["OUTPUT0"],
                inputs=[pb_utils.Tensor('INPUT0', request_input.as_numpy())])
            infer_response = infer_request.exec()
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    f"BLS Response has an error: {infer_response.error().message()}"
                )

            output0 = pb_utils.get_output_tensor_by_name(
                infer_response, "OUTPUT0")
            if np.any(output0.as_numpy() != request_input.as_numpy()):
                raise pb_utils.TritonModelException(
                    f"BLS Request input and BLS response output do not match. {request_input.as_numpy()} != {output0.as_numpy()}"
                )

            thread1 = threading.Thread(target=self.response_thread,
                                       args=(request.get_response_sender(),
                                             pb_utils.get_input_tensor_by_name(
                                                 request, 'IN').as_numpy()))
            thread1.daemon = True
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread1.start()

        return None

    def _get_gpu_bls_outputs(self, input0_pb, input1_pb):
        """
        This function is created to test that the DLPack container works
        properly when the inference response and outputs go out of scope.

        Returns True on success and False on failure.
        """
        infer_request = pb_utils.InferenceRequest(
            model_name='dlpack_add_sub',
            inputs=[input0_pb, input1_pb],
            requested_output_names=['OUTPUT0', 'OUTPUT1'])
        infer_response = infer_request.exec()
        if infer_response.has_error():
            return False

        output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
        output1 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT1')
        if output0 is None or output1 is None:
            return False

        # When one of the inputs is in GPU the output returned by the model must
        # be in GPU, otherwise the outputs will be in CPU.
        if not input0_pb.is_cpu() or not input1_pb.is_cpu():
            if output0.is_cpu() or output1.is_cpu():
                return False
        else:
            if (not output0.is_cpu()) or (not output1.is_cpu()):
                return False

        # Make sure that the reference count is increased by one when DLPack
        # representation is created.
        rc_before_dlpack_output0 = sys.getrefcount(output0)
        rc_before_dlpack_output1 = sys.getrefcount(output1)

        output0_dlpack = output0.to_dlpack()
        output1_dlpack = output1.to_dlpack()

        rc_after_dlpack_output0 = sys.getrefcount(output0)
        rc_after_dlpack_output1 = sys.getrefcount(output1)

        if rc_after_dlpack_output0 - rc_before_dlpack_output0 != 1:
            return False

        if rc_after_dlpack_output1 - rc_before_dlpack_output1 != 1:
            return False

        # Make sure that reference count decreases after destroying the DLPack
        output0_dlpack = None
        output1_dlpack = None
        rc_after_del_dlpack_output0 = sys.getrefcount(output0)
        rc_after_del_dlpack_output1 = sys.getrefcount(output1)
        if rc_after_del_dlpack_output0 - rc_after_dlpack_output0 != -1:
            return False

        if rc_after_del_dlpack_output1 - rc_after_dlpack_output1 != -1:
            return False

        return output0.to_dlpack(), output1.to_dlpack()

    def _test_gpu_bls_add_sub(self, is_input0_gpu, is_input1_gpu):
        input0 = torch.rand(16)
        input1 = torch.rand(16)

        if is_input0_gpu:
            input0 = input0.to('cuda')

        if is_input1_gpu:
            input1 = input1.to('cuda')

        input0_pb = pb_utils.Tensor.from_dlpack('INPUT0', to_dlpack(input0))
        input1_pb = pb_utils.Tensor.from_dlpack('INPUT1', to_dlpack(input1))
        gpu_bls_return = self._get_gpu_bls_outputs(input0_pb, input1_pb)
        if gpu_bls_return:
            output0_dlpack, output1_dlpack = gpu_bls_return
        else:
            return False

        expected_output_0 = from_dlpack(
            input0_pb.to_dlpack()).to('cpu') + from_dlpack(
                input1_pb.to_dlpack()).to('cpu')
        expected_output_1 = from_dlpack(
            input0_pb.to_dlpack()).to('cpu') - from_dlpack(
                input1_pb.to_dlpack()).to('cpu')

        output0_matches = torch.all(
            expected_output_0 == from_dlpack(output0_dlpack).to('cpu'))
        output1_matches = torch.all(
            expected_output_1 == from_dlpack(output1_dlpack).to('cpu'))
        if not output0_matches or not output1_matches:
            return False

        return True

    def execute_gpu_bls(self):
        for input0_device in [True, False]:
            for input1_device in [True, False]:
                test_status = self._test_gpu_bls_add_sub(
                    input0_device, input1_device)
                if not test_status:
                    return False

        return True

    def response_thread(self, response_sender, in_input):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.
        # Sleep 5 seconds to make sure the main thread has exited.
        time.sleep(5)

        status = self.execute_gpu_bls()
        if not status:
            infer_response = pb_utils.InferenceResponse(
                error="GPU BLS test failed.")
            response_sender.send(infer_response)
        else:
            in_value = in_input
            infer_request = pb_utils.InferenceRequest(
                model_name='identity_fp32',
                requested_output_names=["OUTPUT0"],
                inputs=[pb_utils.Tensor('INPUT0', in_input)])
            infer_response = infer_request.exec()
            output0 = pb_utils.get_output_tensor_by_name(
                infer_response, "OUTPUT0")
            if infer_response.has_error():
                response = pb_utils.InferenceResponse(
                    error=infer_response.error().message())
                response_sender.send(
                    response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            elif np.any(in_input != output0.as_numpy()):
                error_message = (
                    "BLS Request input and BLS response output do not match."
                    f" {in_value} != {output0.as_numpy()}")
                response = pb_utils.InferenceResponse(error=error_message)
                response_sender.send(
                    response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            else:
                output_tensors = [pb_utils.Tensor('OUT', in_value)]
                response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors)
                response_sender.send(
                    response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Finalize invoked')

        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = (self.inflight_thread_count != 0)
            if inflight_threads:
                time.sleep(0.1)

        print('Finalize complete...')
