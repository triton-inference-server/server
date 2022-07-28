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
import unittest
import triton_python_backend_utils as pb_utils
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import threading
from multiprocessing import Pool
import sys

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


def bls_add_sub(_=None):
    input0_np = np.random.randn(*[16])
    input0_np = input0_np.astype(np.float32)
    input1_np = np.random.randn(*[16])
    input1_np = input1_np.astype(np.float32)
    input0 = pb_utils.Tensor('INPUT0', input0_np)
    input1 = pb_utils.Tensor('INPUT1', input1_np)
    infer_request = pb_utils.InferenceRequest(
        model_name='add_sub',
        inputs=[input0, input1],
        requested_output_names=['OUTPUT0', 'OUTPUT1'])
    infer_response = infer_request.exec()
    if infer_response.has_error():
        return False

    output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
    output1 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT1')
    if output0 is None or output1 is None:
        return False

    expected_output_0 = input0.as_numpy() + input1.as_numpy()
    expected_output_1 = input0.as_numpy() - input1.as_numpy()

    if not np.all(expected_output_0 == output0.as_numpy()):
        return False

    if not np.all(expected_output_1 == output1.as_numpy()):
        return False

    return True


class PBBLSTest(unittest.TestCase):

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def test_bls_wrong_inputs(self):
        input0 = pb_utils.Tensor('INPUT0', np.random.randn(*[1, 16]))

        infer_request = pb_utils.InferenceRequest(
            model_name='add_sub',
            inputs=[input0],
            requested_output_names=['OUTPUT0', 'OUTPUT1'])
        infer_response = infer_request.exec()
        self.assertTrue(infer_response.has_error())
        self.assertIn("expected 2 inputs but got 1 inputs for model 'add_sub'",
                      infer_response.error().message())
        self.assertTrue(len(infer_response.output_tensors()) == 0)

    def _send_bls_sequence_requests(self, correlation_id):
        # Start request
        try:
            input = pb_utils.Tensor('INPUT', np.array([1000], dtype=np.int32))

            infer_request = pb_utils.InferenceRequest(
                model_name='onnx_nobatch_sequence_int32',
                inputs=[input],
                requested_output_names=['OUTPUT'],
                flags=pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START,
                correlation_id=correlation_id)
            self.assertTrue(infer_request.flags(),
                            pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START)
            infer_response = infer_request.exec()
            self.assertFalse(infer_response.has_error())
            output = pb_utils.get_output_tensor_by_name(infer_response,
                                                        'OUTPUT')
            self.assertFalse(output.is_cpu())
            output = from_dlpack(
                output.to_dlpack()).to('cpu').cpu().detach().numpy()
            self.assertEqual(output[0], input.as_numpy()[0])

            for i in range(10):
                input = pb_utils.Tensor('INPUT', np.array([i], dtype=np.int32))
                infer_request = pb_utils.InferenceRequest(
                    model_name='onnx_nobatch_sequence_int32',
                    inputs=[input],
                    requested_output_names=['OUTPUT'],
                    correlation_id=correlation_id)
                infer_response = infer_request.exec()
                self.assertFalse(infer_response.has_error())

                # The new output is the previous output + the current input
                expected_output = output[0] + i
                output = pb_utils.get_output_tensor_by_name(
                    infer_response, 'OUTPUT')
                self.assertFalse(output.is_cpu())
                output = from_dlpack(
                    output.to_dlpack()).to('cpu').cpu().detach().numpy()
                self.assertEqual(output[0], expected_output)

            # Final request
            input = pb_utils.Tensor('INPUT', np.array([2000], dtype=np.int32))

            infer_request = pb_utils.InferenceRequest(
                model_name='onnx_nobatch_sequence_int32',
                inputs=[input],
                requested_output_names=['OUTPUT'],
                correlation_id=correlation_id)
            infer_request.set_flags(
                pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)
            self.assertTrue(infer_request.flags(),
                            pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)

            infer_response = infer_request.exec()
            self.assertFalse(infer_response.has_error())
            expected_output = output[0] + input.as_numpy()[0]
            output = pb_utils.get_output_tensor_by_name(infer_response,
                                                        'OUTPUT')
            self.assertFalse(output.is_cpu())
            output = from_dlpack(
                output.to_dlpack()).to('cpu').cpu().detach().numpy()
            self.assertEqual(output[0], expected_output)
        except Exception as e:
            self.add_deferred_exception(e)

    def test_bls_sequence(self):
        # Send 2 sequence of BLS requests simultaneously and check the responses.
        threads = []
        thread1 = threading.Thread(target=self._send_bls_sequence_requests,
                                   args=(1000,))
        threads.append(thread1)
        thread2 = threading.Thread(target=self._send_bls_sequence_requests,
                                   args=(1001,))
        threads.append(thread2)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Check if any of the threads had an exception
        self.check_deferred_exception()

    def test_bls_incorrect_args(self):
        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(
                inputs=[], requested_output_names=['OUTPUT0', 'OUTPUT1'])

        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(
                model_name='add_sub',
                requested_output_names=['OUTPUT0', 'OUTPUT1'])

        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(model_name='add_sub', inputs=[])

    def _get_gpu_bls_outputs(self, input0_pb, input1_pb):
        """
        This function is created to test that the DLPack container works
        properly when the inference response and outputs go out of scope.
        """
        infer_request = pb_utils.InferenceRequest(
            model_name='dlpack_add_sub',
            inputs=[input0_pb, input1_pb],
            requested_output_names=['OUTPUT0', 'OUTPUT1'])
        infer_response = infer_request.exec()
        self.assertFalse(infer_response.has_error())

        output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
        output1 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT1')
        self.assertIsNotNone(output0)
        self.assertIsNotNone(output1)

        # When one of the inputs is in GPU the output returned by the model must
        # be in GPU, otherwise the outputs will be in CPU.
        if not input0_pb.is_cpu() or not input1_pb.is_cpu():
            self.assertTrue((not output0.is_cpu()) and (not output1.is_cpu()))
        else:
            self.assertTrue((output0.is_cpu()) and (output1.is_cpu()))

        # Make sure that the reference count is increased by one when DLPack
        # representation is created.
        rc_before_dlpack_output0 = sys.getrefcount(output0)
        rc_before_dlpack_output1 = sys.getrefcount(output1)

        output0_dlpack = output0.to_dlpack()
        output1_dlpack = output1.to_dlpack()

        rc_after_dlpack_output0 = sys.getrefcount(output0)
        rc_after_dlpack_output1 = sys.getrefcount(output1)

        self.assertEqual(rc_after_dlpack_output0 - rc_before_dlpack_output0, 1)
        self.assertEqual(rc_after_dlpack_output1 - rc_before_dlpack_output1, 1)

        # Make sure that reference count decreases after destroying the DLPack
        output0_dlpack = None
        output1_dlpack = None
        rc_after_del_dlpack_output0 = sys.getrefcount(output0)
        rc_after_del_dlpack_output1 = sys.getrefcount(output1)
        self.assertEqual(rc_after_del_dlpack_output0 - rc_after_dlpack_output0,
                         -1)
        self.assertEqual(rc_after_del_dlpack_output1 - rc_after_dlpack_output1,
                         -1)

        return output0.to_dlpack(), output1.to_dlpack()

    def test_zero_length_io(self):
        model_name = 'identity_fp32'
        input0 = np.zeros([1, 0], dtype=np.float32)
        input0_pb = pb_utils.Tensor('INPUT0', input0)
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=[input0_pb],
            requested_output_names=['OUTPUT0'])
        infer_response = infer_request.exec()
        self.assertFalse(infer_response.has_error())

        output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
        self.assertTrue(np.all(output0 == input0))

    def test_bls_tensor_lifecycle(self):
        model_name = 'dlpack_identity'

        # A 10 MB tensor.
        input_size = 10 * 1024 * 1024

        # Sending the tensor 50 times to test whether the deallocation is
        # happening correctly. If the deallocation doesn't happen correctly,
        # there will be an out of shared memory error.
        for _ in range(50):
            input0 = np.ones([1, input_size], dtype=np.float32)
            input0_pb = pb_utils.Tensor('INPUT0', input0)
            infer_request = pb_utils.InferenceRequest(
                model_name=model_name,
                inputs=[input0_pb],
                requested_output_names=['OUTPUT0'])
            infer_response = infer_request.exec()
            self.assertFalse(infer_response.has_error())

            output0 = pb_utils.get_output_tensor_by_name(
                infer_response, 'OUTPUT0')
            np.testing.assert_equal(output0.as_numpy(), input0,
                                    "BLS CPU memory lifecycle failed.")

        # Checking the same with the GPU tensors.
        for index in range(50):
            input0 = None
            infer_request = None
            input0_pb = None

            torch.cuda.empty_cache()
            free_memory, _ = torch.cuda.mem_get_info()
            if index == 1:
                recorded_memory = free_memory

            if index > 1:
                self.assertEqual(free_memory, recorded_memory,
                                 "GPU memory lifecycle test failed.")

            input0 = torch.ones([1, input_size], dtype=torch.float32).to('cuda')
            input0_pb = pb_utils.Tensor.from_dlpack('INPUT0', to_dlpack(input0))
            infer_request = pb_utils.InferenceRequest(
                model_name=model_name,
                inputs=[input0_pb],
                requested_output_names=['OUTPUT0'])
            infer_response = infer_request.exec()
            self.assertFalse(infer_response.has_error())

            output0 = pb_utils.get_output_tensor_by_name(
                infer_response, 'OUTPUT0')
            output0_pytorch = from_dlpack(output0.to_dlpack())

            # Set inference response and output0_pytorch to None, to make sure
            # that the DLPack is still valid.
            output0 = None
            infer_response = None
            self.assertTrue(
                torch.all(output0_pytorch == input0),
                f"input ({input0}) and output ({output0_pytorch}) didn't match for identity model."
            )

    def _test_gpu_bls_add_sub(self, is_input0_gpu, is_input1_gpu):
        input0 = torch.rand(16)
        input1 = torch.rand(16)

        if is_input0_gpu:
            input0 = input0.to('cuda')

        if is_input1_gpu:
            input1 = input1.to('cuda')

        input0_pb = pb_utils.Tensor.from_dlpack('INPUT0', to_dlpack(input0))
        input1_pb = pb_utils.Tensor.from_dlpack('INPUT1', to_dlpack(input1))
        output0_dlpack, output1_dlpack = self._get_gpu_bls_outputs(
            input0_pb, input1_pb)

        expected_output_0 = from_dlpack(
            input0_pb.to_dlpack()).to('cpu') + from_dlpack(
                input1_pb.to_dlpack()).to('cpu')
        expected_output_1 = from_dlpack(
            input0_pb.to_dlpack()).to('cpu') - from_dlpack(
                input1_pb.to_dlpack()).to('cpu')

        self.assertTrue(
            torch.all(
                expected_output_0 == from_dlpack(output0_dlpack).to('cpu')))
        self.assertTrue(
            torch.all(
                expected_output_1 == from_dlpack(output1_dlpack).to('cpu')))

    def test_gpu_bls(self):
        for input0_device in [True, False]:
            for input1_device in [True, False]:
                self._test_gpu_bls_add_sub(input0_device, input1_device)

    def test_multiprocess(self):
        # Test multiprocess Pool with sync BLS
        pool = Pool(10)
        pool.map(bls_add_sub, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pool.close()
        pool.join()

    def test_bls_sync(self):
        infer_request = pb_utils.InferenceRequest(
            model_name='non_existent_model',
            inputs=[],
            requested_output_names=[])
        infer_response = infer_request.exec()

        # Because the model doesn't exist, the inference response must have an
        # error
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "Failed for execute the inference request. Model 'non_existent_model' is not ready.",
            infer_response.error().message())

        # Make sure that the inference requests can be performed properly after
        # an error.
        self.assertTrue(bls_add_sub())

    def test_bls_execute_error(self):
        # Test BLS with a model that has an error during execution.
        infer_request = pb_utils.InferenceRequest(model_name='execute_error',
                                                  inputs=[],
                                                  requested_output_names=[])
        infer_response = infer_request.exec()
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "expected 1 inputs but got 0 inputs for model 'execute_error'",
            infer_response.error().message())
        self.assertTrue(len(infer_response.output_tensors()) == 0)

    def test_multiple_bls(self):
        # Test running multiple BLS requests together
        for _ in range(100):
            self.assertTrue(bls_add_sub())


class TritonPythonModel:

    def execute(self, requests):
        responses = []
        for _ in requests:
            # Run the unittest and store the results in InferenceResponse.
            test = unittest.main('model', exit=False)
            responses.append(
                pb_utils.InferenceResponse([
                    pb_utils.Tensor(
                        'OUTPUT0',
                        np.array([test.result.wasSuccessful()],
                                 dtype=np.float16))
                ]))
        return responses
