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
import unittest
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils


class PBTensorTest(unittest.TestCase):

    def test_pytorch_dlpack(self):
        # Test different dtypes
        pytorch_dtypes = [
            torch.float16, torch.float32, torch.float64, torch.int8,
            torch.int16, torch.int32, torch.int64, torch.uint8
        ]

        for pytorch_dtype in pytorch_dtypes:
            pytorch_tensor = torch.ones([100], dtype=pytorch_dtype)
            dlpack_tensor = to_dlpack(pytorch_tensor)
            pb_tensor = pb_utils.Tensor.from_dlpack('test_tensor',
                                                    dlpack_tensor)
            self.assertTrue(
                np.all(pb_tensor.as_numpy() == pytorch_tensor.numpy()))

            # Convert the tensor back to DLPack and ensure that both tensors are
            # the same
            pytorch_tensor_dlpack = from_dlpack(pb_tensor.to_dlpack())
            self.assertTrue(torch.all(pytorch_tensor_dlpack == pytorch_tensor))

            self.assertTrue(
                pytorch_tensor.type() == pytorch_tensor_dlpack.type())

            # Now let's check that upgraded DLPack implementation also
            # works as expected, i.e. from_dlpack should work with 
            # external pytorch tensor directly

            pb_tensor_upgraded = pb_utils.Tensor.from_dlpack('test_tensor',
                                                    pytorch_tensor)
            self.assertTrue(
                np.all(pb_tensor_upgraded.as_numpy() == pytorch_tensor.numpy()))

            # Here we check that `pb_tensor` as a producer, properly
            # invokes `__dlpack__` and `__dlpack_device__`
            pytorch_tensor_dlpack = from_dlpack(pb_tensor_upgraded)
            self.assertTrue(torch.all(pytorch_tensor_dlpack == pytorch_tensor))

            self.assertTrue(
                pytorch_tensor.type() == pytorch_tensor_dlpack.type())  

    def test_non_contiguous_error(self):
        pytorch_tensor = torch.rand([20, 30], dtype=torch.float16)

        # Transposing a PyTorch tensor leads to a non contiguous tensor.
        pytorch_tensor = torch.transpose(pytorch_tensor, 0, 1)

        with self.assertRaises(Exception) as e:
            pb_utils.Tensor.from_dlpack('test_tensor',
                                        to_dlpack(pytorch_tensor))
        self.assertTrue(
            str(e.exception) ==
            'DLPack tensor is not contiguous. Only contiguous DLPack tensors that are stored in C-Order are supported.'
        )

    def test_dlpack_string_tensor(self):
        np_object = np.array(['An Example String'], dtype=np.object_)
        pb_tensor = pb_utils.Tensor('test_tensor', np_object)

        with self.assertRaises(Exception) as e:
            pb_tensor.to_dlpack()

        self.assertTrue(
            str(e.exception) ==
            'DLPack does not have support for string tensors.')

    def test_dlpack_gpu_tensors(self):
        # Test different dtypes
        # PyTorch does not support DLPack bool type yet:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/DLConvertor.cpp
        pytorch_dtypes = [
            torch.float16, torch.float32, torch.float64, torch.int8,
            torch.int16, torch.int32, torch.int64, torch.uint8
        ]

        for pytorch_dtype in pytorch_dtypes:
            pytorch_tensor = torch.ones([100],
                                        dtype=pytorch_dtype,
                                        device='cuda')
            dlpack_tensor = to_dlpack(pytorch_tensor)
            pb_tensor = pb_utils.Tensor.from_dlpack('test_tensor',
                                                    dlpack_tensor)

            # Convert the tensor back to DLPack and ensure that both tensors are
            # the same
            pytorch_tensor_dlpack = from_dlpack(pb_tensor.to_dlpack())
            self.assertTrue(torch.all(pytorch_tensor_dlpack == pytorch_tensor))
            self.assertTrue(
                pytorch_tensor.type() == pytorch_tensor_dlpack.type())
            
            # Now we make sure that updated DLPack implementation works
            # with GPU as well
            pb_tensor = pb_utils.Tensor.from_dlpack('test_tensor',
                                                    pytorch_tensor)
            pytorch_tensor_dlpack = from_dlpack(pb_tensor)
            self.assertTrue(torch.all(pytorch_tensor_dlpack == pytorch_tensor))
            self.assertTrue(
                pytorch_tensor.type() == pytorch_tensor_dlpack.type())
            

    def test_dlpack_gpu_numpy(self):
        # DLPack tesnors that are in GPU cannot be converted to NumPy
        pytorch_tensor = torch.rand([100], dtype=torch.float16,
                                    device='cuda') * 100
        pb_tensor = pb_utils.Tensor.from_dlpack('tensor',
                                                to_dlpack(pytorch_tensor))
        # Make sure that `__dlpack_device__` works as expected
        self.assertFalse(pb_tensor.is_cpu())
        self.assertTrue(pytorch_tensor.is_cuda)
        self.assertEqual(pb_tensor.__dlpack_device__(), pytorch_tensor.__dlpack_device__())
    
        with self.assertRaises(Exception) as e:
            pb_tensor.as_numpy()
        self.assertTrue(
            str(e.exception) ==
            'Tensor is stored in GPU and cannot be converted to NumPy.')
    
    def test_dlpack_cpu_numpy(self):
        # Check compatibiity of PbTensor DLPack implementation
        # with numpy
        pytorch_tensor = torch.rand([100], dtype=torch.float16,
                                    device='cpu') * 100
        pb_tensor = pb_utils.Tensor.from_dlpack('tensor', pytorch_tensor)
        numpy_tensor_dlpack = np.from_dlpack(pb_tensor)
        self.assertTrue(np.all(numpy_tensor_dlpack == pytorch_tensor.numpy()))
        # Make sure that `__dlpack_device__` works as expected
        self.assertTrue(pb_tensor.is_cpu())
        self.assertFalse(pytorch_tensor.is_cuda)
        self.assertEqual(pb_tensor.__dlpack_device__(), pytorch_tensor.__dlpack_device__())
    
    def test_pdtensor_bool_internal_support(self):
        # [FIXME] pass bool_array directly to `pb_utils.Tensor.from_dlpack`,
        # when numpy release supports DLPack bool type
        bool_array = np.asarray([False, True])
        bool_tensor = pb_utils.Tensor('tensor', bool_array)
        bool_tensor_dlpack = pb_utils.Tensor.from_dlpack('tensor', bool_tensor)
        self.assertTrue(np.all(bool_array == bool_tensor_dlpack.as_numpy()))


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
