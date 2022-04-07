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

import numpy as np
import unittest
import triton_python_backend_utils as pb_utils


class ArgumentValidationTest(unittest.TestCase):
    def test_infer_request_args(self):
        # Dummy arguments used in the tests.
        inputs = [pb_utils.Tensor('INPUT0', np.asarray([1, 2], dtype=np.int32))]
        model_name = 'my_model'
        requested_output_names = ['my_output']

        # inputs field validation
        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(inputs=[None], model_name=model_name,
                    requested_output_names=requested_output_names)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(inputs=None, model_name=model_name,
                    requested_output_names=requested_output_names)

        # model_name validation
        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(model_name=None, inputs=inputs,
                    requested_output_names=requested_output_names)

        # Requested output name validations
        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=[None],
                    inputs=inputs, model_name=model_name)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=None,
                    inputs=inputs, model_name=model_name)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=None,
                    inputs=inputs, model_name=model_name)

        # Other arguments validation
        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=requested_output_names,
                    inputs=inputs, model_name=model_name, correleation_id=None)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=requested_output_names,
                    inputs=inputs, model_name=model_name, request_id=None)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=requested_output_names,
                    inputs=inputs, model_name=model_name, model_version=None)

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceRequest(requested_output_names=requested_output_names,
                    inputs=inputs, model_name=model_name, flags=None)

        # This should not raise an exception
        pb_utils.InferenceRequest(requested_output_names=[], inputs=[],
                model_name=model_name)


    def test_infer_response_args(self):
        outputs = [pb_utils.Tensor('OUTPUT0', np.asarray([1, 2], dtype=np.int32))]

        # Inference Response
        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceResponse(output_tensors=[None])

        with self.assertRaises(BaseException) as e:
            pb_utils.InferenceResponse(output_tensors=None)

        # This should not raise an exception
        pb_utils.InferenceResponse(output_tensors=[])
        pb_utils.InferenceResponse(outputs)

    def test_tensor_args(self):
        np_array = np.asarray([1, 2], dtype=np.int32)

        with self.assertRaises(BaseException) as e:
            pb_utils.Tensor(None, np_array)

        with self.assertRaises(BaseException) as e:
            pb_utils.Tensor("OUTPUT0", None)

        with self.assertRaises(BaseException) as e:
            pb_utils.Tensor.from_dlpack("OUTPUT0", None)


class TritonPythonModel:
    """This model tests the Python API arguments to make sure invalid args are
    rejected."""

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

