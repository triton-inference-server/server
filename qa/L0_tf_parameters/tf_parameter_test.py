# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
sys.path.append('../common')

import test_util as tu
import tritonclient.http as tritonhttpclient
import tritonclient.utils
import numpy as np
import unittest


class TFParameterTest(tu.TestResultCollector):

    def setUp(self):
        self._client = tritonhttpclient.InferenceServerClient("localhost:8000",
                                                              verbose=True)

    def _infer_helper(self):
        # The model has a single variable which is added to the input.  Since the
        # variable is initialized to zero the input and output must match.
        model_name = 'graphdef_variable'
        input = np.array([10], dtype=np.int32)

        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT', input.shape,
                                                  'INT32'))
        inputs[-1].set_data_from_numpy(input)

        outputs = []
        outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT'))

        results = self._client.infer(model_name=model_name,
                                     inputs=inputs,
                                     outputs=outputs)
        output = results.as_numpy('OUTPUT')
        np.testing.assert_array_equal(output, input)

    def test_tf_variable(self):
        self._infer_helper()

    def test_tf_variable_error(self):
        with self.assertRaises(
                tritonclient.utils.InferenceServerException) as e:
            self._infer_helper()
        self.assertIn("Attempting to use uninitialized value VARIABLE",
                      e.exception.message())


if __name__ == '__main__':
    unittest.main()
