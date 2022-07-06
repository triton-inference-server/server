#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append("../common")

import unittest
import numpy as np
from PIL import Image
import test_util as tu

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class InferTest(tu.TestResultCollector):

    def _preprocess(self, img, dtype):
        """
        Pre-process an image to meet the size and type
        requirements specified by the parameters.
        """

        sample_img = img.convert('RGB')
        resized_img = sample_img.resize((224, 224), Image.BILINEAR)
        resized = np.array(resized_img)

        typed = resized.astype(dtype)
        scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
        ordered = np.transpose(scaled, (2, 0, 1))

        return ordered

    def test_resnet50(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000")
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        image_filename = "../images/vulture.jpeg"
        model_name = "resnet50_plan"
        batch_size = 32

        img = Image.open(image_filename)
        image_data = self._preprocess(img, np.int8)
        image_data = np.expand_dims(image_data, axis=0)

        batched_image_data = image_data
        for i in range(1, batch_size):
            batched_image_data = np.concatenate(
                (batched_image_data, image_data), axis=0)

        inputs = [
            httpclient.InferInput('input_tensor_0', [batch_size, 3, 224, 224],
                                  'INT8')
        ]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput('topk_layer_output_index',
                                            binary_data=True)
        ]

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output_data = results.as_numpy('topk_layer_output_index')
        print(output_data)

        # Validate the results by comparing with precomputed values.
        # VULTURE class corresponds with index 23
        EXPECTED_CLASS_INDEX = 23
        for i in range(batch_size):
            self.assertEqual(output_data[i][0][0], EXPECTED_CLASS_INDEX)


if __name__ == '__main__':
    unittest.main()
