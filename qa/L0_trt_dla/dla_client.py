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

import argparse
import numpy as np
import sys
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def preprocess(img, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size and type
    requirements specified by the parameters.
    """

    sample_img = img.convert('RGB')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)

    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(dtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=dtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
    else:
        scaled = typed

    ordered = np.transpose(scaled, (2, 0, 1))

    return ordered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('image_filename',
                        type=str,
                        default=None,
                        help='Input image.')

    FLAGS = parser.parse_args()
    try:
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "resnet50_plan"
    batch_size = 32

    img = Image.open(FLAGS.image_filename)
    image_data = preprocess(img, np.int8, 3, 224, 224, FLAGS.scaling)
    image_data = np.expand_dims(image_data, axis=0)

    batched_image_data = image_data
    for i in range(1, batch_size):
        batched_image_data = np.concatenate((batched_image_data, image_data), axis=0)

    inputs = [httpclient.InferInput('input_tensor_0', [batch_size, 3, 224, 224], 'INT8')]
    inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    outputs = [
        httpclient.InferRequestedOutput('topk_layer_output_index', binary_data=True)
    ]

    results = triton_client.infer(model_name,
                        inputs,
                        outputs=outputs)

    # Validate the results by comparing with precomputed values.
    # VULTURE class corresponds with index 23
    EXPECTED_CLASS_INDEX = 23

    output_data = results.as_numpy('topk_layer_output_index')
    print(output_data)
    for i in range(batch_size):
        if output_data[i][0][0] != EXPECTED_CLASS_INDEX:
            print("Infer error: Incorrect output label")
            sys.exit(1)
