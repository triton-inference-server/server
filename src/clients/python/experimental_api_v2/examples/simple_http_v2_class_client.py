#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from builtins import range
from PIL import Image
import sys

import tritonhttpclient.core as httpclient
from tritonhttpclient.utils import InferenceServerException
from tritonhttpclient.utils import triton_to_np_dtype

FLAGS = None


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_metadata = model_metadata['outputs'][0]

    if output_metadata['datatype'] != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata['name'] + "' output type is " +
                        output_metadata['datatype'])

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok).
    non_one_cnt = 0
    for dim in output_metadata['shape']:
        if dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC
    if len(input_metadata['shape']) != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".
            format(model_metadata.name, len(input_metadata['shape'])))

    if ((input_config['format'] != "FORMAT_NCHW") and
        (input_config['format'] != "FORMAT_NHWC")):
        raise Exception("unexpected input format " + input_config['format'] 
                    + ", expecting FORMAT_NCHW or FORMAT_NHWC")

    if input_config['format'] == "FORMAT_NHWC":
        h = input_metadata['shape'][0]
        w = input_metadata['shape'][1]
        c = input_metadata['shape'][2]
    else:
        c = input_metadata['shape'][0]
        h = input_metadata['shape'][1]
        w = input_metadata['shape'][2]

    return (input_metadata['name'], output_metadata['name'], c, h, w,
            "FORMAT_NHWC", input_metadata['datatype'])


def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    #np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == "FORMAT_NCHW":
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)
    if not "VULTURE" in str(output_array[0][0]):
        raise Exception(
            "expected VULTURE as the first result, instead received: {}".format(
                output_array[0][0]))


def requestGenerator(input_name, output_name, c, h, w, format, dtype, FLAGS):
    inputs = []
    inputs.append(httpclient.InferInput(input_name))

    # Preprocess image into input data according to model requirements
    image_data = None
    with Image.open(FLAGS.image_filename) as img:
        image_data = preprocess(img, format, dtype, c, h, w, FLAGS.scaling)

    repeated_image_data = [image_data for _ in range(FLAGS.batch_size)]
    batched_image_data = np.stack(repeated_image_data, axis=0)

    # Set the input data
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = []
    outputs.append(httpclient.InferOutput(output_name, class_count=FLAGS.classes))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument('-x',
                        '--model-version',
                        type=str,
                        required=False,
                        default="",
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-s',
                        '--scaling',
                        type=str,
                        choices=['NONE', 'INCEPTION', 'VGG'],
                        required=False,
                        default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('image_filename', type=str, help='Input image.')
    FLAGS = parser.parse_args()

    # Create HTTP client for communicating with the server
    try:
        triton_client = httpclient.InferenceServerClient(FLAGS.url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(model_name=FLAGS.model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit()

    try:
        model_config = triton_client.get_model_config(model_name=FLAGS.model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit()

    input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    results = []

    try:
        for inputs, outputs, model_name, model_version in requestGenerator(
                input_name, output_name, c, h, w, format, dtype, FLAGS):
            results.append(triton_client.infer(model_name,
                                            inputs,
                                            outputs=outputs,
                                            model_version=model_version))
    except InferenceServerException as e:
        print("inference failed: " + str(e))
        sys.exit()

    for result in results:
        postprocess(result, output_name, FLAGS.batch_size)

    print("PASS : Classification")
