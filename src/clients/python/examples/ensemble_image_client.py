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
import os
from builtins import range
from PIL import Image
import sys

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as model_config
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

FLAGS = None


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs

    return (input_metadata.name, output_metadata, model_config.max_batch_size)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs']

    return (input_metadata['name'], output_metadata,
            model_config['max_batch_size'])


def postprocess(results, output_names, filenames, batch_size):
    """
    Post-process results to show classifications.
    """
    output_dict = {}
    for output_name in output_names:
        output_dict[output_name] = results.as_numpy(output_name)
        if len(output_dict[output_name]) != batch_size:
            raise Exception("expected {} results for output {}, got {}".format(
                batch_size, output_name, len(output_dict[output_name])))

    for n, f in enumerate(filenames):
        print('\n"{}":'.format(f))
        for output_name in output_names:
            print('  [{}]:'.format(output_name))
            for result in output_dict[output_name][n]:
                if output_dict[output_name][n].dtype.type == np.object_:
                    cls = "".join(chr(x) for x in result).split(':')
                else:
                    cls = result.split(':')
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-m',
        '--model-name',
        type=str,
        required=False,
        default='preprocess_inception_ensemble',
        help='Name of model. Default is preprocess_inception_ensemble.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to ' +
                        'communicate with inference service. Default is HTTP.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    protocol = FLAGS.protocol.lower()

    try:
        if protocol == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            # Create HTTP client for communicating with the server
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    model_name = FLAGS.model_name

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        input_name, output_metadata, batch_size = parse_model_grpc(
            model_metadata, model_config.config)
    else:
        input_name, output_metadata, batch_size = parse_model_http(
            model_metadata, model_config)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    # Set batch size to the smaller value of image size and max batch size
    if len(filenames) <= batch_size:
        batch_size = len(filenames)
    else:
        print("The number of images exceeds maximum batch size," \
                "only the first {} images, sorted by name alphabetically," \
                " will be processed".format(batch_size))

    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    for idx in range(batch_size):
        with open(filenames[idx], "rb") as fd:
            image_data.append(np.array([fd.read()], dtype=bytes))

    # Send requests of batch_size images.
    input_filenames = []
    repeated_image_data = []
    for idx in range(batch_size):
        input_filenames.append(filenames[idx])
        repeated_image_data.append(image_data[idx])

    batched_image_data = np.stack(repeated_image_data, axis=0)

    # Set the input data
    inputs = []
    if FLAGS.protocol.lower() == "grpc":
        inputs.append(
            grpcclient.InferInput(input_name, batched_image_data.shape,
                                  "BYTES"))
        inputs[0].set_data_from_numpy(batched_image_data)
    else:
        inputs.append(
            httpclient.InferInput(input_name, batched_image_data.shape,
                                  "BYTES"))
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    output_names = [
        output.name if FLAGS.protocol.lower() == "grpc" else output['name']
        for output in output_metadata
    ]

    outputs = []
    for output_name in output_names:
        if FLAGS.protocol.lower() == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(output_name,
                                                class_count=FLAGS.classes))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(output_name,
                                                binary_data=True,
                                                class_count=FLAGS.classes))

    # Send request
    result = triton_client.infer(model_name, inputs, outputs=outputs)

    postprocess(result, output_names, input_filenames, batch_size)

    print("PASS")
