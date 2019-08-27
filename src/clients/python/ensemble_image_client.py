#!/usr/bin/env python

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config

FLAGS = None

def parse_model(url, protocol, model_name, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))
    if len(config.output) != 1:
        raise Exception("expecting 1 output, got {}".format(len(config.output)))

    input = config.input[0]
    output = config.output[0]

    return (input.name, output.name, config.max_batch_size)

def postprocess(results, filenames, batch_size):
    """
    Post-process results to show classifications.
    """
    if len(results) != 1:
        raise Exception("expected 1 result, got {}".format(len(results)))

    batched_result = list(results.values())[0]
    if len(batched_result) != batch_size:
        raise Exception("expected {} results, got {}".format(batch_size, len(batched_result)))
    if len(filenames) != batch_size:
        raise Exception("expected {} filenames, got {}".format(batch_size, len(filenames)))

    for (index, result) in enumerate(batched_result):
        print("Image '{}':".format(filenames[index]))
        for cls in result:
            print("    {} ({}) = {}".format(cls[0], cls[2], cls[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='HTTP',
                        help='Protocol (HTTP/gRPC) used to ' +
                        'communicate with inference service. Default is HTTP.')
    parser.add_argument('image_filename', type=str, nargs='?', default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    protocol = ProtocolType.from_str(FLAGS.protocol)

    input_name, output_name, batch_size = parse_model(
        FLAGS.url, protocol, "preprocess_resnet50_ensemble", FLAGS.verbose)

    ctx = InferContext(FLAGS.url, protocol, "preprocess_resnet50_ensemble",
                       -1, FLAGS.verbose)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [os.path.join(FLAGS.image_filename, f)
                     for f in os.listdir(FLAGS.image_filename)
                     if os.path.isfile(os.path.join(FLAGS.image_filename, f))]
    else:
        filenames = [FLAGS.image_filename,]

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
    input_batch = []
    for idx in range(batch_size):
        input_filenames.append(filenames[idx])
        input_batch.append(image_data[idx])

    # Send request
    result = ctx.run(
        { input_name : input_batch },
        { output_name : (InferContext.ResultFormat.CLASS, FLAGS.classes) },
        batch_size)

    postprocess(result, input_filenames, batch_size)
