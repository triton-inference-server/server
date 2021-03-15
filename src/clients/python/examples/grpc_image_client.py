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
from PIL import Image
import os
import sys
import struct

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

FLAGS = None


def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None


def deserialize_bytes_tensor(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (input_metadata.name, output_metadata.name, c, h, w,
            input_config.format, input_metadata.datatype)


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

    npdtype = model_dtype_to_np(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(response, filenames, batch_size):
    """
    Post-process response to show classifications.
    """
    if len(response.outputs) != 1:
        raise Exception("expected 1 output, got {}".format(len(
            response.outputs)))

    if len(response.raw_output_contents) != 1:
        raise Exception("expected 1 output content, got {}".format(
            len(response.raw_output_contents)))

    batched_result = deserialize_bytes_tensor(response.raw_output_contents[0])
    contents = np.reshape(batched_result, response.outputs[0].shape)

    if len(contents) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(contents)))
    if len(filenames) != batch_size:
        raise Exception("expected {} filenames, got {}".format(
            batch_size, len(filenames)))

    for (index, results) in enumerate(contents):
        print("Image '{}':".format(filenames[index]))
        for result in results:
            cls = "".join(chr(x) for x in result).split(':')
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


def requestGenerator(input_name, output_name, c, h, w, format, dtype, FLAGS,
                     result_filenames):
    request = service_pb2.ModelInferRequest()
    request.model_name = FLAGS.model_name
    request.model_version = FLAGS.model_version

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

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = output_name
    output.parameters['classification'].int64_param = FLAGS.classes
    request.outputs.extend([output])

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = dtype
    if format == mc.ModelInput.FORMAT_NHWC:
        input.shape.extend([FLAGS.batch_size, h, w, c])
    else:
        input.shape.extend([FLAGS.batch_size, c, h, w])

    # Preprocess image into input data according to model requirements
    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(preprocess(img, format, dtype, c, h, w,
                                     FLAGS.scaling))

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    image_idx = 0
    last_request = False
    while not last_request:
        input_bytes = None
        input_filenames = []
        request.ClearField("inputs")
        request.ClearField("raw_input_contents")
        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            if input_bytes is None:
                input_bytes = image_data[image_idx].tobytes()
            else:
                input_bytes += image_data[image_idx].tobytes()

            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        request.inputs.extend([input])
        result_filenames.append(input_filenames)
        request.raw_input_contents.extend([input_bytes])
        yield request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
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
    parser.add_argument(
        '-s',
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
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version)
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(name=FLAGS.model_name,
                                                    version=FLAGS.model_version)
    config_response = grpc_stub.ModelConfig(config_request)

    input_name, output_name, c, h, w, format, dtype = parse_model(
        metadata_response, config_response.config)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []

    # Send request
    if FLAGS.streaming:
        for response in grpc_stub.ModelStreamInfer(
                requestGenerator(input_name, output_name, c, h, w, format,
                                 dtype, FLAGS, result_filenames)):
            responses.append(response)
    else:
        for request in requestGenerator(input_name, output_name, c, h, w,
                                        format, dtype, FLAGS, result_filenames):
            if not FLAGS.async_set:
                responses.append(grpc_stub.ModelInfer(request))
            else:
                requests.append(grpc_stub.ModelInfer.future(request))

    # For async, retrieve results according to the send order
    if FLAGS.async_set:
        for request in requests:
            responses.append(request.result())

    error_found = False
    idx = 0
    for response in responses:
        if FLAGS.streaming:
            if response.error_message != "":
                error_found = True
                print(response.error_message)
            else:
                postprocess(response.infer_response, result_filenames[idx],
                            FLAGS.batch_size)
        else:
            postprocess(response, result_filenames[idx], FLAGS.batch_size)
        idx += 1

    if error_found:
        sys.exit(1)

    print("PASS")
