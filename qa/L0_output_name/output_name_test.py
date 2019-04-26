#!/bin/bash
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
from functools import partial
from PIL import Image

import grpc
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config

import json
import requests
FLAGS = None

def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:,:,np.newaxis]

    typed = resized.astype(dtype)

    # Swap to CHW if necessary
    if format == model_config.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(typed, (2, 0, 1))
    else:
        ordered = typed

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered

def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None

def parse_model(status, model_name, batch_size, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    server_status = status.server_status
    if model_name not in server_status.model_status.keys():
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))
    if len(config.output) != 1:
        raise Exception("expecting 1 output, got {}".format(len(config.output)))

    input = config.input[0]
    output = config.output[0]

    if output.data_type != model_config.TYPE_FP32:
        raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                        model_name + "' output type is " +
                        model_config.DataType.Name(output.data_type))

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok).
    non_one_cnt = 0
    for dim in output.dims:
        if dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    max_batch_size = config.max_batch_size
    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_name + "'")
    else: # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception(
                "expecting batch size <= {} for model '{}'".format(max_batch_size, model_name))

    # Model input must have 3 dims, either CHW or HWC
    if len(input.dims) != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".format(
                model_name, len(input.dims)))

    if ((input.format != model_config.ModelInput.FORMAT_NCHW) and
        (input.format != model_config.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " + model_config.ModelInput.Format.Name(input.format) +
                        ", expecting " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW) +
                        " or " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC))

    if input.format == model_config.ModelInput.FORMAT_NHWC:
        h = input.dims[0]
        w = input.dims[1]
        c = input.dims[2]
    else:
        c = input.dims[0]
        h = input.dims[1]
        w = input.dims[2]

    return (input.name, output.name, c, h, w, input.format, model_dtype_to_np(input.data_type))

def requestGenerator(input_name, output_name, c, h, w, format, dtype, FLAGS):
    # Prepare request for Infer gRPC
    # The meta data part can be reused across requests
    request = grpc_service_pb2.InferRequest()
    request.model_name = FLAGS.model_name
    if FLAGS.model_version is None:
        request.model_version = -1
    else:
        request.model_version = FLAGS.model_version
    request.meta_data.batch_size = 1
    output_message = api_pb2.InferRequestHeader.Output()
    output_message.name = output_name
    output_message.cls.count = FLAGS.classes
    request.meta_data.output.extend([output_message])

    # Preprocess the images into input data according to model
    # requirements
    img = Image.open(FLAGS.image_filename)
    image_data = preprocess(img, format, dtype, c, h, w, 'NONE')

    request.meta_data.input.add(name=input_name)

    input_bytes = image_data.tobytes()

    del request.raw_input[:]
    request.raw_input.extend([input_bytes])
    return request

def TestGRPC(FLAGS):
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

    # Prepare request for Status gRPC
    request = grpc_service_pb2.StatusRequest(model_name=FLAGS.model_name)
    # Call and receive response from Status gRPC
    response = grpc_stub.Status(request)
    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    batch_size = 1
    input_name, output_name, c, h, w, format, dtype = parse_model(
        response, FLAGS.model_name, batch_size, FLAGS.verbose)

    request_ = requestGenerator(input_name, "DUMMY", c, h, w, format, dtype, FLAGS)
    # Send request
    response_ = grpc_stub.Infer(request)
    return str(response).find("code: SUCCESS")!=-1

def TestHTTP(FLAGS):
    url_ = 'http://'+FLAGS.url+'/api/infer/'+FLAGS.model_name

    with open('preprocessed', 'rb') as mf:
      data = mf.read()

    payload = data
    headers = {'NV-InferRequest': 'batch_size: 1 input { name: "input" } output { name: "DUMMY" cls { count: 3 } }'}

    r = requests.post(url_, data=data, headers=headers)

    return r.status_code!=200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Name of model')
    parser.add_argument('-x', '--model-version', type=int, required=False,
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('image_filename', type=str, nargs='?', default=None,
                        help='Input image.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='grpc',
                        help='Protocol (HTTP/gRPC) used to ' +
                        'communicate with inference service. Default is HTTP.')
    FLAGS = parser.parse_args()

    if FLAGS.protocol == "grpc":
        assert TestGRPC(FLAGS) == True
    elif FLAGS.protocol =="http":
        assert TestHTTP(FLAGS) == True
