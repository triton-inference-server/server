#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import time

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

FLAGS = None


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it is supported
    by this client.
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
    output_metadata = model_metadata.outputs[0]

    batch_dim = (model_config.max_batch_size > 0)
    expected_dims = 1 + (1 if batch_dim else 0)

    if len(input_metadata.shape) != expected_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if len(output_metadata.shape) != expected_dims:
        raise Exception(
            "expecting output to have {} dimensions, model '{}' output has {}".
            format(expected_dims, model_metadata.name,
                   len(output_metadata.shape)))

    if input_metadata.shape[-1] != -1:
        raise Exception(
            "expecting input to have variable shape [-1], model '{}' input has {}"
            .format(model_metadata.name, input_metadata.shape))

    if output_metadata.shape[-1] != -1:
        raise Exception(
            "expecting output to have variable shape [-1], model '{}' output has {}"
            .format(model_metadata.name, output_metadata.shape))

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, input_metadata.datatype)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it is supported
    by this client.
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
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    batch_dim = (max_batch_size > 0)
    expected_dims = 1 + (1 if batch_dim else 0)

    if len(input_metadata['shape']) != expected_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_dims, model_metadata.name,
                   len(input_metadata['shape'])))

    if len(output_metadata['shape']) != expected_dims:
        raise Exception(
            "expecting output to have {} dimensions, model '{}' output has {}".
            format(expected_dims, model_metadata.name,
                   len(output_metadata['shape'])))

    if input_metadata['shape'][-1] != -1:
        raise Exception(
            "expecting input to have variable shape [-1], model '{}' input has {}"
            .format(model_metadata.name, input_metadata['shape']))

    if output_metadata['shape'][-1] != -1:
        raise Exception(
            "expecting output to have variable shape [-1], model '{}' output has {}"
            .format(model_metadata.name, output_metadata['shape']))

    return (max_batch_size, input_metadata['name'], output_metadata['name'],
            input_metadata['datatype'])


def requestGenerator(input_name, input_data, output_name, dtype, protocol):

    # Set the input data
    inputs = []
    if protocol.lower() == "grpc":
        inputs.append(grpcclient.InferInput(input_name, input_data.shape,
                                            dtype))
        inputs[0].set_data_from_numpy(input_data)
    else:
        inputs.append(httpclient.InferInput(input_name, input_data.shape,
                                            dtype))
        inputs[0].set_data_from_numpy(input_data, binary_data=True)

    outputs = []
    if protocol.lower() == "grpc":
        outputs.append(grpcclient.InferRequestedOutput(output_name))
    else:
        outputs.append(
            httpclient.InferRequestedOutput(output_name, binary_data=True))

    return inputs, outputs


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
    parser.add_argument('-s',
                        '--shape',
                        type=int,
                        required=False,
                        default=1,
                        help='The shape of the tensor. Default is 1.')
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
                        help='Protocol (HTTP/gRPC) used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('-c',
                        '--iteration_count',
                        type=int,
                        required=False,
                        default=1000,
                        help='The number of iterations. Default is 1000.')
    parser.add_argument(
        '-w',
        '--warmup_count',
        type=int,
        required=False,
        default=500,
        help='The number of warm-up iterations. Default is 500.')
    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default=None,
        help='The name of the file to store the results in CSV format')
    FLAGS = parser.parse_args()

    try:
        if FLAGS.protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=1)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        max_batch_size, input_name, output_name, dtype = parse_model_grpc(
            model_metadata, model_config.config)
    else:
        max_batch_size, input_name, output_name, dtype = parse_model_http(
            model_metadata, model_config)

    input_data = np.zeros([FLAGS.batch_size, FLAGS.shape],
                          dtype=triton_to_np_dtype(dtype))

    # --------------------------- Warm-Up --------------------------------------------------------
    for i in range(FLAGS.warmup_count):
        inputs, outputs = requestGenerator(input_name, input_data, output_name,
                                           dtype, FLAGS.protocol.lower())
        triton_client.infer(FLAGS.model_name,
                            inputs,
                            model_version=FLAGS.model_version,
                            outputs=outputs)

    latencies = []

    # --------------------------- Start Load --------------------------------------------------------

    start_time = time.time()

    for i in range(FLAGS.iteration_count):
        t0 = time.time()
        inputs, outputs = requestGenerator(input_name, input_data, output_name,
                                           dtype, FLAGS.protocol.lower())
        triton_client.infer(FLAGS.model_name,
                            inputs,
                            model_version=FLAGS.model_version,
                            outputs=outputs)
        latencies.append(time.time() - t0)

    end_time = time.time()

    throughput = FLAGS.iteration_count / (end_time - start_time)
    average_latency = np.average(latencies) * 1000
    p50_latency = np.percentile(latencies, 50) * 1000
    p90_latency = np.percentile(latencies, 90) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000

    # --------------------------- Print Report -----------------------------------------------------
    print("Throughput: {} infer/sec".format(throughput))
    print("Latencies:")
    print("\tAvg: {} ms".format(average_latency))
    print("\tp50: {} ms".format(p50_latency))
    print("\tp90: {} ms".format(p90_latency))
    print("\tp95: {} ms".format(p95_latency))
    print("\tp99: {} ms".format(p99_latency))

    # --------------------------- Write CSV --------------------------------------------------------
    if FLAGS.csv != None:
        file = open(FLAGS.csv, 'w')
        file.write(
            "Concurrency,Inferences/Second,p50 latency,p90 latency,p95 latency,p99 latency\n"
        )
        file.write("1,{},{},{},{},{}".format(throughput, p50_latency * 1000,
                                             p90_latency * 1000,
                                             p95_latency * 1000,
                                             p99_latency * 1000))
        file.close()
