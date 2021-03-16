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
from builtins import range
import os
import sys

import numpy as np


def np_to_model_dtype(np_dtype):
    if np_dtype == bool:
        return "TYPE_BOOL"
    elif np_dtype == np.int8:
        return "TYPE_INT8"
    elif np_dtype == np.int16:
        return "TYPE_INT16"
    elif np_dtype == np.int32:
        return "TYPE_INT32"
    elif np_dtype == np.int64:
        return "TYPE_INT64"
    elif np_dtype == np.uint8:
        return "TYPE_UINT8"
    elif np_dtype == np.uint16:
        return "TYPE_UINT16"
    elif np_dtype == np.float16:
        return "TYPE_FP16"
    elif np_dtype == np.float32:
        return "TYPE_FP32"
    elif np_dtype == np.float64:
        return "TYPE_FP64"
    elif np_dtype == np_dtype_string:
        return "TYPE_STRING"
    return None


def np_to_trt_dtype(np_dtype):
    if np_dtype == bool:
        return trt.bool
    elif np_dtype == np.int8:
        return trt.int8
    elif np_dtype == np.int32:
        return trt.int32
    elif np_dtype == np.float16:
        return trt.float16
    elif np_dtype == np.float32:
        return trt.float32
    return None


def create_plan_modelfile(models_dir, model_version, dtype):
    # Create specific model for L0_trt_batch_input. Because the ragged input
    # nature and the server is not supporting output scattering,
    # 'BATCH_AND_SIZE_INPUT' is used as hint to generate output with batch
    # dimension, 'BATCH_AND_SIZE_INPUT' must have shape [batch_size]. In the
    # context of BATCH_INPUT, that implies the test must ensure the requests and
    # kind of 'BATCH_AND_SIZE_INPUT' are configured to form end-tensor with the
    # right shape, i.e. send batch-1 request with kind 'BATCH_ELEMENT_COUNT'.
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt_dtype = np_to_trt_dtype(dtype)

    in_node = network.add_input("RAGGED_INPUT", trt_dtype, [-1])
    bs_node = network.add_input("BATCH_AND_SIZE_INPUT", trt_dtype, [-1])
    batch_node = network.add_input("BATCH_INPUT", trt_dtype, [-1])

    reshape_dims = trt.Dims([-1, 1])
    in_mat = network.add_shuffle(in_node)
    in_mat.reshape_dims = reshape_dims
    bs_mat = network.add_shuffle(bs_node)
    bs_mat.reshape_dims = reshape_dims
    batch_mat = network.add_shuffle(batch_node)
    batch_mat.reshape_dims = reshape_dims

    batch_entry = network.add_elementwise(bs_mat.get_output(0),
                                          bs_mat.get_output(0),
                                          trt.ElementWiseOperation.DIV)
    out_node = network.add_matrix_multiply(batch_entry.get_output(0),
                                           trt.MatrixOperation.NONE,
                                           in_mat.get_output(0),
                                           trt.MatrixOperation.TRANSPOSE)
    bs_out_node = network.add_matrix_multiply(batch_entry.get_output(0),
                                              trt.MatrixOperation.NONE,
                                              bs_mat.get_output(0),
                                              trt.MatrixOperation.TRANSPOSE)
    batch_out_node = network.add_matrix_multiply(batch_entry.get_output(0),
                                                 trt.MatrixOperation.NONE,
                                                 batch_mat.get_output(0),
                                                 trt.MatrixOperation.TRANSPOSE)
    out_node.get_output(0).name = "RAGGED_OUTPUT"
    bs_out_node.get_output(0).name = "BATCH_AND_SIZE_OUTPUT"
    batch_out_node.get_output(0).name = "BATCH_OUTPUT"
    network.mark_output(out_node.get_output(0))
    network.mark_output(bs_out_node.get_output(0))
    network.mark_output(batch_out_node.get_output(0))

    # Hard coded optimization profile
    min_shape = [1]
    opt_shape = [8]
    max_shape = [32]

    profile = builder.create_optimization_profile()
    for input_name in ["RAGGED_INPUT", "BATCH_AND_SIZE_INPUT", "BATCH_INPUT"]:
        profile.set_shape("{}".format(input_name), min_shape, opt_shape,
                          max_shape)
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    model_name = "plan_batch_input"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def create_plan_modelconfig(models_dir, max_batch, model_version, dtype):
    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = "plan_batch_input"
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "RAGGED_INPUT"
    data_type: {data_type}
    dims: [ -1 ]
    allow_ragged_batch: true
  }}
]
output [
  {{
    name: "RAGGED_OUTPUT"
    data_type: {data_type}
    dims: [ -1 ]
   }}
]
output [
  {{
    name: "BATCH_AND_SIZE_OUTPUT"
    data_type: {data_type}
    dims: [ -1 ]
   }}
]
output [
  {{
    name: "BATCH_OUTPUT"
    data_type: {data_type}
    dims: [ -1 ]
   }}
]
batch_input [
  {{
    kind: BATCH_ELEMENT_COUNT
    target_name: "BATCH_AND_SIZE_INPUT"
    data_type: {data_type}
    source_input: "RAGGED_INPUT"
  }},
  {{
    kind: BATCH_ACCUMULATED_ELEMENT_COUNT
    target_name: "BATCH_INPUT"
    data_type: {data_type}
    source_input: "RAGGED_INPUT"
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 1000000
}}
'''.format(model_name,
           max_batch,
           version_policy_str,
           data_type=np_to_model_dtype(dtype))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_batch_input_models(models_dir):
    model_version = 1
    if FLAGS.tensorrt:
        create_plan_modelconfig(models_dir, 8, model_version, np.float32)
        create_plan_modelfile(models_dir, model_version, np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    parser.add_argument('--tensorrt',
                        required=False,
                        action='store_true',
                        help='Generate TensorRT PLAN models')
    FLAGS, unparsed = parser.parse_known_args()

    import test_util as tu
    if FLAGS.tensorrt:
        import tensorrt as trt

    create_batch_input_models(FLAGS.models_dir)
