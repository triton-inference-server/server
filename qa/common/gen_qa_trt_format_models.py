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
import os
import numpy as np
import tensorrt as trt
import test_util as tu


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


def trt_format_to_string(trt_format):
    # FIXME uncomment the following formats once TRT used is up-to-date
    # if trt_format == trt.TensorFormat.CDHW32:
    #     return "CDHW32"
    # if trt_format == trt.TensorFormat.DHWC8:
    #     return "DHWC8"
    # if trt_format == trt.TensorFormat.HWC:
    #     return "HWC"
    if trt_format == trt.TensorFormat.CHW2:
        return "CHW2"
    if trt_format == trt.TensorFormat.CHW32:
        return "CHW32"
    if trt_format == trt.TensorFormat.LINEAR:
        return "LINEAR"
    if trt_format == trt.TensorFormat.CHW4:
        return "CHW4"
    if trt_format == trt.TensorFormat.HWC8:
        return "HWC8"
    if trt_format == trt.TensorFormat.CHW16:
        return "CHW16"
    return "INVALID"


def create_plan_dynamic_modelfile(models_dir,
                                  max_batch,
                                  model_version,
                                  input_shape,
                                  output0_shape,
                                  output1_shape,
                                  input_dtype,
                                  output0_dtype,
                                  output1_dtype,
                                  input_memory_format,
                                  output_memory_format,
                                  min_dim=1,
                                  max_dim=64):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)
    trt_input_memory_format = input_memory_format
    trt_output_memory_format = output_memory_format

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_batch == 0:
        input_with_batchsize = [i for i in input_shape]
    else:
        input_with_batchsize = [-1] + [i for i in input_shape]

    in0 = network.add_input("INPUT0", trt_input_dtype, input_with_batchsize)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_with_batchsize)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = network.add_identity(add.get_output(0))
    out1 = network.add_identity(sub.get_output(0))

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    out0.get_output(0).dtype = trt_output0_dtype
    out1.get_output(0).dtype = trt_output1_dtype

    in0.allowed_formats = 1 << int(trt_input_memory_format)
    in1.allowed_formats = 1 << int(trt_input_memory_format)
    out0.get_output(0).allowed_formats = 1 << int(trt_output_memory_format)
    out1.get_output(0).allowed_formats = 1 << int(trt_output_memory_format)

    if (trt_input_dtype == trt.int8):
        in0.dynamic_range = (-128.0, 127.0)
        in1.dynamic_range = (-128.0, 127.0)
    if (trt_output0_dtype == trt.int8):
        out0.get_output(0).dynamic_range = (-128.0, 127.0)
    if (trt_output1_dtype == trt.int8):
        out1.get_output(0).dynamic_range = (-128.0, 127.0)

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in input_shape:
        if i == -1:
            min_shape = min_shape + [min_dim]
            opt_shape = opt_shape + [int((max_dim + min_dim) / 2)]
            max_shape = max_shape + [max_dim]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT0", min_shape, opt_shape, max_shape)
    profile.set_shape("INPUT1", min_shape, opt_shape, max_shape)
    flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    datatype_set = set([trt_input_dtype, trt_output0_dtype, trt_output1_dtype])
    for dt in datatype_set:
        if (dt == trt.int8):
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif (dt == trt.float16):
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config = builder.create_builder_config()
    config.flags = flags
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 20
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    # Use a different model name for different kinds of models
    base_name = "plan_nobatch" if max_batch == 0 else "plan"
    base_name += "_" + trt_format_to_string(
        input_memory_format) + "_" + trt_format_to_string(output_memory_format)
    model_name = tu.get_model_name(base_name, input_dtype, output0_dtype,
                                   output1_dtype)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_fixed_modelfile(models_dir, max_batch, model_version,
                                input_shape, output0_shape, output1_shape,
                                input_dtype, output0_dtype, output1_dtype,
                                input_memory_format, output_memory_format):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)
    trt_input_memory_format = input_memory_format
    trt_output_memory_format = output_memory_format

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    in0 = network.add_input("INPUT0", trt_input_dtype, input_shape)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_shape)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = network.add_identity(add.get_output(0))
    out1 = network.add_identity(sub.get_output(0))

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    out0.get_output(0).dtype = trt_output0_dtype
    out1.get_output(0).dtype = trt_output1_dtype

    in0.allowed_formats = 1 << int(trt_input_memory_format)
    in1.allowed_formats = 1 << int(trt_input_memory_format)
    out0.get_output(0).allowed_formats = 1 << int(trt_output_memory_format)
    out1.get_output(0).allowed_formats = 1 << int(trt_output_memory_format)

    if (trt_input_dtype == trt.int8):
        in0.dynamic_range = (-128.0, 127.0)
        in1.dynamic_range = (-128.0, 127.0)
    if (trt_output0_dtype == trt.int8):
        out0.get_output(0).dynamic_range = (-128.0, 127.0)
    if (trt_output1_dtype == trt.int8):
        out1.get_output(0).dynamic_range = (-128.0, 127.0)

    flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    datatype_set = set([trt_input_dtype, trt_output0_dtype, trt_output1_dtype])
    for dt in datatype_set:
        if (dt == trt.int8):
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif (dt == trt.float16):
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config = builder.create_builder_config()
    config.flags = flags
    config.max_workspace_size = 1 << 20
    builder.max_batch_size = max(1, max_batch)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    base_name = "plan_nobatch" if max_batch == 0 else "plan"
    base_name += "_" + trt_format_to_string(
        input_memory_format) + "_" + trt_format_to_string(output_memory_format)
    model_name = tu.get_model_name(base_name, input_dtype, output0_dtype,
                                   output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_modelconfig(models_dir, max_batch, model_version, input_shape,
                            output0_shape, output1_shape, input_dtype,
                            output0_dtype, output1_dtype, input_memory_format,
                            output_memory_format, version_policy):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for different kinds of models
    base_name = "plan_nobatch" if max_batch == 0 else "plan"
    base_name += "_" + trt_format_to_string(
        input_memory_format) + "_" + trt_format_to_string(output_memory_format)
    model_name = tu.get_model_name(base_name, input_dtype, output0_dtype,
                                   output1_dtype)

    config_dir = models_dir + "/" + model_name
    if -1 in input_shape:
        profile_index = 0
        config = '''
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
   }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
instance_group [
  {{
      profile:"{}"
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape), profile_index)
    else:
        config = '''
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
   }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_plan_model(models_dir, max_batch, model_version, input_shape,
                      output0_shape, output1_shape, input_dtype, output0_dtype,
                      output1_dtype, input_memory_format, output_memory_format):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    create_plan_modelconfig(models_dir, max_batch, model_version, input_shape,
                            output0_shape, output1_shape, input_dtype,
                            output0_dtype, output1_dtype, input_memory_format,
                            output_memory_format, None)

    if (not tu.shape_is_fixed(input_shape) or
            not tu.shape_is_fixed(output0_shape) or
            not tu.shape_is_fixed(output1_shape)):
        create_plan_dynamic_modelfile(models_dir, max_batch, model_version,
                                      input_shape, output0_shape, output1_shape,
                                      input_dtype, output0_dtype, output1_dtype,
                                      input_memory_format, output_memory_format)
    else:
        create_plan_fixed_modelfile(models_dir, max_batch, model_version,
                                    input_shape, output0_shape, output1_shape,
                                    input_dtype, output0_dtype, output1_dtype,
                                    input_memory_format, output_memory_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    FLAGS, unparsed = parser.parse_known_args()

    # reformat-free input
    # Fixed shape
    create_plan_model(FLAGS.models_dir, 0, 1, (13, 2, 1), (13, 2, 1),
                      (13, 2, 1), np.float16, np.float16, np.float16,
                      trt.TensorFormat.CHW2, trt.TensorFormat.LINEAR)
    create_plan_model(FLAGS.models_dir, 8, 1, (13, 2, 1), (13, 2, 1),
                      (13, 2, 1), np.float16, np.float16, np.float16,
                      trt.TensorFormat.CHW2, trt.TensorFormat.LINEAR)

    # Dynamic shape
    create_plan_model(FLAGS.models_dir, 0, 1, (-1, 2, 1), (-1, 2, 1),
                      (-1, 2, 1), np.float32, np.float32, np.float32,
                      trt.TensorFormat.CHW32, trt.TensorFormat.LINEAR)
    create_plan_model(FLAGS.models_dir, 8, 1, (-1, 2, 1), (-1, 2, 1),
                      (-1, 2, 1), np.float32, np.float32, np.float32,
                      trt.TensorFormat.CHW32, trt.TensorFormat.LINEAR)

    # reformat-free output
    # reformat-free I/O
