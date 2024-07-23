#!/usr/bin/env python3

# Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from gen_common import np_to_model_dtype, np_to_onnx_dtype, np_to_trt_dtype

FLAGS = None
np_dtype_string = np.dtype(object)


def create_onnx_modelfile(models_dir, model_version, max_batch, dtype, shape):
    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    model_name = tu.get_dyna_sequence_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", dtype
    )
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    # Create the model. For now don't implement a proper accumulator
    # just return 0 if not-ready and 'INPUT'+'START'*('END'*'CORRID')
    # otherwise...  the tests know to expect this.
    onnx_dtype = np_to_onnx_dtype(dtype)
    onnx_input_shape, idx = tu.shape_to_onnx_shape(shape, 0)
    onnx_output_shape, idx = tu.shape_to_onnx_shape(shape, idx)

    # If the input is a string then use int32 for operation and just
    # cast to/from string for input and output.
    onnx_control_dtype = onnx_dtype
    if onnx_dtype == onnx.TensorProto.STRING:
        onnx_control_dtype = onnx.TensorProto.INT32

    # If input dtype is bool, then use bool type for control and
    # int32 type for input/output
    if onnx_dtype == onnx.TensorProto.BOOL:
        onnx_dtype = onnx.TensorProto.INT32

    batch_dim = [] if max_batch == 0 else [None]

    onnx_input = onnx.helper.make_tensor_value_info(
        "INPUT", onnx_dtype, batch_dim + onnx_input_shape
    )
    onnx_input_state = onnx.helper.make_tensor_value_info(
        "INPUT_STATE", onnx_dtype, batch_dim + onnx_input_shape
    )
    onnx_start = onnx.helper.make_tensor_value_info(
        "START", onnx_control_dtype, batch_dim + [1]
    )
    onnx_ready = onnx.helper.make_tensor_value_info(
        "READY", onnx_control_dtype, batch_dim + [1]
    )
    onnx_corrid = onnx.helper.make_tensor_value_info(
        "CORRID", onnx.TensorProto.UINT64, batch_dim + [1]
    )
    onnx_end = onnx.helper.make_tensor_value_info(
        "END", onnx_control_dtype, batch_dim + [1]
    )
    onnx_output = onnx.helper.make_tensor_value_info(
        "OUTPUT", onnx_dtype, batch_dim + onnx_output_shape
    )
    onnx_output_state = onnx.helper.make_tensor_value_info(
        "OUTPUT_STATE", onnx_dtype, batch_dim + onnx_output_shape
    )

    internal_input = onnx.helper.make_node("Identity", ["INPUT"], ["_INPUT"])
    internal_input_state = onnx.helper.make_node(
        "Identity", ["INPUT_STATE"], ["_INPUT_STATE"]
    )
    # cast int8, int16 input to higher precision int as Onnx Add/Sub operator doesn't support those type
    # Also casting String data type to int32
    if (
        (onnx_dtype == onnx.TensorProto.INT8)
        or (onnx_dtype == onnx.TensorProto.INT16)
        or (onnx_dtype == onnx.TensorProto.STRING)
    ):
        internal_input = onnx.helper.make_node(
            "Cast", ["INPUT"], ["_INPUT"], to=onnx.TensorProto.INT32
        )
        internal_input_state = onnx.helper.make_node(
            "Cast", ["INPUT_STATE"], ["_INPUT_STATE"], to=onnx.TensorProto.INT32
        )

    # Convert boolean value to int32 value
    if onnx_control_dtype == onnx.TensorProto.BOOL:
        internal_input1 = onnx.helper.make_node(
            "Cast", ["START"], ["_START"], to=onnx.TensorProto.INT32
        )
        internal_input2 = onnx.helper.make_node(
            "Cast", ["READY"], ["_READY"], to=onnx.TensorProto.INT32
        )
        not_start_cast = onnx.helper.make_node("Not", ["START"], ["_NOT_START_CAST"])
        not_start = onnx.helper.make_node(
            "Cast", ["_NOT_START_CAST"], ["_NOT_START"], to=onnx.TensorProto.INT32
        )
        not_ready_cast = onnx.helper.make_node("Not", ["START"], ["_NOT_READY_CAST"])
        not_ready = onnx.helper.make_node(
            "Cast", ["_NOT_READY_CAST"], ["_NOT_READY"], to=onnx.TensorProto.INT32
        )

        input_state_cond = onnx.helper.make_node(
            "And", ["READY", "_NOT_START_CAST"], ["input_state_cond"]
        )
        input_state_cond_cast = onnx.helper.make_node(
            "Cast",
            ["input_state_cond"],
            ["input_state_cond_cast"],
            to=onnx.TensorProto.INT32,
        )
        mul_state = onnx.helper.make_node(
            "Mul", ["_INPUT_STATE", "input_state_cond_cast"], ["mul_state"]
        )
        add = onnx.helper.make_node("Add", ["_INPUT", "mul_state"], ["CAST"])

    else:
        start_cast = onnx.helper.make_node(
            "Cast", ["START"], ["_START_CAST"], to=onnx.TensorProto.BOOL
        )
        not_start_cast = onnx.helper.make_node(
            "Not", ["_START_CAST"], ["_NOT_START_CAST"]
        )
        not_start = onnx.helper.make_node(
            "Cast", ["_NOT_START_CAST"], ["_NOT_START"], to=onnx.TensorProto.INT32
        )

        ready_cast = onnx.helper.make_node(
            "Cast", ["READY"], ["_READY_CAST"], to=onnx.TensorProto.BOOL
        )
        not_ready_cast = onnx.helper.make_node(
            "Not", ["_READY_CAST"], ["_NOT_READY_CAST"]
        )
        not_ready = onnx.helper.make_node(
            "Cast", ["_NOT_READY_CAST"], ["_NOT_READY"], to=onnx.TensorProto.INT32
        )

        # Take advantage of knowledge that the READY false value is 0 and true is 1
        input_state_cond = onnx.helper.make_node(
            "And", ["_NOT_START_CAST", "_READY_CAST"], ["input_state_cond"]
        )
        input_state_cond_cast = onnx.helper.make_node(
            "Cast",
            ["input_state_cond"],
            ["input_state_cond_cast"],
            to=onnx.TensorProto.INT32,
        )
        mul_state = onnx.helper.make_node(
            "Mul", ["_INPUT_STATE", "input_state_cond_cast"], ["mul_state"]
        )
        add = onnx.helper.make_node("Add", ["_INPUT", "mul_state"], ["CAST"])

    cast = onnx.helper.make_node("Cast", ["CAST"], ["OUTPUT"], to=onnx_dtype)
    cast_output_state = onnx.helper.make_node(
        "Cast", ["CAST"], ["OUTPUT_STATE"], to=onnx_dtype
    )

    # Avoid cast from float16 to float16
    # (bug in Onnx Runtime, cast from float16 to float16 will become cast from float16 to float32)
    if onnx_dtype == onnx.TensorProto.FLOAT16:
        cast = onnx.helper.make_node("Identity", ["CAST"], ["OUTPUT"])
        cast_output_state = onnx.helper.make_node(
            "Identity", ["CAST"], ["OUTPUT_STATE"]
        )

    if onnx_control_dtype == onnx.TensorProto.BOOL:
        onnx_nodes = [
            internal_input,
            internal_input_state,
            internal_input1,
            internal_input2,
            not_start_cast,
            not_start,
            not_ready_cast,
            not_ready,
            input_state_cond,
            input_state_cond_cast,
            mul_state,
            add,
            cast,
            cast_output_state,
        ]
    else:
        onnx_nodes = [
            internal_input,
            internal_input_state,
            start_cast,
            not_start_cast,
            not_start,
            ready_cast,
            not_ready_cast,
            not_ready,
            input_state_cond,
            input_state_cond_cast,
            mul_state,
            add,
            cast,
            cast_output_state,
        ]

    onnx_inputs = [
        onnx_end,
        onnx_corrid,
        onnx_input_state,
        onnx_input,
        onnx_start,
        onnx_ready,
    ]
    onnx_outputs = [onnx_output, onnx_output_state]
    graph_proto = onnx.helper.make_graph(
        onnx_nodes, model_name, onnx_inputs, onnx_outputs
    )

    if FLAGS.onnx_opset > 0:
        model_opset = onnx.helper.make_operatorsetid("", FLAGS.onnx_opset)
        model_def = onnx.helper.make_model(
            graph_proto, producer_name="triton", opset_imports=[model_opset]
        )
    else:
        model_def = onnx.helper.make_model(graph_proto, producer_name="triton")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_onnx_modelconfig(models_dir, model_version, max_batch, dtype, shape):
    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    model_name = tu.get_dyna_sequence_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", dtype
    )
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "onnxruntime_onnx"
max_batch_size: {}
sequence_batching {{
  max_sequence_idle_microseconds: 5000000
  {}
  control_input [
    {{
      name: "START"
      control [
        {{
          kind: CONTROL_SEQUENCE_START
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "END"
      control [
        {{
          kind: CONTROL_SEQUENCE_END
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "READY"
      control [
        {{
          kind: CONTROL_SEQUENCE_READY
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "CORRID"
      control [
        {{
          kind: CONTROL_SEQUENCE_CORRID
          data_type: TYPE_UINT64
        }}
      ]
    }}
  ]
  state [
    {{
      input_name: "INPUT_STATE"
      output_name: "OUTPUT_STATE"
      data_type: {dtype}
      dims: {dims}
    }}
  ]
}}
input [
  {{
    name: "INPUT"
    data_type: {dtype}
    dims: [ {dims} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {dtype}
    dims: [ {dims} ]
  }}
]
instance_group [
  {{
    kind: KIND_CPU
  }}
]
""".format(
        model_name,
        max_batch,
        (
            "oldest { max_candidate_sequences: 6\npreferred_batch_size: [ 4 ]\nmax_queue_delay_microseconds: 0\n}"
            if max_batch > 0
            else ""
        ),
        dtype=np_to_model_dtype(dtype),
        dims=tu.shape_to_dims_str(shape),
        type="fp32" if dtype == np.float32 else "int32",
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_plan_modelfile(models_dir, model_version, max_batch, dtype, shape):
    trt_dtype = np_to_trt_dtype(dtype)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    unit_shape = [1] * len(shape)
    if max_batch != 0:
        in0 = network.add_input("INPUT", trt_dtype, [-1] + shape)
        in_state0 = network.add_input("INPUT_STATE", trt_dtype, [-1] + shape)
        start0 = network.add_input("START", trt_dtype, [-1] + unit_shape)
        network.add_input("END", trt_dtype, [-1] + unit_shape)
        ready0 = network.add_input("READY", trt_dtype, [-1] + unit_shape)
        network.add_input("CORRID", trt.int32, [-1] + unit_shape)
        constant_1_data = trt.Weights(np.ones(unit_shape + [1], dtype=dtype))
        constant_1 = network.add_constant(unit_shape + [1], constant_1_data)
    else:
        in0 = network.add_input("INPUT", trt_dtype, shape)
        in_state0 = network.add_input("INPUT_STATE", trt_dtype, shape)
        start0 = network.add_input("START", trt_dtype, unit_shape)
        network.add_input("END", trt_dtype, unit_shape)
        ready0 = network.add_input("READY", trt_dtype, unit_shape)
        network.add_input("CORRID", trt.int32, unit_shape)
        constant_1_data = trt.Weights(np.ones(unit_shape, dtype=dtype))
        constant_1 = network.add_constant(unit_shape, constant_1_data)

    not_start = network.add_elementwise(
        constant_1.get_output(0), start0, trt.ElementWiseOperation.SUB
    )
    not_start.set_output_type(0, trt_dtype)

    input_state_cond_temp = network.add_elementwise(
        ready0, not_start.get_output(0), trt.ElementWiseOperation.SUM
    )
    constant_2 = network.add_elementwise(
        constant_1.get_output(0), constant_1.get_output(0), trt.ElementWiseOperation.SUM
    )
    input_state_cond = network.add_elementwise(
        input_state_cond_temp.get_output(0),
        constant_2.get_output(0),
        trt.ElementWiseOperation.FLOOR_DIV,
    )
    internal_state = network.add_elementwise(
        in_state0, input_state_cond.get_output(0), trt.ElementWiseOperation.PROD
    )
    out0 = network.add_elementwise(
        internal_state.get_output(0), in0, trt.ElementWiseOperation.SUM
    )
    out0_state = network.add_elementwise(
        internal_state.get_output(0), in0, trt.ElementWiseOperation.SUM
    )

    out0.get_output(0).name = "OUTPUT"
    network.mark_output(out0.get_output(0))

    out0_state.get_output(0).name = "OUTPUT_STATE"
    network.mark_output(out0_state.get_output(0))

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in shape:
        if i == -1:
            min_shape = min_shape + [1]
            opt_shape = opt_shape + [8]
            max_shape = max_shape + [32]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT", min_shape, opt_shape, max_shape)
    profile.set_shape("INPUT_STATE", min_shape, opt_shape, max_shape)
    if max_batch != 0:
        profile.set_shape(
            "START",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
        profile.set_shape(
            "END", [1] + unit_shape, [max_batch] + unit_shape, [max_batch] + unit_shape
        )
        profile.set_shape(
            "READY",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
        profile.set_shape(
            "CORRID",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
    else:
        profile.set_shape("START", unit_shape, unit_shape, unit_shape)
        profile.set_shape("END", unit_shape, unit_shape, unit_shape)
        profile.set_shape("READY", unit_shape, unit_shape, unit_shape)
        profile.set_shape("CORRID", unit_shape, unit_shape, unit_shape)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine
    del network

    model_name = tu.get_dyna_sequence_model_name(
        "plan_nobatch" if max_batch == 0 else "plan", dtype
    )
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_rf_modelfile(models_dir, model_version, max_batch, dtype, shape):
    trt_dtype = np_to_trt_dtype(dtype)
    trt_memory_format = trt.TensorFormat.LINEAR

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    unit_shape = [1] * len(shape)
    if max_batch != 0:
        in0 = network.add_input("INPUT", trt_dtype, [-1] + shape)
        in_state0 = network.add_input("INPUT_STATE", trt_dtype, [-1] + shape)
        start0 = network.add_input("START", trt_dtype, [-1] + unit_shape)
        network.add_input("END", trt_dtype, [-1] + unit_shape)
        ready0 = network.add_input("READY", trt_dtype, [-1] + unit_shape)
        network.add_input("CORRID", trt.int32, [-1] + unit_shape)
        constant_1_data = trt.Weights(np.ones(unit_shape + [1], dtype=dtype))
        constant_1 = network.add_constant(unit_shape + [1], constant_1_data)
    else:
        in0 = network.add_input("INPUT", trt_dtype, shape)
        in_state0 = network.add_input("INPUT_STATE", trt_dtype, shape)
        start0 = network.add_input("START", trt_dtype, unit_shape)
        network.add_input("END", trt_dtype, unit_shape)
        ready0 = network.add_input("READY", trt_dtype, unit_shape)
        network.add_input("CORRID", trt.int32, unit_shape)
        constant_1_data = trt.Weights(np.ones(unit_shape, dtype=dtype))
        constant_1 = network.add_constant(unit_shape, constant_1_data)

    not_start = network.add_elementwise(
        constant_1.get_output(0), start0, trt.ElementWiseOperation.SUB
    )
    not_start.set_output_type(0, trt_dtype)

    input_state_cond_temp = network.add_elementwise(
        ready0, not_start.get_output(0), trt.ElementWiseOperation.SUM
    )
    constant_2 = network.add_elementwise(
        constant_1.get_output(0), constant_1.get_output(0), trt.ElementWiseOperation.SUM
    )
    input_state_cond = network.add_elementwise(
        input_state_cond_temp.get_output(0),
        constant_2.get_output(0),
        trt.ElementWiseOperation.FLOOR_DIV,
    )
    internal_state = network.add_elementwise(
        in_state0, input_state_cond.get_output(0), trt.ElementWiseOperation.PROD
    )
    out0 = network.add_elementwise(
        internal_state.get_output(0), in0, trt.ElementWiseOperation.SUM
    )
    out0_state = network.add_elementwise(
        internal_state.get_output(0), in0, trt.ElementWiseOperation.SUM
    )

    out0.get_output(0).name = "OUTPUT"
    network.mark_output(out0.get_output(0))
    out0.get_output(0).dtype = trt_dtype

    out0_state.get_output(0).name = "OUTPUT_STATE"
    network.mark_output(out0_state.get_output(0))
    out0_state.get_output(0).dtype = trt_dtype

    in0.allowed_formats = 1 << int(trt_memory_format)
    in_state0.allowed_formats = 1 << int(trt_memory_format)
    start0.allowed_formats = 1 << int(trt_memory_format)
    ready0.allowed_formats = 1 << int(trt_memory_format)
    out0.get_output(0).allowed_formats = 1 << int(trt_memory_format)
    out0_state.get_output(0).allowed_formats = 1 << int(trt_memory_format)

    if trt_dtype == trt.int8:
        in0.dynamic_range = (-128.0, 127.0)
        in_state0.dynamic_range = (-128.0, 127.0)
        out0.dynamic_range = (-128.0, 127.0)
        out0_state.dynamic_range = (-128.0, 127.0)
        start0.dynamic_range = (-128.0, 127.0)
        ready0.dynamic_range = (-128.0, 127.0)

    flags = 1 << int(trt.BuilderFlag.DIRECT_IO)
    flags |= 1 << int(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    flags |= 1 << int(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

    if trt_dtype == trt.int8:
        flags |= 1 << int(trt.BuilderFlag.INT8)
    elif trt_dtype == trt.float16:
        flags |= 1 << int(trt.BuilderFlag.FP16)

    config = builder.create_builder_config()
    config.flags = flags

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in shape:
        if i == -1:
            min_shape = min_shape + [1]
            opt_shape = opt_shape + [8]
            max_shape = max_shape + [32]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT", min_shape, opt_shape, max_shape)
    profile.set_shape("INPUT_STATE", min_shape, opt_shape, max_shape)
    if max_batch != 0:
        profile.set_shape(
            "START",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
        profile.set_shape(
            "END", [1] + unit_shape, [max_batch] + unit_shape, [max_batch] + unit_shape
        )
        profile.set_shape(
            "READY",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
        profile.set_shape(
            "CORRID",
            [1] + unit_shape,
            [max_batch] + unit_shape,
            [max_batch] + unit_shape,
        )
    else:
        profile.set_shape("START", unit_shape, unit_shape, unit_shape)
        profile.set_shape("END", unit_shape, unit_shape, unit_shape)
        profile.set_shape("READY", unit_shape, unit_shape, unit_shape)
        profile.set_shape("CORRID", unit_shape, unit_shape, unit_shape)

    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name = tu.get_dyna_sequence_model_name(
        "plan_nobatch" if max_batch == 0 else "plan", dtype
    )
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_models(models_dir, model_version, max_batch, dtype, shape):
    if not tu.validate_for_trt_model(dtype, dtype, dtype, shape, shape, shape):
        return

    if dtype != np.float32:
        create_plan_rf_modelfile(models_dir, model_version, max_batch, dtype, shape)
    else:
        create_plan_modelfile(models_dir, model_version, max_batch, dtype, shape)


def create_plan_modelconfig(models_dir, model_version, max_batch, dtype, shape):
    if not tu.validate_for_trt_model(dtype, dtype, dtype, shape, shape, shape):
        return

    model_name = tu.get_dyna_sequence_model_name(
        "plan_nobatch" if max_batch == 0 else "plan", dtype
    )
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
sequence_batching {{
  max_sequence_idle_microseconds: 5000000
  {}
  control_input [
    {{
      name: "START"
      control [
        {{
          kind: CONTROL_SEQUENCE_START
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "END"
      control [
        {{
          kind: CONTROL_SEQUENCE_END
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "READY"
      control [
        {{
          kind: CONTROL_SEQUENCE_READY
          {type}_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "CORRID"
      control [
        {{
          kind: CONTROL_SEQUENCE_CORRID
          data_type: TYPE_INT32
        }}
      ]
    }}
  ]
  state [
    {{
      input_name: "INPUT_STATE"
      output_name: "OUTPUT_STATE"
      data_type: {dtype}
      dims: {dims}
    }}
  ]
}}
input [
  {{
    name: "INPUT"
    data_type: {dtype}
    dims: [ {dims} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {dtype}
    dims: [ {dims} ]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
  }}
]
""".format(
        model_name,
        max_batch,
        (
            "oldest { max_candidate_sequences: 6\npreferred_batch_size: [ 4 ]\nmax_queue_delay_microseconds: 0\n}"
            if max_batch > 0
            else ""
        ),
        dtype=np_to_model_dtype(dtype),
        dims=tu.shape_to_dims_str(shape),
        type="fp32" if dtype == np.float32 else "int32",
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_models(models_dir, dtype, shape, no_batch=True):
    model_version = 1

    if FLAGS.onnx:
        create_onnx_modelconfig(models_dir, model_version, 8, dtype, shape)
        create_onnx_modelfile(models_dir, model_version, 8, dtype, shape)
        if no_batch:
            create_onnx_modelconfig(models_dir, model_version, 0, dtype, shape)
            create_onnx_modelfile(models_dir, model_version, 0, dtype, shape)

    if FLAGS.tensorrt:
        if dtype == bool:
            return

        create_plan_modelconfig(models_dir, model_version, 8, dtype, shape)
        create_plan_models(models_dir, model_version, 8, dtype, shape)
        if no_batch:
            create_plan_modelconfig(models_dir, model_version, 0, dtype, shape)
            create_plan_models(models_dir, model_version, 0, dtype, shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    parser.add_argument(
        "--graphdef",
        required=False,
        action="store_true",
        help="Generate GraphDef models",
    )
    parser.add_argument(
        "--savedmodel",
        required=False,
        action="store_true",
        help="Generate SavedModel models",
    )
    parser.add_argument(
        "--tensorrt",
        required=False,
        action="store_true",
        help="Generate TensorRT PLAN models",
    )
    parser.add_argument(
        "--tensorrt-shape-io",
        required=False,
        action="store_true",
        help="Generate TensorRT PLAN models w/ shape tensor i/o",
    )
    parser.add_argument(
        "--onnx", required=False, action="store_true", help="Generate Onnx models"
    )
    parser.add_argument(
        "--onnx_opset",
        type=int,
        required=False,
        default=0,
        help="Opset used for Onnx models. Default is to use ONNXRT default",
    )
    parser.add_argument(
        "--libtorch",
        required=False,
        action="store_true",
        help="Generate Pytorch LibTorch models",
    )
    parser.add_argument(
        "--openvino",
        required=False,
        action="store_true",
        help="Generate OpenVino models",
    )
    parser.add_argument(
        "--variable",
        required=False,
        action="store_true",
        help="Used variable-shape tensors for input/output",
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.onnx:
        import onnx

    if FLAGS.tensorrt:
        import tensorrt as trt

    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_models(
            FLAGS.models_dir,
            np.int32,
            [
                1,
            ],
        )

    # Tests with models that accept variable-shape input/output tensors
    if FLAGS.variable:
        create_models(
            FLAGS.models_dir,
            np.int32,
            [
                -1,
            ],
            False,
        )
