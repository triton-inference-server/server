#!/usr/bin/env python3

# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from builtins import range

import gen_ensemble_model_utils as emu
import numpy as np
from gen_common import (
    np_to_model_dtype,
    np_to_onnx_dtype,
    np_to_tf_dtype,
    np_to_trt_dtype,
    openvino_save_model,
)

FLAGS = None
np_dtype_string = np.dtype(object)
from typing import List, Tuple


def create_tf_modelfile(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    tf_dtype = np_to_tf_dtype(dtype)

    # Create the model that copies inputs to corresponding outputs.
    tf.compat.v1.reset_default_graph()
    for io_num in range(io_cnt):
        input_name = "INPUT{}".format(io_num)
        output_name = "OUTPUT{}".format(io_num)
        if max_batch == 0:
            tin = tf.compat.v1.placeholder(
                tf_dtype, tu.shape_to_tf_shape(shape), input_name
            )
        else:
            tin = tf.compat.v1.placeholder(
                tf_dtype,
                [
                    None,
                ]
                + tu.shape_to_tf_shape(shape),
                input_name,
            )
        toutput = tf.identity(tin, name=output_name)

    # Use model name based on io_cnt and non-batching variant
    if create_savedmodel:
        model_name = tu.get_zero_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", io_cnt, dtype
        )
    else:
        model_name = tu.get_zero_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", io_cnt, dtype
        )

    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    if create_savedmodel:
        with tf.compat.v1.Session() as sess:
            input_dict = {}
            output_dict = {}
            for io_num in range(io_cnt):
                input_name = "INPUT{}".format(io_num)
                output_name = "OUTPUT{}".format(io_num)
                input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    input_name + ":0"
                )
                output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    output_name + ":0"
                )
                input_dict[input_name] = input_tensor
                output_dict[output_name] = output_tensor
            tf.compat.v1.saved_model.simple_save(
                sess,
                model_version_dir + "/model.savedmodel",
                inputs=input_dict,
                outputs=output_dict,
            )
    else:
        with tf.compat.v1.Session() as sess:
            graph_io.write_graph(
                sess.graph.as_graph_def(),
                model_version_dir,
                "model.graphdef",
                as_text=False,
            )


def create_tf_modelconfig(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    shape_str = tu.shape_to_dims_str(shape)

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_zero_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", io_cnt, dtype
        )
    else:
        model_name = tu.get_zero_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", io_cnt, dtype
        )

    config_dir = os.path.join(models_dir, model_name)
    config = """
name: "{}"
platform: "{}"
max_batch_size: {}
""".format(
        model_name,
        "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef",
        max_batch,
    )

    for io_num in range(io_cnt):
        config += """
input [
  {{
    name: "INPUT{}"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT{}"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
        )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_ensemble_modelfile(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_ensemble_model(
        "zero", dtype, dtype, dtype, shape, shape, shape
    ):
        return

    emu.create_identity_ensemble_modelfile(
        "zero",
        models_dir,
        model_version,
        max_batch,
        dtype,
        [shape] * io_cnt,
        [shape] * io_cnt,
    )


def create_ensemble_modelconfig(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_ensemble_model(
        "zero", dtype, dtype, dtype, shape, shape, shape
    ):
        return

    emu.create_identity_ensemble_modelconfig(
        "zero",
        models_dir,
        model_version,
        max_batch,
        dtype,
        [shape] * io_cnt,
        [shape] * io_cnt,
        [shape] * io_cnt,
        [shape] * io_cnt,
    )


def create_onnx_modelfile(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    onnx_dtype = np_to_onnx_dtype(dtype)

    # Create the model
    model_name = tu.get_zero_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", io_cnt, dtype
    )
    model_version_dir = os.path.join(models_dir, model_name, str(model_version))

    batch_dim = [] if max_batch == 0 else [None]

    onnx_nodes = []
    onnx_inputs = []
    onnx_outputs = []
    idx = 0
    for io_num in range(io_cnt):
        # Repeat so that the variable dimension name is different
        in_shape, idx = tu.shape_to_onnx_shape(shape, idx)
        out_shape, idx = tu.shape_to_onnx_shape(shape, idx)
        in_name = "INPUT{}".format(io_num)
        out_name = "OUTPUT{}".format(io_num)

        onnx_inputs.append(
            onnx.helper.make_tensor_value_info(
                in_name, onnx_dtype, batch_dim + in_shape
            )
        )
        onnx_outputs.append(
            onnx.helper.make_tensor_value_info(
                out_name, onnx_dtype, batch_dim + out_shape
            )
        )
        onnx_nodes.append(onnx.helper.make_node("Identity", [in_name], [out_name]))

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

    os.makedirs(model_version_dir, exist_ok=True)

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_onnx_modelconfig(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # Use a different model name for the non-batching variant
    model_name = tu.get_zero_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", io_cnt, dtype
    )
    config_dir = os.path.join(models_dir, model_name)

    config = emu.create_general_modelconfig(
        model_name,
        "onnxruntime_onnx",
        max_batch,
        emu.repeat(dtype, io_cnt),
        emu.repeat(shape, io_cnt),
        emu.repeat(shape, io_cnt),
        emu.repeat(dtype, io_cnt),
        emu.repeat(shape, io_cnt),
        emu.repeat(shape, io_cnt),
        emu.repeat(None, io_cnt),
        force_tensor_number_suffix=True,
    )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_libtorch_modelfile(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_libtorch_model(
        dtype, dtype, dtype, shape, shape, shape, max_batch
    ):
        return

    model_name = tu.get_zero_model_name(
        "libtorch_nobatch" if max_batch == 0 else "libtorch", io_cnt, dtype
    )

    # Create the model
    if io_cnt == 1:
        if dtype == np_dtype_string:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(self, input0: List[str]) -> List[str]:
                    return input0

        else:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(self, input0):
                    return input0

    elif io_cnt == 2:
        if dtype == np_dtype_string:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(
                    self, input0: List[str], input1: List[str]
                ) -> Tuple[List[str], List[str]]:
                    return input0, input1

        else:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(self, input0, input1):
                    return input0, input1

    elif io_cnt == 3:
        if dtype == np_dtype_string:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(
                    self, input0: List[str], input1: List[str], input2: List[str]
                ) -> Tuple[List[str], List[str], List[str]]:
                    return input0, input1, input2

        else:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(self, input0, input1, input2):
                    return input0, input1, input2

    elif io_cnt == 4:
        if dtype == np_dtype_string:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(
                    self,
                    input0: List[str],
                    input1: List[str],
                    input2: List[str],
                    input3: List[str],
                ) -> Tuple[List[str], List[str], List[str], List[str]]:
                    return input0, input1, input2, input3

        else:

            class IdentityNet(nn.Module):
                def __init__(self):
                    super(IdentityNet, self).__init__()

                def forward(self, input0, input1, input2, input3):
                    return input0, input1, input2, input3

    identityModel = IdentityNet()
    traced = torch.jit.script(identityModel)

    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    traced.save(model_version_dir + "/model.pt")


def create_libtorch_modelconfig(
    create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    if not tu.validate_for_libtorch_model(
        dtype, dtype, dtype, shape, shape, shape, max_batch
    ):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_zero_model_name(
        "libtorch_nobatch" if max_batch == 0 else "libtorch", io_cnt, dtype
    )
    shape_str = tu.shape_to_dims_str(shape)

    config_dir = os.path.join(models_dir, model_name)
    config = """
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: {}
version_policy: {}
""".format(
        model_name, max_batch, version_policy_str
    )

    for io_num in range(io_cnt):
        config += """
input [
  {{
    name: "INPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
        )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_libtorch_linalg_modelfile(create_savedmodel, models_dir, model_version):
    model_name = "libtorch_float32_linalg"

    # To test the linalg library, this script uses two inverse matrix operations
    # to return the original input.
    class IdentityNet(nn.Module):
        def __init__(self, ref_pts):
            super(IdentityNet, self).__init__()
            ref_pts = torch.as_tensor(ref_pts)
            self.register_buffer("ref_pts", ref_pts)

        def forward(self, src: torch.Tensor):
            X = torch.linalg.tensorsolve(self.ref_pts, src)
            Y = torch.tensordot(self.ref_pts, X, dims=X.ndim)
            return Y

    ref_pts = torch.eye(2 * 3 * 4).reshape(2 * 3, 4, 2, 3, 4)
    identityModel = IdentityNet(ref_pts)
    traced = torch.jit.script(identityModel)

    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    traced.save(model_version_dir + "/model.pt")


def create_libtorch_linalg_modelconfig(create_savedmodel, models_dir, model_version):
    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"

    model_name = "libtorch_float32_linalg"
    dtype = np.float32
    io_cnt = 1
    max_batch = 0
    shape = [6, 4]
    shape_str = tu.shape_to_dims_str(shape)

    config_dir = os.path.join(models_dir, model_name)
    config = """
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: {}
version_policy: {}
""".format(
        model_name, max_batch, version_policy_str
    )

    for io_num in range(io_cnt):
        config += """
input [
  {{
    name: "INPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
        )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_openvino_modelfile(
    models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    batch_dim = (
        []
        if max_batch == 0
        else [
            max_batch,
        ]
    )
    if not tu.validate_for_openvino_model(
        dtype, dtype, dtype, batch_dim + shape, batch_dim + shape, batch_dim + shape
    ):
        return

    # Create the model
    model_name = tu.get_zero_model_name(
        "openvino_nobatch" if max_batch == 0 else "openvino", io_cnt, dtype
    )
    model_version_dir = os.path.join(models_dir, model_name, str(model_version))

    openvino_inputs = []
    openvino_outputs = []
    for io_num in range(io_cnt):
        in_name = "INPUT{}".format(io_num)
        out_name = "OUTPUT{}".format(io_num)
        openvino_inputs.append(
            ov.opset1.parameter(shape=batch_dim + shape, dtype=dtype, name=in_name)
        )
        openvino_outputs.append(
            ov.opset1.result(openvino_inputs[io_num], name=out_name)
        )

    model = ov.Model(openvino_outputs, openvino_inputs, model_name)
    openvino_save_model(model_version_dir, model)


def create_openvino_modelconfig(
    models_dir, model_version, io_cnt, max_batch, dtype, shape
):
    batch_dim = (
        []
        if max_batch == 0
        else [
            max_batch,
        ]
    )
    if not tu.validate_for_openvino_model(
        dtype, dtype, dtype, batch_dim + shape, batch_dim + shape, batch_dim + shape
    ):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_zero_model_name(
        "openvino_nobatch" if max_batch == 0 else "openvino", io_cnt, dtype
    )
    shape_str = tu.shape_to_dims_str(shape)

    config_dir = os.path.join(models_dir, model_name)
    config = """
name: "{}"
backend: "openvino"
max_batch_size: {}
version_policy: {}
""".format(
        model_name, max_batch, version_policy_str
    )

    for io_num in range(io_cnt):
        config += """
input [
  {{
    name: "INPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT__{}"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
            io_num,
            np_to_model_dtype(dtype),
            shape_str,
        )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_plan_modelfile(
    create_savedmodel,
    models_dir,
    model_version,
    io_cnt,
    max_batch,
    dtype,
    shape,
    profile_max_size,
):
    if not tu.validate_for_trt_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # generate models with different configuration to ensure test coverage
    if dtype != np.float32:
        create_plan_dynamic_rf_modelfile(
            models_dir, model_version, io_cnt, max_batch, dtype, shape, profile_max_size
        )
    else:
        create_plan_dynamic_modelfile(
            models_dir, model_version, io_cnt, max_batch, dtype, shape, profile_max_size
        )


def create_plan_dynamic_rf_modelfile(
    models_dir, model_version, io_cnt, max_batch, dtype, shape, profile_max_size
):
    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    if max_batch == 0:
        shape_with_batchsize = [i for i in shape]
    else:
        shape_with_batchsize = [-1] + [i for i in shape]

    trt_dtype = np_to_trt_dtype(dtype)
    trt_memory_format = trt.TensorFormat.LINEAR
    for io_num in range(io_cnt):
        in_node = network.add_input(
            "INPUT{}".format(io_num), trt_dtype, shape_with_batchsize
        )
        in_node.allowed_formats = 1 << int(trt_memory_format)

        out_node = network.add_identity(in_node)

        out_node.get_output(0).name = "OUTPUT{}".format(io_num)
        out_node.get_output(0).dtype = trt_dtype
        network.mark_output(out_node.get_output(0))
        out_node.get_output(0).allowed_formats = 1 << int(trt_memory_format)

        if trt_dtype == trt.int8:
            in_node.dynamic_range = (-128.0, 127.0)
            out_node.get_output(0).dynamic_range = (-128.0, 127.0)

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in shape:
        if i == -1:
            # Generating a very generous optimization profile
            min_shape = min_shape + [1]
            opt_shape = opt_shape + [8]
            max_shape = max_shape + [profile_max_size]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    for io_num in range(io_cnt):
        profile.set_shape("INPUT{}".format(io_num), min_shape, opt_shape, max_shape)

    flags = 1 << int(trt.BuilderFlag.DIRECT_IO)
    flags |= 1 << int(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    flags |= 1 << int(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    datatype_set = set([trt_dtype])
    for dt in datatype_set:
        if dt == trt.int8:
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif dt == trt.float16:
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config = builder.create_builder_config()
    config.flags = flags
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    config.add_optimization_profile(profile)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name = tu.get_zero_model_name(
        "plan_nobatch" if max_batch == 0 else "plan", io_cnt, dtype
    )
    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_shape_tensor_modelfile(
    models_dir,
    model_version,
    io_cnt,
    max_batch,
    dtype,
    shape,
    profile_max_size,
    shape_tensor_input_dtype,
):
    # Note that resize layer does not support int tensors.
    # The model takes two inputs (INPUT and DUMMY_INPUT)
    # and produce two outputs.
    # OUTPUT : The shape of resized output 'DUMMY_OUTPUT'.
    # DUMMY_OUTPUT : Obtained after resizing 'DUMMY_INPUT'
    # to shape specified in 'INPUT'.
    # Note that values of OUTPUT tensor must be identical
    # to INPUT values

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    if max_batch == 0:
        shape_with_batchsize = len(shape)
        dummy_shape = [-1] * shape_with_batchsize
    else:
        shape_with_batchsize = len(shape) + 1
        dummy_shape = [-1] * shape_with_batchsize

    trt_dtype = np_to_trt_dtype(dtype)
    trt_shape_dtype = np_to_trt_dtype(shape_tensor_input_dtype)
    trt_memory_format = trt.TensorFormat.LINEAR
    for io_num in range(io_cnt):
        in_node = network.add_input(
            "INPUT{}".format(io_num), trt_shape_dtype, [shape_with_batchsize]
        )
        in_node.allowed_formats = 1 << int(trt_memory_format)
        dummy_in_node = network.add_input(
            "DUMMY_INPUT{}".format(io_num), trt_dtype, dummy_shape
        )
        dummy_in_node.allowed_formats = 1 << int(trt_memory_format)
        resize_layer = network.add_resize(dummy_in_node)
        resize_layer.set_input(1, in_node)
        out_node = network.add_shape(resize_layer.get_output(0))

        dummy_out_node = resize_layer.get_output(0)
        out_node.get_output(0).name = "OUTPUT{}".format(io_num)

        dummy_out_node.name = "DUMMY_OUTPUT{}".format(io_num)

        dummy_out_node.dtype = trt_dtype
        network.mark_output(dummy_out_node)
        dummy_out_node.allowed_formats = 1 << int(trt_memory_format)

        out_node.get_output(0).dtype = trt.int64
        network.mark_output_for_shapes(out_node.get_output(0))
        out_node.get_output(0).allowed_formats = 1 << int(trt_memory_format)

        if trt_dtype == trt.int8:
            in_node.dynamic_range = (-128.0, 127.0)
            out_node.get_output(0).dynamic_range = (-128.0, 127.0)

    config = builder.create_builder_config()

    min_prefix = []
    opt_prefix = []
    max_prefix = []

    if max_batch != 0:
        min_prefix = [1]
        opt_prefix = [max(1, max_batch)]
        max_prefix = [max(1, max_batch)]

    min_shape = min_prefix + [1] * len(shape)
    opt_shape = opt_prefix + [8] * len(shape)
    max_shape = max_prefix + [profile_max_size] * len(shape)

    profile = builder.create_optimization_profile()
    for io_num in range(io_cnt):
        profile.set_shape_input(
            "INPUT{}".format(io_num), min_shape, opt_shape, max_shape
        )
        profile.set_shape(
            "DUMMY_INPUT{}".format(io_num), min_shape, opt_shape, max_shape
        )

    config.add_optimization_profile(profile)

    flags = 1 << int(trt.BuilderFlag.DIRECT_IO)
    flags |= 1 << int(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    flags |= 1 << int(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    datatype_set = set([trt_dtype])
    for dt in datatype_set:
        if dt == trt.int8:
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif dt == trt.float16:
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config.flags = flags

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name = tu.get_zero_model_name(
        "plan_nobatch" if max_batch == 0 else "plan", io_cnt, dtype
    )
    model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_dynamic_modelfile(
    models_dir, model_version, io_cnt, max_batch, dtype, shape, profile_max_size
):
    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    if max_batch == 0:
        shape_with_batchsize = [i for i in shape]
    else:
        shape_with_batchsize = [-1] + [i for i in shape]

    trt_dtype = np_to_trt_dtype(dtype)
    for io_num in range(io_cnt):
        in_node = network.add_input(
            "INPUT{}".format(io_num), trt_dtype, shape_with_batchsize
        )
        out_node = network.add_identity(in_node)
        out_node.get_output(0).name = "OUTPUT{}".format(io_num)
        network.mark_output(out_node.get_output(0))

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in shape:
        if i == -1:
            # Generating a very generous optimization profile
            min_shape = min_shape + [1]
            opt_shape = opt_shape + [8]
            max_shape = max_shape + [profile_max_size]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    for io_num in range(io_cnt):
        profile.set_shape("INPUT{}".format(io_num), min_shape, opt_shape, max_shape)
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    if FLAGS.tensorrt_compat:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name_base = "plan"
    if max_batch == 0:
        model_name_base += "_nobatch"
    if FLAGS.tensorrt_compat:
        model_name_base += "_compatible"

    model_name = tu.get_zero_model_name(model_name_base, io_cnt, dtype)
    model_version_dir = os.path.join(models_dir, model_name, str(model_version))
    os.makedirs(model_version_dir, exist_ok=True)

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_modelconfig(
    create_savedmodel,
    models_dir,
    model_version,
    io_cnt,
    max_batch,
    dtype,
    shape,
    shape_tensor_input_dtype=None,
):
    if not tu.validate_for_trt_model(dtype, dtype, dtype, shape, shape, shape):
        return

    shape_str = tu.shape_to_dims_str(shape)

    model_name_base = "plan"
    if max_batch == 0:
        model_name_base += "_nobatch"
    if FLAGS.tensorrt_compat:
        model_name_base += "_compatible"
    model_name = tu.get_zero_model_name(model_name_base, io_cnt, dtype)
    if shape_tensor_input_dtype:
        model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
    config_dir = os.path.join(models_dir, model_name)

    if FLAGS.tensorrt_shape_io:
        shape_tensor_dim = len(shape)
        config = """
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
""".format(
            model_name, max_batch
        )

        for io_num in range(io_cnt):
            config += """
input [
  {{
    name: "DUMMY_INPUT{}"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT{}"
    data_type: {}
    dims: [ {} ]
    is_shape_tensor: true
  }}
]
output [
  {{
    name: "DUMMY_OUTPUT{}"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "OUTPUT{}"
    data_type: TYPE_INT64
    dims: [ {} ]
    is_shape_tensor: true
  }}
]
""".format(
                io_num,
                np_to_model_dtype(dtype),
                shape_str,
                io_num,
                np_to_model_dtype(shape_tensor_input_dtype),
                shape_tensor_dim,
                io_num,
                np_to_model_dtype(dtype),
                shape_str,
                io_num,
                shape_tensor_dim,
            )

    else:
        config = """
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
""".format(
            model_name, max_batch
        )

        for io_num in range(io_cnt):
            config += """
input [
  {{
    name: "INPUT{}"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT{}"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
                io_num,
                np_to_model_dtype(dtype),
                shape_str,
                io_num,
                np_to_model_dtype(dtype),
                shape_str,
            )

    os.makedirs(config_dir, exist_ok=True)

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_shape_tensor_models(
    models_dir, dtype, shape, shape_tensor_input_dtype, io_cnt=1, no_batch=True
):
    model_version = 1

    create_plan_modelconfig(
        True,
        models_dir,
        model_version,
        io_cnt,
        8,
        dtype,
        shape,
        shape_tensor_input_dtype,
    )
    create_plan_shape_tensor_modelfile(
        models_dir, model_version, io_cnt, 8, dtype, shape, 32, shape_tensor_input_dtype
    )
    if no_batch:
        create_plan_modelconfig(
            True,
            models_dir,
            model_version,
            io_cnt,
            0,
            dtype,
            shape,
            shape_tensor_input_dtype,
        )
        create_plan_shape_tensor_modelfile(
            models_dir,
            model_version,
            io_cnt,
            0,
            dtype,
            shape,
            32,
            shape_tensor_input_dtype,
        )


def create_models(models_dir, dtype, shape, io_cnt=1, no_batch=True):
    model_version = 1

    if FLAGS.graphdef:
        create_tf_modelconfig(False, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_tf_modelfile(False, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_tf_modelconfig(
                False, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_tf_modelfile(
                False, models_dir, model_version, io_cnt, 0, dtype, shape
            )

    if FLAGS.savedmodel:
        create_tf_modelconfig(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_tf_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_tf_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_tf_modelfile(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )

    if FLAGS.onnx:
        create_onnx_modelconfig(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        create_onnx_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_onnx_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_onnx_modelfile(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )

    if FLAGS.openvino:
        create_openvino_modelconfig(models_dir, model_version, io_cnt, 8, dtype, shape)
        create_openvino_modelfile(models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_openvino_modelconfig(
                models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_openvino_modelfile(
                models_dir, model_version, io_cnt, 0, dtype, shape
            )

    if FLAGS.libtorch:
        create_libtorch_modelconfig(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        create_libtorch_modelfile(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        if no_batch:
            create_libtorch_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_libtorch_modelfile(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )

    if FLAGS.tensorrt or FLAGS.tensorrt_compat:
        create_plan_modelconfig(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        create_plan_modelfile(
            True, models_dir, model_version, io_cnt, 8, dtype, shape, 32
        )
        if no_batch:
            create_plan_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_plan_modelfile(
                True, models_dir, model_version, io_cnt, 0, dtype, shape, 32
            )

    if FLAGS.tensorrt_big:
        create_plan_modelconfig(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        create_plan_modelfile(
            True, models_dir, model_version, io_cnt, 8, dtype, shape, 16 * 1024 * 1024
        )
        if no_batch:
            create_plan_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_plan_modelfile(
                True,
                models_dir,
                model_version,
                io_cnt,
                0,
                dtype,
                shape,
                16 * 1024 * 1024,
            )

    if FLAGS.ensemble:
        emu.create_nop_modelconfig(models_dir, shape, dtype)
        create_ensemble_modelconfig(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        create_ensemble_modelfile(
            True, models_dir, model_version, io_cnt, 8, dtype, shape
        )
        if no_batch:
            create_ensemble_modelconfig(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )
            create_ensemble_modelfile(
                True, models_dir, model_version, io_cnt, 0, dtype, shape
            )


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
        "--onnx",
        required=False,
        action="store_true",
        help="Generate Onnx Runtime Onnx models",
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
        "--tensorrt",
        required=False,
        action="store_true",
        help="Generate TensorRT PLAN models",
    )
    parser.add_argument(
        "--tensorrt-big",
        required=False,
        action="store_true",
        help="Generate TensorRT PLAN models w/ opt profile with large max",
    )
    parser.add_argument(
        "--tensorrt-compat",
        required=False,
        action="store_true",
        help="Generate TensorRT version-compatible models",
    )
    parser.add_argument(
        "--tensorrt-shape-io",
        required=False,
        action="store_true",
        help="Generate TensorRT PLAN models w/ shape tensor i/o",
    )
    parser.add_argument(
        "--ensemble",
        required=False,
        action="store_true",
        help="Generate ensemble models",
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io

        tf.compat.v1.disable_eager_execution()
    if FLAGS.onnx:
        import onnx
    if FLAGS.libtorch:
        import torch
        from torch import nn
    if (
        FLAGS.tensorrt
        or FLAGS.tensorrt_big
        or FLAGS.tensorrt_compat
        or FLAGS.tensorrt_shape_io
    ):
        import tensorrt as trt
    if FLAGS.openvino:
        import openvino.runtime as ov

    import test_util as tu

    # Create models with variable-sized input and output. For big
    # and version-compatible TensorRT models, only create the one
    # needed for testing.
    if FLAGS.tensorrt_big:
        create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=1)
    elif FLAGS.tensorrt_compat:
        create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=1, no_batch=False)
    elif FLAGS.tensorrt_shape_io:
        create_shape_tensor_models(
            FLAGS.models_dir, np.float32, [-1, -1], np.int32, io_cnt=1
        )
        create_shape_tensor_models(
            FLAGS.models_dir, np.float32, [-1, -1], np.int64, io_cnt=1
        )
    else:
        create_models(FLAGS.models_dir, bool, [-1], io_cnt=1)
        create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=1)
        create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=3)
        create_models(FLAGS.models_dir, np.float16, [-1, -1], io_cnt=1)
        create_models(FLAGS.models_dir, np.float16, [-1, -1], io_cnt=3)
        create_models(FLAGS.models_dir, np_dtype_string, [-1], io_cnt=1)
        create_models(FLAGS.models_dir, np_dtype_string, [-1, -1], io_cnt=3)

    # Create libtorch linalg model
    if FLAGS.libtorch:
        model_version = 1
        create_libtorch_linalg_modelconfig(True, FLAGS.models_dir, model_version)
        create_libtorch_linalg_modelfile(True, FLAGS.models_dir, model_version)
