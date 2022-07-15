# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def np_to_tf_dtype(np_dtype):
    if np_dtype == bool:
        return tf.bool
    elif np_dtype == np.int8:
        return tf.int8
    elif np_dtype == np.int16:
        return tf.int16
    elif np_dtype == np.int32:
        return tf.int32
    elif np_dtype == np.int64:
        return tf.int64
    elif np_dtype == np.uint8:
        return tf.uint8
    elif np_dtype == np.uint16:
        return tf.uint16
    elif np_dtype == np.float16:
        return tf.float16
    elif np_dtype == np.float32:
        return tf.float32
    elif np_dtype == np.float64:
        return tf.float64
    elif np_dtype == np_dtype_string:
        return tf.string
    return None


def np_to_onnx_dtype(np_dtype):
    if np_dtype == bool:
        return onnx.TensorProto.BOOL
    elif np_dtype == np.int8:
        return onnx.TensorProto.INT8
    elif np_dtype == np.int16:
        return onnx.TensorProto.INT16
    elif np_dtype == np.int32:
        return onnx.TensorProto.INT32
    elif np_dtype == np.int64:
        return onnx.TensorProto.INT64
    elif np_dtype == np.uint8:
        return onnx.TensorProto.UINT8
    elif np_dtype == np.uint16:
        return onnx.TensorProto.UINT16
    elif np_dtype == np.float16:
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.float32:
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.float64:
        return onnx.TensorProto.DOUBLE
    elif np_dtype == np_dtype_string:
        return onnx.TensorProto.STRING
    return None


def create_savedmodel_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input testing.
    # Because the ragged input and batch input are one dimensional vector
    # when passing to the model, the model must generate output with batch
    # dimension so that Triton can scatter it to different responses along
    # the batch dimension.
    # 'BATCH_AND_SIZE_INPUT' is also used as a hint to generate output with
    # batch dimension, 'BATCH_AND_SIZE_INPUT' must have shape [batch_size].
    # Each output corresponds to the input with the same name, so if there
    # are two requests, one has "RAGGED_INPUT" [2, 4] and the other has [1],
    # since the input is ragged, the model sees the input as [2, 4, 1], and
    # "BATCH_AND_SIZE_INPUT" will have shape [2]. Then the model output will
    # be [[2, 4, 1], [2, 4, 1]] and Triton will send responses that each has
    # value [[2, 4, 1]].
    # For "BATCH_INPUT", the input tensor must only have one variable dimension
    # to be broadcasted along the batch dimension properly, thus the currently
    # allowed batch input types are:
    # - BATCH_ACCUMULATED_ELEMENT_COUNT
    # - BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO
    # - BATCH_MAX_ELEMENT_COUNT_AS_SHAPE
    # - BATCH_ITEM_SHAPE_FLATTEN

    tf_dtype = np_to_tf_dtype(dtype)

    tf.reset_default_graph()
    in_node = tf.placeholder(tf_dtype, tu.shape_to_tf_shape([-1]),
                             "TENSOR_RAGGED_INPUT")
    bs_node = tf.placeholder(tf_dtype, tu.shape_to_tf_shape([-1]),
                             "TENSOR_BATCH_AND_SIZE_INPUT")
    batch_node = tf.placeholder(tf_dtype, tu.shape_to_tf_shape([-1]),
                                "TENSOR_BATCH_INPUT")

    in_mat = tf.reshape(in_node, [1, -1])
    bs_mat = tf.reshape(bs_node, [1, -1])
    batch_mat = tf.reshape(batch_node, [1, -1])

    output_expander = tf.reshape(tf.divide(bs_node, bs_node), [-1, 1])

    out_node = tf.matmul(output_expander, in_mat, name="TENSOR_RAGGED_OUTPUT")
    bs_out_node = tf.matmul(output_expander,
                            bs_mat,
                            name="TENSOR_BATCH_AND_SIZE_OUTPUT")
    batch_out_node = tf.matmul(output_expander,
                               batch_mat,
                               name="TENSOR_BATCH_OUTPUT")

    model_name = "savedmodel_batch_input"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        in_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_RAGGED_INPUT:0")
        bs_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_AND_SIZE_INPUT:0")
        batch_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_INPUT:0")
        out_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_RAGGED_OUTPUT:0")
        bs_out_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_AND_SIZE_OUTPUT:0")
        batch_out_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_OUTPUT:0")
        tf.saved_model.simple_save(sess,
                                   model_version_dir + "/model.savedmodel",
                                   inputs={
                                       "RAGGED_INPUT": in_tensor,
                                       "BATCH_AND_SIZE_INPUT": bs_tensor,
                                       "BATCH_INPUT": batch_tensor,
                                   },
                                   outputs={
                                       "RAGGED_OUTPUT": out_tensor,
                                       "BATCH_AND_SIZE_OUTPUT": bs_out_tensor,
                                       "BATCH_OUTPUT": batch_out_tensor,
                                   })


def create_plan_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input testing.
    # Because the ragged input and batch input are one dimensional vector
    # when passing to the model, the model must generate output with batch
    # dimension so that Triton can scatter it to different responses along
    # the batch dimension.
    # 'BATCH_AND_SIZE_INPUT' is also used as a hint to generate output with
    # batch dimension, 'BATCH_AND_SIZE_INPUT' must have shape [batch_size].
    # Each output corresponds to the input with the same name, so if there
    # are two requests, one has "RAGGED_INPUT" [2, 4] and the other has [1],
    # since the input is ragged, the model sees the input as [2, 4, 1], and
    # "BATCH_AND_SIZE_INPUT" will have shape [2]. Then the model output will
    # be [[2, 4, 1], [2, 4, 1]] and Triton will send responses that each has
    # value [[2, 4, 1]].
    # For "BATCH_INPUT", the input tensor must only have one variable dimension
    # to be broadcasted along the batch dimension properly, thus the currently
    # allowed batch input types are:
    # - BATCH_ACCUMULATED_ELEMENT_COUNT
    # - BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO
    # - BATCH_MAX_ELEMENT_COUNT_AS_SHAPE
    # - BATCH_ITEM_SHAPE_FLATTEN

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
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name = "plan_batch_input"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_onnx_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input testing.
    # Because the ragged input and batch input are one dimensional vector
    # when passing to the model, the model must generate output with batch
    # dimension so that Triton can scatter it to different responses along
    # the batch dimension.
    # 'BATCH_AND_SIZE_INPUT' is also used as a hint to generate output with
    # batch dimension, 'BATCH_AND_SIZE_INPUT' must have shape [batch_size].
    # Each output corresponds to the input with the same name, so if there
    # are two requests, one has "RAGGED_INPUT" [2, 4] and the other has [1],
    # since the input is ragged, the model sees the input as [2, 4, 1], and
    # "BATCH_AND_SIZE_INPUT" will have shape [2]. Then the model output will
    # be [[2, 4, 1], [2, 4, 1]] and Triton will send responses that each has
    # value [[2, 4, 1]].
    # For "BATCH_INPUT", the input tensor must only have one variable dimension
    # to be broadcasted along the batch dimension properly, thus the currently
    # allowed batch input types are:
    # - BATCH_ACCUMULATED_ELEMENT_COUNT
    # - BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO
    # - BATCH_MAX_ELEMENT_COUNT_AS_SHAPE
    # - BATCH_ITEM_SHAPE_FLATTEN

    onnx_dtype = np_to_onnx_dtype(dtype)

    # Create the model
    model_name = "onnx_batch_input"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    in0_shape, idx = tu.shape_to_onnx_shape([-1], 0)
    bs_shape, idx = tu.shape_to_onnx_shape([-1], 0)
    batch_shape, idx = tu.shape_to_onnx_shape([-1], 0)

    in0 = onnx.helper.make_tensor_value_info("RAGGED_INPUT", onnx_dtype,
                                             in0_shape)
    bs_in = onnx.helper.make_tensor_value_info("BATCH_AND_SIZE_INPUT",
                                               onnx_dtype, bs_shape)
    batch_in = onnx.helper.make_tensor_value_info("BATCH_INPUT", onnx_dtype,
                                                  batch_shape)

    out_shape, idx = tu.shape_to_onnx_shape([-1, -1], idx)
    bs_out_shape, idx = tu.shape_to_onnx_shape([-1, -1], idx)
    batch_out_shape, idx = tu.shape_to_onnx_shape([-1, -1], idx)

    out = onnx.helper.make_tensor_value_info("RAGGED_OUTPUT", onnx_dtype,
                                             out_shape)
    bs_out = onnx.helper.make_tensor_value_info("BATCH_AND_SIZE_OUTPUT",
                                                onnx_dtype, bs_out_shape)
    batch_out = onnx.helper.make_tensor_value_info("BATCH_OUTPUT", onnx_dtype,
                                                   batch_out_shape)

    const_node_shape = onnx.helper.make_node(
        'Constant', [], ["shape"],
        value=onnx.helper.make_tensor("const_shape", onnx.TensorProto.INT64,
                                      [2], [1, -1]))

    const_node_expander_shape = onnx.helper.make_node(
        'Constant', [], ["expander_shape"],
        value=onnx.helper.make_tensor("const_expander_shape",
                                      onnx.TensorProto.INT64, [2], [-1, 1]))

    in0_mat_node = onnx.helper.make_node("Reshape", ["RAGGED_INPUT", "shape"],
                                         ["in_mat"])
    bs_mat_node = onnx.helper.make_node("Reshape",
                                        ["BATCH_AND_SIZE_INPUT", "shape"],
                                        ["bs_mat"])
    batch_mat_node = onnx.helper.make_node("Reshape", ["BATCH_INPUT", "shape"],
                                           ["batch_mat"])

    internal_node_div = onnx.helper.make_node(
        "Div", ["BATCH_AND_SIZE_INPUT", "BATCH_AND_SIZE_INPUT"],
        ["output_expander_int"])
    internal_node_reshape = onnx.helper.make_node(
        "Reshape", ["output_expander_int", "expander_shape"],
        ["output_expander"])

    out_node = onnx.helper.make_node("MatMul", ["output_expander", "in_mat"],
                                     ["RAGGED_OUTPUT"])
    bs_out_node = onnx.helper.make_node("MatMul", ["output_expander", "bs_mat"],
                                        ["BATCH_AND_SIZE_OUTPUT"])
    batch_out_node = onnx.helper.make_node("MatMul",
                                           ["output_expander", "batch_mat"],
                                           ["BATCH_OUTPUT"])

    onnx_nodes = [
        const_node_shape, const_node_expander_shape, in0_mat_node, bs_mat_node,
        batch_mat_node, internal_node_div, internal_node_reshape, out_node,
        bs_out_node, batch_out_node
    ]
    onnx_inputs = [in0, bs_in, batch_in]
    onnx_outputs = [out, bs_out, batch_out]

    graph_proto = onnx.helper.make_graph(onnx_nodes, model_name, onnx_inputs,
                                         onnx_outputs)
    if FLAGS.onnx_opset > 0:
        model_opset = onnx.helper.make_operatorsetid("", FLAGS.onnx_opset)
        model_def = onnx.helper.make_model(graph_proto,
                                           producer_name="triton",
                                           opset_imports=[model_opset])
    else:
        model_def = onnx.helper.make_model(graph_proto, producer_name="triton")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_modelconfig(models_dir, max_batch, model_version, dtype, backend,
                       platform):
    version_policy_str = "{ latest { num_versions: 1 }}"

    backend_spec = '''
backend: "{}"
'''.format(backend)
    if backend == "tensorflow":
        backend_spec += '''
platform: "{}_{}"
'''.format(backend, platform)

    model_name = "{}_batch_input".format(platform)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
{}
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
    kind: BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO
    target_name: "BATCH_INPUT"
    data_type: {data_type}
    source_input: "RAGGED_INPUT"
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 1000000
}}
'''.format(model_name,
           backend_spec,
           max_batch,
           version_policy_str,
           data_type=np_to_model_dtype(dtype))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_savedmodel_itemshape_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input 'BATCH_ITEM_SHAPE' testing,
    # such model has one ragged input and one batch input, and one output to
    # return the batch input directly. Because 'BATCH_ITEM_SHAPE' should be
    # generated to have matching batch dimension, the output can be produced
    # via identity op and expect Triton will scatter the output properly.

    tf_dtype = np_to_tf_dtype(dtype)

    tf.reset_default_graph()
    in_node = tf.placeholder(tf_dtype, tu.shape_to_tf_shape([-1]),
                             "TENSOR_RAGGED_INPUT")
    # Shape is predefined
    batch_node = tf.placeholder(tf_dtype, tu.shape_to_tf_shape([-1, 2]),
                                "TENSOR_BATCH_INPUT")
    batch_output_node = tf.identity(batch_node, name="TENSOR_BATCH_OUTPUT")

    model_name = "savedmodel_batch_item"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        in_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_RAGGED_INPUT:0")
        batch_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_INPUT:0")
        batch_out_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_BATCH_OUTPUT:0")
        tf.saved_model.simple_save(sess,
                                   model_version_dir + "/model.savedmodel",
                                   inputs={
                                       "RAGGED_INPUT": in_tensor,
                                       "BATCH_INPUT": batch_tensor,
                                   },
                                   outputs={
                                       "BATCH_OUTPUT": batch_out_tensor,
                                   })


def create_plan_itemshape_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input 'BATCH_ITEM_SHAPE' testing,
    # such model has one ragged input and one batch input, and one output to
    # return the batch input directly. Because 'BATCH_ITEM_SHAPE' should be
    # generated to have matching batch dimension, the output can be produced
    # via identity op and expect Triton will scatter the output properly.

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt_dtype = np_to_trt_dtype(dtype)

    in_node = network.add_input("RAGGED_INPUT", trt_dtype, [-1])
    batch_node = network.add_input("BATCH_INPUT", trt_dtype, [-1, 2])

    batch_out_node = network.add_identity(batch_node)
    batch_out_node.get_output(0).name = "BATCH_OUTPUT"
    network.mark_output(batch_out_node.get_output(0))

    # Hard coded optimization profile
    min_shape = [1]
    opt_shape = [8]
    max_shape = [32]

    profile = builder.create_optimization_profile()
    profile.set_shape("RAGGED_INPUT", min_shape, opt_shape, max_shape)
    profile.set_shape("BATCH_INPUT", min_shape + [2], opt_shape + [2],
                      max_shape + [2])
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 20
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_name = "plan_batch_item"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_onnx_itemshape_modelfile(models_dir, model_version, dtype):
    # Create special identity model for batch input 'BATCH_ITEM_SHAPE' testing,
    # such model has one ragged input and one batch input, and one output to
    # return the batch input directly. Because 'BATCH_ITEM_SHAPE' should be
    # generated to have matching batch dimension, the output can be produced
    # via identity op and expect Triton will scatter the output properly.

    onnx_dtype = np_to_onnx_dtype(dtype)

    # Create the model
    model_name = "onnx_batch_item"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    in0_shape, idx = tu.shape_to_onnx_shape([-1], 0)
    batch_shape, idx = tu.shape_to_onnx_shape([-1, 2], 0)

    in0 = onnx.helper.make_tensor_value_info("RAGGED_INPUT", onnx_dtype,
                                             in0_shape)
    batch_in = onnx.helper.make_tensor_value_info("BATCH_INPUT", onnx_dtype,
                                                  batch_shape)

    batch_out_shape, idx = tu.shape_to_onnx_shape([-1, -1], idx)
    batch_out = onnx.helper.make_tensor_value_info("BATCH_OUTPUT", onnx_dtype,
                                                   batch_out_shape)

    onnx_nodes = [
        onnx.helper.make_node("Identity", ["BATCH_INPUT"], ["BATCH_OUTPUT"])
    ]
    onnx_inputs = [in0, batch_in]
    onnx_outputs = [batch_out]

    graph_proto = onnx.helper.make_graph(onnx_nodes, model_name, onnx_inputs,
                                         onnx_outputs)
    if FLAGS.onnx_opset > 0:
        model_opset = onnx.helper.make_operatorsetid("", FLAGS.onnx_opset)
        model_def = onnx.helper.make_model(graph_proto,
                                           producer_name="triton",
                                           opset_imports=[model_opset])
    else:
        model_def = onnx.helper.make_model(graph_proto, producer_name="triton")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_itemshape_modelconfig(models_dir, max_batch, model_version, dtype,
                                 backend, platform):
    version_policy_str = "{ latest { num_versions: 1 }}"

    backend_spec = '''
backend: "{}"
'''.format(backend)
    if backend == "tensorflow":
        backend_spec += '''
platform: "{}_{}"
'''.format(backend, platform)

    model_name = "{}_batch_item".format(platform)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
{}
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "RAGGED_INPUT"
    data_type: {data_type}
    dims: [ -1, -1 ]
    allow_ragged_batch: true
  }}
]
output [
  {{
    name: "BATCH_OUTPUT"
    data_type: {data_type}
    dims: [ 2 ]
   }}
]
batch_input [
  {{
    kind: BATCH_ITEM_SHAPE
    target_name: "BATCH_INPUT"
    data_type: {data_type}
    source_input: "RAGGED_INPUT"
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 1000000
}}
'''.format(model_name,
           backend_spec,
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
        create_modelconfig(models_dir, 4, model_version, np.float32, "tensorrt",
                           "plan")
        create_plan_modelfile(models_dir, model_version, np.float32)
        create_itemshape_modelconfig(models_dir, 4, model_version, np.float32,
                                     "tensorrt", "plan")
        create_plan_itemshape_modelfile(models_dir, model_version, np.float32)
    if FLAGS.savedmodel:
        create_modelconfig(models_dir, 4, model_version, np.float32,
                           "tensorflow", "savedmodel")
        create_savedmodel_modelfile(models_dir, model_version, np.float32)
        create_itemshape_modelconfig(models_dir, 4, model_version, np.float32,
                                     "tensorflow", "savedmodel")
        create_savedmodel_itemshape_modelfile(models_dir, model_version,
                                              np.float32)
    if FLAGS.onnx:
        create_modelconfig(models_dir, 4, model_version, np.float32,
                           "onnxruntime", "onnx")
        create_onnx_modelfile(models_dir, model_version, np.float32)
        create_itemshape_modelconfig(models_dir, 4, model_version, np.float32,
                                     "onnxruntime", "onnx")
        create_onnx_itemshape_modelfile(models_dir, model_version, np.float32)


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
    parser.add_argument('--savedmodel',
                        required=False,
                        action='store_true',
                        help='Generate SavedModel models')
    parser.add_argument('--graphdef',
                        required=False,
                        action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--onnx',
                        required=False,
                        action='store_true',
                        help='Generate Onnx Runtime Onnx models')
    parser.add_argument(
        '--onnx_opset',
        type=int,
        required=False,
        default=0,
        help='Opset used for Onnx models. Default is to use ONNXRT default')

    FLAGS, unparsed = parser.parse_known_args()

    import test_util as tu
    if FLAGS.tensorrt:
        import tensorrt as trt
    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
    if FLAGS.onnx:
        import onnx

    create_batch_input_models(FLAGS.models_dir)
