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
from builtins import range
import os
import sys
import numpy as np
import gen_ensemble_model_utils as emu

FLAGS = None
np_dtype_string = np.dtype(object)

def np_to_model_dtype(np_dtype):
    if np_dtype == np.bool:
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

def np_to_tf_dtype(np_dtype):
    if np_dtype == np.bool:
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

def np_to_c2_dtype(np_dtype):
    if np_dtype == np.bool:
        return c2core.DataType.BOOL
    elif np_dtype == np.int8:
        return c2core.DataType.INT8
    elif np_dtype == np.int16:
        return c2core.DataType.INT16
    elif np_dtype == np.int32:
        return c2core.DataType.INT32
    elif np_dtype == np.int64:
        return c2core.DataType.INT64
    elif np_dtype == np.uint8:
        return c2core.DataType.UINT8
    elif np_dtype == np.uint16:
        return c2core.DataType.UINT16
    elif np_dtype == np.float16:
        return c2core.DataType.FLOAT16
    elif np_dtype == np.float32:
        return c2core.DataType.FLOAT
    elif np_dtype == np.float64:
        return c2core.DataType.DOUBLE
    elif np_dtype == np_dtype_string:
        return c2core.DataType.STRING
    return None

def np_to_trt_dtype(np_dtype):
    if np_dtype == np.int8:
        return trt.infer.DataType.INT8
    elif np_dtype == np.int32:
        return trt.infer.DataType.INT32
    elif np_dtype == np.float16:
        return trt.infer.DataType.HALF
    elif np_dtype == np.float32:
        return trt.infer.DataType.FLOAT
    return None

def np_to_onnx_dtype(np_dtype):
    if np_dtype == np.bool:
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

def create_tf_modelfile(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    tf_dtype = np_to_tf_dtype(dtype)

    # Create the model that copies inputs to corresponding outputs.
    tf.reset_default_graph()
    for io_num in range(io_cnt):
        input_name = "INPUT{}".format(io_num)
        output_name = "OUTPUT{}".format(io_num)
        if max_batch == 0:
            tin = tf.placeholder(tf_dtype, tu.shape_to_tf_shape(shape), input_name)
        else:
            tin = tf.placeholder(tf_dtype, [None,] + tu.shape_to_tf_shape(shape), input_name)
        toutput = tf.identity(tin, name=output_name)

    # Use model name based on io_cnt and non-batching variant
    if create_savedmodel:
        model_name = tu.get_zero_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", io_cnt, dtype)
    else:
        model_name = tu.get_zero_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", io_cnt, dtype)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    if create_savedmodel:
        with tf.Session() as sess:
            input_dict = {}
            output_dict = {}
            for io_num in range(io_cnt):
                input_name = "INPUT{}".format(io_num)
                output_name = "OUTPUT{}".format(io_num)
                input_tensor = tf.get_default_graph().get_tensor_by_name(input_name + ":0")
                output_tensor = tf.get_default_graph().get_tensor_by_name(output_name + ":0")
                input_dict[input_name] = input_tensor
                output_dict[output_name] = output_tensor
            tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                       inputs=input_dict, outputs=output_dict)
    else:
        with tf.Session() as sess:
            graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                                 "model.graphdef", as_text=False)

def create_tf_modelconfig(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    shape_str = tu.shape_to_dims_str(shape)

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_zero_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", io_cnt, dtype)
    else:
        model_name = tu.get_zero_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", io_cnt, dtype)

    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "{}"
max_batch_size: {}
'''.format(model_name,
           "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef",
           max_batch)

    for io_num in range(io_cnt):
        config += '''
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
'''.format(io_num, np_to_model_dtype(dtype), shape_str,
           io_num, np_to_model_dtype(dtype), shape_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_netdef_modelfile(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    if not tu.validate_for_c2_model(dtype, dtype, dtype, shape, shape, shape):
        return

    c2_dtype = np_to_c2_dtype(dtype)
    model_name = tu.get_zero_model_name(
        "netdef_nobatch" if max_batch == 0 else "netdef", io_cnt, dtype)

    # Create the model that copies inputs to corresponding outputs.
    model = c2model_helper.ModelHelper(name=model_name)
    for io_num in range(io_cnt):
        model.net.Copy("INPUT{}".format(io_num), "OUTPUT{}".format(io_num))

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(model_version_dir + "/model.netdef", "wb") as f:
        f.write(model.Proto().SerializeToString())
    with open(model_version_dir + "/init_model.netdef", "wb") as f:
        f.write(model.InitProto().SerializeToString())


def create_netdef_modelconfig(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    if not tu.validate_for_c2_model(dtype, dtype, dtype, shape, shape, shape):
        return

    shape_str = tu.shape_to_dims_str(shape)

    model_name = tu.get_zero_model_name(
        "netdef_nobatch" if max_batch == 0 else "netdef", io_cnt, dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "caffe2_netdef"
max_batch_size: {}
'''.format(model_name, max_batch)

    for io_num in range(io_cnt):
        config += '''
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
'''.format(io_num, np_to_model_dtype(dtype), shape_str,
           io_num, np_to_model_dtype(dtype), shape_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_ensemble_modelfile(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):
    if not tu.validate_for_ensemble_model("zero", dtype, dtype, dtype,
                                    shape, shape, shape):
        return

    emu.create_identity_ensemble_modelfile(
        "zero", models_dir, model_version, max_batch,
        dtype, [shape] * io_cnt, [shape] * io_cnt)


def create_ensemble_modelconfig(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):
    if not tu.validate_for_ensemble_model("zero", dtype, dtype, dtype,
                                    shape, shape, shape):
        return

    emu.create_identity_ensemble_modelconfig(
        "zero", models_dir, model_version, max_batch, dtype,
        [shape] * io_cnt, [shape] * io_cnt, [shape] * io_cnt, [shape] * io_cnt)


def create_onnx_modelfile(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    # Onnx use string for variable size dimension, and the same string
    # will be inferred to have same value for the model run.
    def normalize_variable_shape(shape, increment_index=True):
        res = []
        for dim in shape:
            if dim == -1:
                res.append("var_" + str(normalize_variable_shape.idx))
                if increment_index:
                    normalize_variable_shape.idx += 1
            else:
                res.append(dim)
        return res
        
    normalize_variable_shape.idx = 0

    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    onnx_dtype = np_to_onnx_dtype(dtype)

    # Create the model
    model_name = tu.get_zero_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   io_cnt, dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    batch_dim = [] if max_batch == 0 else [max_batch]

    onnx_nodes = []
    onnx_inputs = []
    onnx_outputs = []
    for io_num in range(io_cnt):
        # Repeat so that the variable dimension name is different
        in_shape = normalize_variable_shape(shape)
        out_shape = normalize_variable_shape(shape)
        in_name = "INPUT{}".format(io_num)
        out_name = "OUTPUT{}".format(io_num)

        onnx_inputs.append(onnx.helper.make_tensor_value_info(in_name, onnx_dtype, batch_dim + in_shape))
        onnx_outputs.append(onnx.helper.make_tensor_value_info(out_name, onnx_dtype, batch_dim + out_shape))
        onnx_nodes.append(onnx.helper.make_node("Identity", [in_name], [out_name]))

    graph_proto = onnx.helper.make_graph(onnx_nodes, model_name, onnx_inputs, onnx_outputs)
    model_def = onnx.helper.make_model(graph_proto, producer_name="TRTIS")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_onnx_modelconfig(
        create_savedmodel, models_dir, model_version, io_cnt, max_batch, dtype, shape):

    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # Use a different model name for the non-batching variant
    model_name = tu.get_zero_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   io_cnt, dtype)
    config_dir = models_dir + "/" + model_name
    
    # Must make sure all Onnx models will be loaded to the same GPU if they are
    # run on GPU. This is due to the current limitation of Onnx Runtime
    instance_group_string = '''
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
'''
    
    config = emu.create_general_modelconfig(model_name, "onnxruntime_onnx", max_batch,
            emu.repeat(dtype, io_cnt), emu.repeat(shape, io_cnt), emu.repeat(shape, io_cnt),
            emu.repeat(dtype, io_cnt), emu.repeat(shape, io_cnt), emu.repeat(shape, io_cnt),
            emu.repeat(None, io_cnt), force_tensor_number_suffix=True,
            instance_group_str=instance_group_string)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_models(models_dir, dtype, shape, io_cnt=1, no_batch=True):
    model_version = 1

    if FLAGS.graphdef:
        create_tf_modelconfig(False, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_tf_modelfile(False, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_tf_modelconfig(False, models_dir, model_version, io_cnt, 0, dtype, shape)
            create_tf_modelfile(False, models_dir, model_version, io_cnt, 0, dtype, shape)

    if FLAGS.savedmodel:
        create_tf_modelconfig(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_tf_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_tf_modelconfig(True, models_dir, model_version, io_cnt, 0, dtype, shape)
            create_tf_modelfile(True, models_dir, model_version, io_cnt, 0, dtype, shape)

    if FLAGS.netdef:
        create_netdef_modelconfig(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_netdef_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_netdef_modelconfig(True, models_dir, model_version, io_cnt, 0, dtype, shape)
            create_netdef_modelfile(True, models_dir, model_version, io_cnt, 0, dtype, shape)

    if FLAGS.onnx:
        create_onnx_modelconfig(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_onnx_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_onnx_modelconfig(True, models_dir, model_version, io_cnt, 0, dtype, shape)
            create_onnx_modelfile(True, models_dir, model_version, io_cnt, 0, dtype, shape)

    if FLAGS.ensemble:
        emu.create_nop_modelconfig(models_dir, shape, dtype)
        create_ensemble_modelconfig(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        create_ensemble_modelfile(True, models_dir, model_version, io_cnt, 8, dtype, shape)
        if no_batch:
            create_ensemble_modelconfig(True, models_dir, model_version, io_cnt, 0, dtype, shape)
            create_ensemble_modelfile(True, models_dir, model_version, io_cnt, 0, dtype, shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Top-level model directory')
    parser.add_argument('--graphdef', required=False, action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--savedmodel', required=False, action='store_true',
                        help='Generate SavedModel models')
    parser.add_argument('--netdef', required=False, action='store_true',
                        help='Generate NetDef models')
    parser.add_argument('--onnx', required=False, action='store_true',
                        help='Generate Onnx Runtime Onnx models')
    parser.add_argument('--ensemble', required=False, action='store_true',
                        help='Generate ensemble models')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.netdef:
        from caffe2.python import core as c2core
        from caffe2.python import model_helper as c2model_helper
    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util
    if FLAGS.onnx:
        import onnx

    import test_util as tu

    # Create models with variable-sized input and output.
    create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=1)
    create_models(FLAGS.models_dir, np.float32, [-1], io_cnt=3)
    create_models(FLAGS.models_dir, np.float16, [-1,-1], io_cnt=1)
    create_models(FLAGS.models_dir, np.float16, [-1,-1], io_cnt=3)
    create_models(FLAGS.models_dir, np_dtype_string, [-1], io_cnt=1)
    create_models(FLAGS.models_dir, np_dtype_string, [-1,-1], io_cnt=3)
