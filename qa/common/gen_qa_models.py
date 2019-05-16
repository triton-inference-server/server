# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

def create_graphdef_modelfile(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    tf_input_dtype = np_to_tf_dtype(input_dtype)
    tf_output0_dtype = np_to_tf_dtype(output0_dtype)
    tf_output1_dtype = np_to_tf_dtype(output1_dtype)

    # Create the model. If non-batching then don't include the batch
    # dimension.
    tf.reset_default_graph()
    if max_batch == 0:
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape), "INPUT0")
        in1 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape), "INPUT1")
    else:
        in0 = tf.placeholder(tf_input_dtype, [None,] + tu.shape_to_tf_shape(input_shape), "INPUT0")
        in1 = tf.placeholder(tf_input_dtype, [None,] + tu.shape_to_tf_shape(input_shape), "INPUT1")

    # If the input is a string, then convert each string to the
    # equivalent int32 value.
    if tf_input_dtype == tf.string:
        in0 = tf.strings.to_number(in0, tf.int32)
        in1 = tf.strings.to_number(in1, tf.int32)

    # TF doesn't have GPU add or subtract operation for int8, int16 or
    # int32 so force those onto CPU.
    if ((input_dtype == np.int8) or (input_dtype == np.int16) or
        (input_dtype == np.int32)):
        with tf.device('/cpu:0'):
            add = tf.add(in0, in1, "ADD")
            sub = tf.subtract(in0, in1, "SUB")
    else:
        add = tf.add(in0, in1, "ADD")
        sub = tf.subtract(in0, in1, "SUB")

    # Cast or convert result to the output dtype.
    if tf_output0_dtype == tf.string:
        cast0 = tf.dtypes.as_string(add if not swap else sub, name="TOSTR0")
    else:
        cast0 = tf.cast(add if not swap else sub, tf_output0_dtype, "CAST0")

    if tf_output1_dtype == tf.string:
        cast1 = tf.dtypes.as_string(sub if not swap else add, name="TOSTR1")
    else:
        cast1 = tf.cast(sub if not swap else add, tf_output1_dtype, "CAST1")

    out0 = tf.identity(cast0, "OUTPUT0")
    out1 = tf.identity(cast1, "OUTPUT1")

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("graphdef_nobatch" if max_batch == 0 else "graphdef",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    with tf.Session() as sess:
        graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                             "model.graphdef", as_text=False)


def create_graphdef_modelconfig(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("graphdef_nobatch" if max_batch == 0 else "graphdef",
                                   input_dtype, output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_graphdef"
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
    label_filename: "output0_labels.txt"
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
           np_to_model_dtype(output0_dtype), tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype), tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_savedmodel_modelfile(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    tf_input_dtype = np_to_tf_dtype(input_dtype)
    tf_output0_dtype = np_to_tf_dtype(output0_dtype)
    tf_output1_dtype = np_to_tf_dtype(output1_dtype)

    # Create the model. If non-batching then don't include the batch
    # dimension.
    tf.reset_default_graph()
    if max_batch == 0:
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT1")
    else:
        in0 = tf.placeholder(tf_input_dtype, [None,] + tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, [None,] + tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT1")

    # If the input is a string, then convert each string to the
    # equivalent float value.
    if tf_input_dtype == tf.string:
        in0 = tf.strings.to_number(in0, tf.int32)
        in1 = tf.strings.to_number(in1, tf.int32)

    # TF doesn't have GPU add or subtract operation for int8, int16 or
    # int32 so force those onto CPU.
    if ((input_dtype == np.int8) or (input_dtype == np.int16) or
        (input_dtype == np.int32)):
        with tf.device('/cpu:0'):
            add = tf.add(in0, in1, "ADD")
            sub = tf.subtract(in0, in1, "SUB")
    else:
        add = tf.add(in0, in1, "ADD")
        sub = tf.subtract(in0, in1, "SUB")

    # Cast or convert result to the output dtype.
    if tf_output0_dtype == tf.string:
        cast0 = tf.dtypes.as_string(add if not swap else sub, name="TOSTR0")
    else:
        cast0 = tf.cast(add if not swap else sub, tf_output0_dtype, "CAST0")

    if tf_output1_dtype == tf.string:
        cast1 = tf.dtypes.as_string(sub if not swap else add, name="TOSTR1")
    else:
        cast1 = tf.cast(sub if not swap else add, tf_output1_dtype, "CAST1")

    out0 = tf.identity(cast0, "TENSOR_OUTPUT0")
    out1 = tf.identity(cast1, "TENSOR_OUTPUT1")

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("savedmodel_nobatch" if max_batch == 0 else "savedmodel",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    with tf.Session() as sess:
        input0_tensor = tf.get_default_graph().get_tensor_by_name("TENSOR_INPUT0:0")
        input1_tensor = tf.get_default_graph().get_tensor_by_name("TENSOR_INPUT1:0")
        output0_tensor = tf.get_default_graph().get_tensor_by_name("TENSOR_OUTPUT0:0")
        output1_tensor = tf.get_default_graph().get_tensor_by_name("TENSOR_OUTPUT1:0")
        tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                   inputs={"INPUT0": input0_tensor, "INPUT1": input1_tensor},
                                   outputs={"OUTPUT0": output0_tensor, "OUTPUT1": output1_tensor})


def create_savedmodel_modelconfig(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("savedmodel_nobatch" if max_batch == 0 else "savedmodel",
                                   input_dtype, output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_savedmodel"
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
    label_filename: "output0_labels.txt"
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
           np_to_model_dtype(output0_dtype), tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype), tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_netdef_modelfile(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

    if not tu.validate_for_c2_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    c2_input_dtype = np_to_c2_dtype(input_dtype)
    c2_output0_dtype = np_to_c2_dtype(output0_dtype)
    c2_output1_dtype = np_to_c2_dtype(output1_dtype)

    model_name = tu.get_model_name("netdef_nobatch" if max_batch == 0 else "netdef",
                                   input_dtype, output0_dtype, output1_dtype)

    # Create the model
    model = c2model_helper.ModelHelper(name=model_name)
    add = model.net.Add(["INPUT0", "INPUT1"], "add")
    sub = model.net.Sub(["INPUT0", "INPUT1"], "sub")
    out0 = model.net.Cast(["add" if not swap else "sub"], "OUTPUT0", to=c2_output0_dtype)
    out1 = model.net.Cast(["sub" if not swap else "add"], "OUTPUT1", to=c2_output1_dtype)

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
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    if not tu.validate_for_c2_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("netdef_nobatch" if max_batch == 0 else "netdef",
                                   input_dtype, output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "caffe2_netdef"
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
    label_filename: "output0_labels.txt"
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
           np_to_model_dtype(output0_dtype), tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype), tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_plan_modelfile(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)

    # Create the model
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    network = builder.create_network()
    in0 = network.add_input("INPUT0", trt_input_dtype, input_shape)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_shape)
    add = network.add_elementwise(in0, in1, trt.infer.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.infer.ElementWiseOperation.SUB)

    out0 = add if not swap else sub
    out1 = sub if not swap else add

    out0.get_output(0).set_name("OUTPUT0")
    out1.get_output(0).set_name("OUTPUT1")
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    builder.set_max_batch_size(max(1, max_batch))
    builder.set_max_workspace_size(1 << 20)
    engine = builder.build_cuda_engine(network)
    network.destroy()

    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    lengine = trt.lite.Engine(engine_stream=engine.serialize(),
                              max_batch_size=max(1, max_batch))
    lengine.save(model_version_dir + "/model.plan")
    engine.destroy()
    builder.destroy()


def create_plan_modelconfig(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
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
    label_filename: "output0_labels.txt"
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
           np_to_model_dtype(output0_dtype), tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype), tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_onnx_modelfile(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

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

    if not tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    onnx_input_dtype = np_to_onnx_dtype(input_dtype)
    onnx_output0_dtype = np_to_onnx_dtype(output0_dtype)
    onnx_output1_dtype = np_to_onnx_dtype(output1_dtype)

    onnx_input_shape = normalize_variable_shape(input_shape)
    onnx_output0_shape = normalize_variable_shape(output0_shape)
    onnx_output1_shape = normalize_variable_shape(output1_shape)

    # Create the model
    model_name = tu.get_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    batch_dim = [] if max_batch == 0 else [max_batch]

    in0 = onnx.helper.make_tensor_value_info("INPUT0", onnx_input_dtype, batch_dim + onnx_input_shape)
    in1 = onnx.helper.make_tensor_value_info("INPUT1", onnx_input_dtype, batch_dim + onnx_input_shape)

    out0 = onnx.helper.make_tensor_value_info("OUTPUT0", onnx_output0_dtype, batch_dim + onnx_output0_shape)
    out1 = onnx.helper.make_tensor_value_info("OUTPUT1", onnx_output1_dtype, batch_dim + onnx_output1_shape)

    internal_in0 = onnx.helper.make_node("Identity", ["INPUT0"], ["_INPUT0"])
    internal_in1 = onnx.helper.make_node("Identity", ["INPUT1"], ["_INPUT1"])

    # cast int8, int16 input to higer precision int as Onnx Add/Sub operator doesn't support those type
    if (onnx_input_dtype == onnx.TensorProto.INT8) or (onnx_input_dtype == onnx.TensorProto.INT16):
        internal_in0 = onnx.helper.make_node("Cast", ["INPUT0"], ["_INPUT0"], to=onnx.TensorProto.INT32)
        internal_in1 = onnx.helper.make_node("Cast", ["INPUT1"], ["_INPUT1"], to=onnx.TensorProto.INT32)

    add = onnx.helper.make_node("Add", ["_INPUT0", "_INPUT1"], ["CAST0" if not swap else "CAST1"])
    sub = onnx.helper.make_node("Sub", ["_INPUT0", "_INPUT1"], ["CAST1" if not swap else "CAST0"])
    cast0 = onnx.helper.make_node("Cast", ["CAST0"], ["OUTPUT0"], to=onnx_output0_dtype)
    cast1 = onnx.helper.make_node("Cast", ["CAST1"], ["OUTPUT1"], to=onnx_output1_dtype)

    # Avoid cast from float16 to float16
    # (bug in Onnx Runtime, cast from float16 to float16 will become cast from float16 to float32)
    if onnx_input_dtype == onnx.TensorProto.FLOAT16:
        if onnx_output0_dtype == onnx_input_dtype:
            cast0 = onnx.helper.make_node("Identity", ["CAST0"], ["OUTPUT0"])
        if onnx_output1_dtype == onnx_input_dtype:
            cast1 = onnx.helper.make_node("Identity", ["CAST1"], ["OUTPUT1"])

    onnx_nodes = [internal_in0, internal_in1, add, sub, cast0, cast1]
    onnx_inputs = [in0, in1]
    onnx_outputs = [out0, out1]

    graph_proto = onnx.helper.make_graph(onnx_nodes, model_name, onnx_inputs, onnx_outputs)
    model_def = onnx.helper.make_model(graph_proto, producer_name="TRTIS")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_onnx_modelconfig(
        models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    if not tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   input_dtype, output0_dtype, output1_dtype)
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
    # [TODO] move create_general_modelconfig() out of emu as it is general
    # enough for all backends to use
    config = emu.create_general_modelconfig(model_name, "onnxruntime_onnx", max_batch,
            emu.repeat(input_dtype, 2), emu.repeat(input_shape, 2), emu.repeat(None, 2),
            [output0_dtype, output1_dtype], [output0_shape, output1_shape], emu.repeat(None, 2),
            ["output0_labels.txt", None],
            version_policy=version_policy, force_tensor_number_suffix=True,
            instance_group_str=instance_group_string)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_models(
        models_dir, input_dtype, output0_dtype, output1_dtype,
        input_shape, output0_shape, output1_shape,
        output0_label_cnt, version_policy=None):
    model_version = 1

    # Create two models, one that supports batching with a max-batch
    # of 8, and one that does not with a max-batch of 0
    if FLAGS.graphdef:
        # max-batch 8
        create_graphdef_modelconfig(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_graphdef_modelfile(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)
        # max-batch 0
        create_graphdef_modelconfig(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_graphdef_modelfile(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)

    if FLAGS.savedmodel:
        # max-batch 8
        create_savedmodel_modelconfig(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_savedmodel_modelfile(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)
        # max-batch 0
        create_savedmodel_modelconfig(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_savedmodel_modelfile(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)

    if FLAGS.netdef:
        # max-batch 8
        create_netdef_modelconfig(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_netdef_modelfile(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)
        # max-batch 0
        create_netdef_modelconfig(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_netdef_modelfile(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)

    if FLAGS.tensorrt:
        # max-batch 8
        create_plan_modelconfig(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_plan_modelfile(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)
        # max-batch 0
        create_plan_modelconfig(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_plan_modelfile(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)

    if FLAGS.onnx:
        # max-batch 8
        create_onnx_modelconfig(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_onnx_modelfile(
            models_dir, 8, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)
        # max-batch 0
        create_onnx_modelconfig(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype,
            output0_label_cnt, version_policy)
        create_onnx_modelfile(
            models_dir, 0, model_version,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype)

    if FLAGS.ensemble:
        for pair in emu.platform_types_and_validation():
            if not pair[1](input_dtype, output0_dtype, output1_dtype,
                            input_shape, output0_shape, output1_shape):
                continue
            if pair[0] == "plan":
                if len(input_shape) == 1:
                    input_shape = (input_shape[0], 1, 1)
                if len(output0_shape) == 1:
                    output0_shape = (output0_shape[0], 1, 1)
                if len(output1_shape) == 1:
                    output1_shape = (output1_shape[0], 1, 1)

            # max-batch 8
            emu.create_ensemble_modelconfig(
                pair[0], models_dir, 8, model_version,
                input_shape, output0_shape, output1_shape,
                input_dtype, output0_dtype, output1_dtype,
                output0_label_cnt, version_policy)
            emu.create_ensemble_modelfile(
                pair[0], models_dir, 8, model_version,
                input_shape, output0_shape, output1_shape,
                input_dtype, output0_dtype, output1_dtype)
            # max-batch 0
            emu.create_ensemble_modelconfig(
                pair[0], models_dir, 0, model_version,
                input_shape, output0_shape, output1_shape,
                input_dtype, output0_dtype, output1_dtype,
                output0_label_cnt, version_policy)
            emu.create_ensemble_modelfile(
                pair[0], models_dir, 0, model_version,
                input_shape, output0_shape, output1_shape,
                input_dtype, output0_dtype, output1_dtype)


def create_fixed_models(
        models_dir, input_dtype, output0_dtype, output1_dtype, version_policy=None):
    input_size = 16

    if FLAGS.tensorrt:
        create_models(models_dir, input_dtype, output0_dtype, output1_dtype,
                      (input_size, 1, 1), (input_size, 1, 1), (input_size, 1, 1),
                      input_size, version_policy)
    else:
        create_models(models_dir, input_dtype, output0_dtype, output1_dtype,
                      (input_size,), (input_size,), (input_size,),
                      input_size, version_policy)


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
    parser.add_argument('--tensorrt', required=False, action='store_true',
                        help='Generate TensorRT PLAN models')
    parser.add_argument('--onnx', required=False, action='store_true',
                        help='Generate Onnx Runtime Onnx models')
    parser.add_argument('--variable', required=False, action='store_true',
                        help='Used variable-shape tensors for input/output')
    parser.add_argument('--ensemble', required=False, action='store_true',
                        help='Generate ensemble models against the models'
                        + ' in all platforms. Note that the models generated'
                        + ' are not completed.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.netdef:
        from caffe2.python import core as c2core
        from caffe2.python import model_helper as c2model_helper
    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util
    if FLAGS.tensorrt:
        import tensorrt.legacy as trt
    if FLAGS.onnx:
        import onnx

    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_fixed_models(FLAGS.models_dir, np.int8, np.int8, np.int8, ('latest', 1))
        create_fixed_models(FLAGS.models_dir, np.int16, np.int16, np.int16, ('latest', 2))
        create_fixed_models(FLAGS.models_dir, np.int32, np.int32, np.int32, ('all', None))
        create_fixed_models(FLAGS.models_dir, np.int64, np.int64, np.int64)
        create_fixed_models(FLAGS.models_dir, np.float16, np.float16, np.float16, ('specific', [1,]))
        create_fixed_models(FLAGS.models_dir, np.float32, np.float32, np.float32, ('specific', [1, 3]))
        create_fixed_models(FLAGS.models_dir, np.float16, np.float32, np.float32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int8)
        create_fixed_models(FLAGS.models_dir, np.int8, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int16)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float32, np.float32)
        create_fixed_models(FLAGS.models_dir, np.float32, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float16, np.int16)

        create_fixed_models(FLAGS.models_dir, np_dtype_string, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np_dtype_string, np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np.int32, np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np_dtype_string, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np_dtype_string, np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int32, np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np.int32, np_dtype_string, np.int32)

        # Make multiple versions of some models for version testing
        # (they use different version policies when created above)
        if FLAGS.graphdef:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_graphdef_modelfile(FLAGS.models_dir, 8, 2,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_graphdef_modelfile(FLAGS.models_dir, 8, 3,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_graphdef_modelfile(FLAGS.models_dir, 0, 2,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_graphdef_modelfile(FLAGS.models_dir, 0, 3,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)

        if FLAGS.savedmodel:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_savedmodel_modelfile(FLAGS.models_dir, 8, 2,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir, 8, 3,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir, 0, 2,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir, 0, 3,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)

        if FLAGS.netdef:
            for vt in [np.float32, np.int32]:
                create_netdef_modelfile(FLAGS.models_dir, 8, 2,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_netdef_modelfile(FLAGS.models_dir, 8, 3,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_netdef_modelfile(FLAGS.models_dir, 0, 2,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_netdef_modelfile(FLAGS.models_dir, 0, 3,
                                            (16,), (16,), (16,), vt, vt, vt, swap=True)

        if FLAGS.tensorrt:
            for vt in [np.float32,]:
                create_plan_modelfile(FLAGS.models_dir, 8, 2,
                                            (16,1,1), (16,1,1), (16,1,1), vt, vt, vt, swap=True)
                create_plan_modelfile(FLAGS.models_dir, 8, 3,
                                            (16,1,1), (16,1,1), (16,1,1), vt, vt, vt, swap=True)
                create_plan_modelfile(FLAGS.models_dir, 0, 2,
                                            (16,1,1), (16,1,1), (16,1,1), vt, vt, vt, swap=True)
                create_plan_modelfile(FLAGS.models_dir, 0, 3,
                                            (16,1,1), (16,1,1), (16,1,1), vt, vt, vt, swap=True)

        if FLAGS.onnx:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_onnx_modelfile(FLAGS.models_dir, 8, 2,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_onnx_modelfile(FLAGS.models_dir, 8, 3,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_onnx_modelfile(FLAGS.models_dir, 0, 2,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)
                create_onnx_modelfile(FLAGS.models_dir, 0, 3,
                                          (16,), (16,), (16,), vt, vt, vt, swap=True)

        if FLAGS.ensemble:
            for pair in emu.platform_types_and_validation():
                shape = (16, 1, 1) if pair[0] == "plan" else (16,)
                for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                    if not pair[1](vt, vt, vt, shape, shape, shape):
                        continue
                    emu.create_ensemble_modelfile(pair[0], FLAGS.models_dir, 8, 2,
                                            shape, shape, shape, vt, vt, vt, swap=True)
                    emu.create_ensemble_modelfile(pair[0], FLAGS.models_dir, 8, 3,
                                            shape, shape, shape, vt, vt, vt, swap=True)
                    emu.create_ensemble_modelfile(pair[0], FLAGS.models_dir, 0, 2,
                                            shape, shape, shape, vt, vt, vt, swap=True)
                    emu.create_ensemble_modelfile(pair[0], FLAGS.models_dir, 0, 3,
                                            shape, shape, shape, vt, vt, vt, swap=True)

    # Tests with models that accept variable-shape input/output tensors
    if FLAGS.variable:
        create_models(FLAGS.models_dir, np.float32, np.float32, np.float32, (-1,), (16,), (16,), 16)
        create_models(FLAGS.models_dir, np.float32, np.int32, np.int32, (-1,-1), (2,8), (2,8), 16)
        create_models(FLAGS.models_dir, np.float32, np.int64, np.int64, (8,-1), (8,4), (8,4), 32)
        create_models(FLAGS.models_dir, np.float32, np.int32, np.int64, (-1,8,-1), (2,8,2), (2,8,2), 32)
        create_models(FLAGS.models_dir, np.float32, np.float32, np.int32, (-1,), (-1,), (-1,), 16)
        create_models(FLAGS.models_dir, np.int32, np.int32, np.int32, (-1,-1), (-1,8), (-1,-1), 16)
        create_models(FLAGS.models_dir, np.int32, np.int32, np.float32, (-1,8,-1), (-1,8,-1), (2,8,2), 32)

        create_models(FLAGS.models_dir, np_dtype_string, np_dtype_string, np_dtype_string,
                      (-1,), (16,), (16,), 16)
        create_models(FLAGS.models_dir, np_dtype_string, np.int32, np.int32,
                      (-1,-1), (2,8), (2,8), 16)
        create_models(FLAGS.models_dir, np_dtype_string, np_dtype_string, np.int32,
                      (8,-1), (8,4), (8,4), 32)
        create_models(FLAGS.models_dir, np_dtype_string, np.int32, np_dtype_string,
                      (-1,8,-1), (2,8,2), (2,8,2), 32)

    if FLAGS.ensemble:
        # Create utility models used in ensemble
        # nop (only creates model config, should add model file before use)
        model_dtypes = ["TYPE_BOOL", "TYPE_STRING"]
        for s in [8, 16, 32, 64]:
            for t in ["INT", "UINT", "FP"]:
                if t == "FP" and s == 8:
                    continue
                model_dtypes.append("TYPE_{}{}".format(t, s))

        for model_dtype in model_dtypes:
            # Use variable size to handle all shape. Note: piping variable size output
            # to fixed size model is not safe but doable
            for model_shape in [(-1,), (-1, -1), (-1, -1, -1)]:
                emu.create_nop_modelconfig(FLAGS.models_dir, model_shape, model_dtype)
