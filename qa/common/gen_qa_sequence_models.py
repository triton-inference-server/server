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

def create_tf_modelfile(
        create_savedmodel, models_dir, model_version, max_batch, dtype):

    tf_dtype = np_to_tf_dtype(dtype)

    # Create the model. If non-batching then don't include the batch
    # dimension.
    tf.reset_default_graph()
    if create_savedmodel and (max_batch == 0):
        input0 = tf.placeholder(tf_dtype, [1,], "INPUT")
        start0 = tf.placeholder(tf.int32, [1,], "START")
        ready0 = tf.placeholder(tf.int32, [1,], "READY")
        acc = tf.get_variable("ACC", [1,], dtype=tf_dtype)
        tmp = tf.where(tf.equal(start0, 1), input0, tf.add(acc, input0))
        newacc = tf.where(tf.equal(ready0, 1), tmp, acc)
        assign = tf.assign(acc, newacc)
        output0 = tf.identity(assign, name="OUTPUT")
    else:
        # For batching we can't use a tf.variable to hold the
        # accumulated values since that forces the size of the output
        # to the size of the variable (which must be a max-batch-size
        # vector since require one accumulator each), instead of the
        # output shape being [None, 1]. So instead we just return the
        # 0 if not-ready and 'INPUT'+'START' otherwise... the tests
        # know to expect this.
        input0 = tf.placeholder(tf_dtype, [None,1], "INPUT")
        start0 = tf.placeholder(tf.int32, [None,1], "START")
        ready0 = tf.placeholder(tf.int32, [None,1], "READY")
        tmp = tf.where(tf.equal(ready0, 1), tf.add(start0, input0),
                       tf.zeros(tf.shape(input0), dtype=tf.int32))
        output0 = tf.identity(tmp, name="OUTPUT")

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_sequence_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", dtype)
    else:
        model_name = tu.get_sequence_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", dtype)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    if create_savedmodel:
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            input0_tensor = tf.get_default_graph().get_tensor_by_name("INPUT:0")
            start0_tensor = tf.get_default_graph().get_tensor_by_name("START:0")
            ready0_tensor = tf.get_default_graph().get_tensor_by_name("READY:0")
            output0_tensor = tf.get_default_graph().get_tensor_by_name("OUTPUT:0")
            tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                       inputs={"INPUT": input0_tensor, "START": start0_tensor,
                                               "READY" : ready0_tensor},
                                       outputs={"OUTPUT": output0_tensor})
    else:
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                                 "model.graphdef", as_text=False)

def create_tf_modelconfig(
        create_savedmodel, models_dir, model_version, max_batch, dtype):

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_sequence_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", dtype)
    else:
        model_name = tu.get_sequence_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", dtype)

    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "{}"
max_batch_size: {}
sequence_batching {{
  max_queue_delay_microseconds: 0
  control_input [
    {{
      name: "START"
      control [
        {{
          kind: CONTROL_SEQUENCE_START
          int32_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "READY"
      control [
        {{
          kind: CONTROL_SEQUENCE_READY
          int32_false_true: [ 0, 1 ]
        }}
      ]
    }}
  ]
}}
input [
  {{
    name: "INPUT"
    data_type: {}
    dims: [ 1 ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {}
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
  }}
]
'''.format(model_name,
           "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef",
           max_batch, np_to_model_dtype(dtype), np_to_model_dtype(dtype))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_models(models_dir, dtype):
    model_version = 1

    if FLAGS.graphdef:
        create_tf_modelconfig(False, models_dir, model_version, 8, dtype);
        create_tf_modelfile(False, models_dir, model_version, 8, dtype);
        create_tf_modelconfig(False, models_dir, model_version, 0, dtype);
        create_tf_modelfile(False, models_dir, model_version, 0, dtype);

    if FLAGS.savedmodel:
        create_tf_modelconfig(True, models_dir, model_version, 8, dtype);
        create_tf_modelfile(True, models_dir, model_version, 8, dtype);
        create_tf_modelconfig(True, models_dir, model_version, 0, dtype);
        create_tf_modelfile(True, models_dir, model_version, 0, dtype);

#    if FLAGS.netdef:
#        create_netdef_modelconfig(
#            models_dir, 8, model_version, dtype, (1,));
#        create_netdef_modelfile(
#            models_dir, 0, model_version, dtype, (1,));

#    if FLAGS.tensorrt:
#        create_tensorrt_modelconfig(
#            models_dir, 8, model_version, dtype, (1,1,1));
#        create_tensorrt_modelfile(
#            models_dir, 0, model_version, dtype, (1,1,1));

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
    parser.add_argument('--variable', required=False, action='store_true',
                        help='Used variable-shape tensors for input/output')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.netdef:
        from caffe2.python import core as c2core
        from caffe2.python import model_helper as c2model_helper
    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util
    if FLAGS.tensorrt:
        import tensorrt.legacy as trt

    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_models(FLAGS.models_dir, np.int32)

    # Tests with models that accept variable-shape input/output tensors
    if FLAGS.variable:
        pass
