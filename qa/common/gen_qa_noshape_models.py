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
import numpy as np

FLAGS = None
np_dtype_string = np.dtype(object)


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


def create_savedmodel_modelfile(models_dir,
                                max_batch,
                                model_version,
                                input_shape,
                                output0_shape,
                                output1_shape,
                                input_dtype,
                                output0_dtype,
                                output1_dtype,
                                swap=False):

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
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape([]),
                             "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape),
                             "TENSOR_INPUT1")
    else:
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape([]),
                             "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, [
            None,
        ] + tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT1")

    # If the input is a string, then convert each string to the
    # equivalent float value.
    if tf_input_dtype == tf.string:
        in0 = tf.strings.to_number(in0, tf.int32)
        in1 = tf.strings.to_number(in1, tf.int32)

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
    model_name = tu.get_model_name(
        "savedmodel_nobatch" if max_batch == 0 else "savedmodel", input_dtype,
        output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        input0_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_INPUT0:0")
        input1_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_INPUT1:0")
        output0_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_OUTPUT0:0")
        output1_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_OUTPUT1:0")
        tf.saved_model.simple_save(sess,
                                   model_version_dir + "/model.savedmodel",
                                   inputs={
                                       "INPUT0": input0_tensor,
                                       "INPUT1": input1_tensor
                                   },
                                   outputs={
                                       "OUTPUT0": output0_tensor,
                                       "OUTPUT1": output1_tensor
                                   })


def create_savedmodel_modelconfig(models_dir, max_batch, model_version,
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
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "savedmodel_nobatch" if max_batch == 0 else "savedmodel", input_dtype,
        output0_dtype, output1_dtype)
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

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_models(models_dir,
                  input_dtype,
                  output0_dtype,
                  output1_dtype,
                  input_shape,
                  output0_shape,
                  output1_shape,
                  output0_label_cnt,
                  version_policy=None):
    model_version = 1

    # Create two models, one that supports batching with a max-batch
    # of 8, and one that does not with a max-batch of 0

    if FLAGS.savedmodel:
        # max-batch 8
        create_savedmodel_modelconfig(models_dir, 8, model_version, input_shape,
                                      output0_shape, output1_shape, input_dtype,
                                      output0_dtype, output1_dtype,
                                      output0_label_cnt, version_policy)
        create_savedmodel_modelfile(models_dir, 8, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype)
        # max-batch 0
        create_savedmodel_modelconfig(models_dir, 0, model_version, input_shape,
                                      output0_shape, output1_shape, input_dtype,
                                      output0_dtype, output1_dtype,
                                      output0_label_cnt, version_policy)
        create_savedmodel_modelfile(models_dir, 0, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype)


def create_fixed_models(models_dir,
                        input_dtype,
                        output0_dtype,
                        output1_dtype,
                        version_policy=None):
    input_size = 16

    create_models(models_dir, input_dtype, output0_dtype, output1_dtype,
                  (input_size,), (input_size,), (input_size,), input_size,
                  version_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    parser.add_argument('--graphdef',
                        required=False,
                        action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--savedmodel',
                        required=False,
                        action='store_true',
                        help='Generate SavedModel models')
    parser.add_argument('--tensorrt',
                        required=False,
                        action='store_true',
                        help='Generate TensorRT PLAN models')
    parser.add_argument('--onnx',
                        required=False,
                        action='store_true',
                        help='Generate Onnx Runtime Onnx models')
    parser.add_argument('--libtorch',
                        required=False,
                        action='store_true',
                        help='Generate Pytorch LibTorch models')
    parser.add_argument('--variable',
                        required=False,
                        action='store_true',
                        help='Used variable-shape tensors for input/output')
    parser.add_argument('--ensemble',
                        required=False,
                        action='store_true',
                        help='Generate ensemble models against the models' +
                        ' in all platforms. Note that the models generated' +
                        ' are not completed.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.savedmodel:
        import tensorflow as tf

    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_fixed_models(FLAGS.models_dir, np.int8, np.int8, np.int8,
                            ('latest', 1))
        create_fixed_models(FLAGS.models_dir, np.int16, np.int16, np.int16,
                            ('latest', 2))
        create_fixed_models(FLAGS.models_dir, np.int32, np.int32, np.int32,
                            ('all', None))
        create_fixed_models(FLAGS.models_dir, np.int64, np.int64, np.int64)
        create_fixed_models(FLAGS.models_dir, np.float16, np.float16,
                            np.float16, ('specific', [
                                1,
                            ]))
        create_fixed_models(FLAGS.models_dir, np.float32, np.float32,
                            np.float32, ('specific', [1, 3]))
        create_fixed_models(FLAGS.models_dir, np.float16, np.float32,
                            np.float32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int8)
        create_fixed_models(FLAGS.models_dir, np.int8, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int16)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float32, np.float32)
        create_fixed_models(FLAGS.models_dir, np.float32, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float16, np.int16)

        if FLAGS.savedmodel:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            8,
                                            2, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            8,
                                            3, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            0,
                                            2, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            0,
                                            3, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
