# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gen_ensemble_model_utils as emu

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


def np_to_torch_dtype(np_dtype):
    if np_dtype == bool:
        return torch.bool
    elif np_dtype == np.int8:
        return torch.int8
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int
    elif np_dtype == np.int64:
        return torch.long
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.uint16:
        return None  # Not supported in Torch
    elif np_dtype == np.float16:
        return None
    elif np_dtype == np.float32:
        return torch.float
    elif np_dtype == np.float64:
        return torch.double
    elif np_dtype == np_dtype_string:
        return None  # Not supported in Torch
    return None


def create_onnx_modelfile(models_dir, model_version, max_batch, dtype, shape):

    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape,
                                      shape):
        return

    model_name = tu.get_sequence_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(
        model_version)

    # Create the model. For now don't implement a proper accumulator
    # just return 0 if not-ready and 'INPUT'+'START' otherwise...  the
    # tests know to expect this.
    onnx_dtype = np_to_onnx_dtype(dtype)
    onnx_control_dtype = onnx_dtype
    onnx_input_shape, idx = tu.shape_to_onnx_shape(shape, 0)
    onnx_output_shape, idx = tu.shape_to_onnx_shape(shape, idx)

    # If the input is a string then use int32 for operation and just
    # cast to/from string for input and output.
    if onnx_dtype == onnx.TensorProto.STRING:
        onnx_control_dtype = onnx.TensorProto.INT32

    # If input dtype is bool, then use bool type for control and
    # int32 type for input/output
    if onnx_dtype == onnx.TensorProto.BOOL:
        onnx_dtype = onnx.TensorProto.INT32

    batch_dim = [] if max_batch == 0 else [None]

    onnx_input = onnx.helper.make_tensor_value_info(
        "INPUT", onnx_dtype, batch_dim + onnx_input_shape)
    onnx_input_state = onnx.helper.make_tensor_value_info(
        "INPUT_STATE", onnx_dtype, batch_dim + onnx_input_shape)
    onnx_start = onnx.helper.make_tensor_value_info("START",
                                                    onnx_control_dtype,
                                                    batch_dim + [1])
    onnx_ready = onnx.helper.make_tensor_value_info("READY",
                                                    onnx_control_dtype,
                                                    batch_dim + [1])
    onnx_output = onnx.helper.make_tensor_value_info(
        "OUTPUT", onnx_dtype, batch_dim + onnx_output_shape)
    onnx_output_state = onnx.helper.make_tensor_value_info(
        "OUTPUT_STATE", onnx_dtype, batch_dim + onnx_output_shape)

    internal_input = onnx.helper.make_node("Identity", ["INPUT"], ["_INPUT"])
    internal_input_state = onnx.helper.make_node("Identity", ["INPUT_STATE"],
                                                 ["_INPUT_STATE"])
    # cast int8, int16 input to higer precision int as Onnx Add/Sub operator doesn't support those type
    # Also casting String data type to int32
    if ((onnx_dtype == onnx.TensorProto.INT8)
            or (onnx_dtype == onnx.TensorProto.INT16)
            or (onnx_dtype == onnx.TensorProto.STRING)):
        internal_input = onnx.helper.make_node("Cast", ["INPUT"], ["_INPUT"],
                                               to=onnx.TensorProto.INT32)
        internal_input_state = onnx.helper.make_node("Cast", ["INPUT_STATE"],
                                                     ["_INPUT_STATE"],
                                                     to=onnx.TensorProto.INT32)

    # Convert boolean value to int32 value
    if onnx_control_dtype == onnx.TensorProto.BOOL:
        internal_input1 = onnx.helper.make_node("Cast", ["START"], ["_START"],
                                                to=onnx.TensorProto.INT32)
        internal_input2 = onnx.helper.make_node("Cast", ["READY"], ["_READY"],
                                                to=onnx.TensorProto.INT32)
        add = onnx.helper.make_node("Add", ["_INPUT", "_START"], ["add"])
        add_state = onnx.helper.make_node("Add", ["add", "_INPUT_STATE"],
                                          ["add_state"])

        # Take advantage of knowledge that the READY false value is 0 and true is 1
        mul = onnx.helper.make_node("Mul", ["_READY", "add_state"], ["CAST"])

    else:
        add = onnx.helper.make_node("Add", ["_INPUT", "START"], ["add"])
        add_state = onnx.helper.make_node("Add", ["add", "_INPUT_STATE"],
                                          ["add_state"])
        # Take advantage of knowledge that the READY false value is 0 and true is 1
        mul = onnx.helper.make_node("Mul", ["READY", "add_state"], ["CAST"])

    cast = onnx.helper.make_node("Cast", ["CAST"], ["OUTPUT"], to=onnx_dtype)
    cast_output_state = onnx.helper.make_node("Cast", ["CAST"],
                                              ["OUTPUT_STATE"],
                                              to=onnx_dtype)

    # Avoid cast from float16 to float16
    # (bug in Onnx Runtime, cast from float16 to float16 will become cast from float16 to float32)
    if onnx_dtype == onnx.TensorProto.FLOAT16:
        cast = onnx.helper.make_node("Identity", ["CAST"], ["OUTPUT"])
        cast_output_state = onnx.helper.make_node("Identity", ["CAST"],
                                                  ["OUTPUT_STATE"])

    if onnx_control_dtype == onnx.TensorProto.BOOL:
        start_output_state = onnx.helper.make_node(
            "Identity",
            ["_START"],
            ["OUTPUT_STATE"],
        )
        start_output = onnx.helper.make_node("Identity", ["_START"],
                                             ["OUTPUT"])
    else:
        start_output_state = onnx.helper.make_node("Identity", ["START"],
                                                   ["OUTPUT_STATE"])
        start_output = onnx.helper.make_node("Identity", ["START"], ["OUTPUT"])

    if onnx_control_dtype == onnx.TensorProto.BOOL:
        onnx_nodes = [
            internal_input, internal_input_state, internal_input1,
            internal_input2, add, add_state, mul, cast, cast_output_state
        ]
        onnx_then_branch_nodes = [
            internal_input1, start_output, start_output_state
        ]
    else:
        onnx_nodes = [
            internal_input, internal_input_state, add, add_state, mul, cast,
            cast_output_state
        ]
        onnx_then_branch_nodes = [start_output, start_output_state]

    onnx_inputs = [onnx_input_state, onnx_input, onnx_start, onnx_ready]
    onnx_outputs = [onnx_output, onnx_output_state]

    then_body = onnx.helper.make_graph(onnx_then_branch_nodes, 'then_body',
                                       [onnx_start], onnx_outputs)
    else_body = onnx.helper.make_graph(onnx_nodes, 'else_body', onnx_inputs,
                                       onnx_outputs)
    start_cond = onnx.helper.make_node('Cast', ['START'], ['_START_COND'],
                                       to=onnx.TensorProto.BOOL)
    if_node = onnx.helper.make_node('If',
                                    inputs=['_START_COND'],
                                    outputs=['OUTPUT', 'OUTPUT_STATE'],
                                    then_branch=then_body,
                                    else_branch=else_body)

    graph_proto = onnx.helper.make_graph([start_cond, if_node], model_name, onnx_inputs,
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


def create_onnx_modelconfig(models_dir, model_version, max_batch, dtype,
                            shape):

    if not tu.validate_for_onnx_model(dtype, dtype, dtype, shape, shape,
                                      shape):
        return

    model_name = tu.get_sequence_model_name(
        "onnx_nobatch" if max_batch == 0 else "onnx", dtype)
    config_dir = models_dir + "/" + model_name

    if dtype == np.float32:
        control_type = "fp32"
    elif dtype == np.bool:
        control_type = "bool"
        dtype = np.int32
    else:
        control_type = "int32"

    instance_group_string = '''
instance_group [
  {
    kind: KIND_GPU
  }
]
'''

    # [TODO] move create_general_modelconfig() out of emu as it is general
    # enough for all backends to use
    config = emu.create_general_modelconfig(
        model_name,
        "onnxruntime_onnx",
        max_batch, [dtype], [shape], [None], [dtype], [shape], [None], [None],
        force_tensor_number_suffix=False,
        instance_group_str=instance_group_string)

    config += '''
sequence_batching
{{
  max_sequence_idle_microseconds: 5000000
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
      name: "READY"
      control [
        {{
          kind: CONTROL_SEQUENCE_READY
          {type}_false_true: [ 0, 1 ]
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
'''.format(type=control_type,
           dims=tu.shape_to_dims_str(shape),
           dtype=emu.dtype_str(dtype))

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
    parser.add_argument(
        '--tensorrt-shape-io',
        required=False,
        action='store_true',
        help='Generate TensorRT PLAN models w/ shape tensor i/o')
    parser.add_argument('--onnx',
                        required=False,
                        action='store_true',
                        help='Generate Onnx models')
    parser.add_argument(
        '--onnx_opset',
        type=int,
        required=False,
        default=0,
        help='Opset used for Onnx models. Default is to use ONNXRT default')
    parser.add_argument('--libtorch',
                        required=False,
                        action='store_true',
                        help='Generate Pytorch LibTorch models')
    parser.add_argument('--openvino',
                        required=False,
                        action='store_true',
                        help='Generate OpenVino models')
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

    if FLAGS.onnx:
        import onnx
    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_models(FLAGS.models_dir, np.float32, [
            1,
        ])
        create_models(FLAGS.models_dir, np.int32, [
            1,
        ])
        create_models(FLAGS.models_dir, np_dtype_string, [
            1,
        ])
        create_models(FLAGS.models_dir, np.bool, [
            1,
        ])

    # Tests with models that accept variable-shape input/output tensors
    if FLAGS.variable:
        create_models(FLAGS.models_dir, np.int32, [
            -1,
        ], False)
        create_models(FLAGS.models_dir, np.float32, [
            -1,
        ], False)
        create_models(FLAGS.models_dir, np_dtype_string, [
            -1,
        ], False)
        create_models(FLAGS.models_dir, np.bool, [
            -1,
        ], False)
