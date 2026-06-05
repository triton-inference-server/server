#!/usr/bin/env python3

# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

sys.path.append("../common")
import argparse
import os

import numpy as np
import onnx
import test_util as tu
from gen_common import np_to_model_dtype, np_to_onnx_dtype


def create_onnx_modelfile(models_dir, shape, dtype, model_version=1):
    onnx_io_dtype = np_to_onnx_dtype(dtype)

    # Create the model
    model_name = f"onnx_scalar_{len(shape)}dim"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    input = onnx.helper.make_tensor_value_info("INPUT", onnx_io_dtype, None)

    output = onnx.helper.make_tensor_value_info("OUTPUT", onnx_io_dtype, None)

    identity = onnx.helper.make_node("Identity", ["INPUT"], ["OUTPUT"])

    onnx_nodes = [identity]
    onnx_inputs = [input]
    onnx_outputs = [output]

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


def create_onnx_modelconfig(models_dir, dtype, shape):
    # Create the model
    model_name = f"onnx_scalar_{len(shape)}dim"
    config_dir = models_dir + "/" + model_name

    config = """
input [
  {{
    name: "INPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
""".format(
        np_to_model_dtype(dtype),
        tu.shape_to_dims_str(shape),
        np_to_model_dtype(dtype),
        tu.shape_to_dims_str(shape),
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    parser.add_argument(
        "--onnx_opset",
        type=int,
        required=False,
        default=0,
        help="Opset used for Onnx models. Default is to use ONNXRT default",
    )

    FLAGS = parser.parse_args()

    if not FLAGS.models_dir:
        raise Exception("--models_dir is required")

    create_onnx_modelfile(FLAGS.models_dir, shape=[1], dtype=np.float32)
    create_onnx_modelconfig(FLAGS.models_dir, shape=[1], dtype=np.float32)
    create_onnx_modelfile(FLAGS.models_dir, shape=[1, 1], dtype=np.float32)
    create_onnx_modelconfig(FLAGS.models_dir, shape=[1, 1], dtype=np.float32)
