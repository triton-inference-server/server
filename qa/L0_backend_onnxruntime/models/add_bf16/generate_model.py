#!/usr/bin/env python
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import onnx


def generate_bf16_add_model(models_dir):
    """Generate a simple BFLOAT16 Add model (INPUT0 + INPUT1 = OUTPUT)."""
    model_name = "add_bf16"
    shape = [5, 5]
    onnx_dtype = onnx.TensorProto.BFLOAT16

    add = onnx.helper.make_node("Add", ["INPUT0", "INPUT1"], ["OUTPUT"])

    input0 = onnx.helper.make_tensor_value_info("INPUT0", onnx_dtype, shape)
    input1 = onnx.helper.make_tensor_value_info("INPUT1", onnx_dtype, shape)
    output = onnx.helper.make_tensor_value_info("OUTPUT", onnx_dtype, shape)

    graph_proto = onnx.helper.make_graph(
        [add],
        "bf16_add",
        [input0, input1],
        [output],
    )
    model_def = onnx.helper.make_model(graph_proto, producer_name="triton")
    # Cap IR version for older ONNX Runtime (e.g. max supported 11)
    model_def.ir_version = min(model_def.ir_version, 11)
    # BFLOAT16 support requires opset 13+
    model_def.opset_import[0].version = 13

    model_dir = os.path.join(models_dir, model_name, "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx.save(model_def, os.path.join(model_dir, "model.onnx"))

    # Write config.pbtxt
    config = """platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "INPUT0"
    data_type: TYPE_BF16
    dims: {shape}
  }},
  {{
    name: "INPUT1"
    data_type: TYPE_BF16
    dims: {shape}
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: TYPE_BF16
    dims: {shape}
  }}
]
""".format(
        shape=shape
    )

    config_path = os.path.join(models_dir, model_name, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)

    print(f"Generated model '{model_name}' in {models_dir}")


if __name__ == "__main__":
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    generate_bf16_add_model(models_dir)
