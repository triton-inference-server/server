#!/usr/bin/env python3
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

# Writes models/add_onnx/1/model.onnx, an Add graph: OUTPUT0 = INPUT0 + INPUT1.

import os

import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Add", ["INPUT0", "INPUT1"], ["OUTPUT0"])],
    "add",
    [
        helper.make_tensor_value_info("INPUT0", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("INPUT1", TensorProto.FLOAT, [4]),
    ],
    [helper.make_tensor_value_info("OUTPUT0", TensorProto.FLOAT, [4])],
)
# ir_version 7 pairs with opset 13; onnx's default (its newest IR) can exceed
# what ORT accepts.
model = helper.make_model(
    graph, ir_version=7, opset_imports=[helper.make_opsetid("", 13)]
)
os.makedirs("models/add_onnx/1", exist_ok=True)
onnx.save(model, "models/add_onnx/1/model.onnx")
