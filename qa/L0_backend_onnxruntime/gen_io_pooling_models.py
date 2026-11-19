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

# Reference script used to create the checked-in io_pooling_models/*/1/model.onnx.
# Usage: python3 gen_io_pooling_models.py <io_pooling_models dir>
import os
import sys

import onnx
from onnx import TensorProto, helper

# Constants are Constant nodes rather than initializers: an initializer listed
# in graph.input is overridable and would mix concerns into unrelated models.


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _model(graph, opset, ir_version):
    model = helper.make_model(
        graph,
        producer_name="triton",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = ir_version
    onnx.checker.check_model(model)
    return model


def add_sub(k):
    nodes = [
        helper.make_node("Add", ["INPUT0", "INPUT1"], ["OUTPUT0"]),
        helper.make_node("Sub", ["INPUT0", "INPUT1"], ["OUTPUT1"]),
    ]
    graph = helper.make_graph(
        nodes,
        "add_sub",
        [_vi("INPUT0", ["N", k]), _vi("INPUT1", ["N", k])],
        [_vi("OUTPUT0", ["N", k]), _vi("OUTPUT1", ["N", k])],
    )
    return _model(graph, 17, 8)


def passthrough():
    one = helper.make_tensor("c", TensorProto.FLOAT, [1], [1.0])
    nodes = [
        helper.make_node("Constant", [], ["c"], value=one),
        helper.make_node("Add", ["X", "c"], ["Y"]),
    ]
    x = _vi("X", ["N", 4])
    # The graph input X is directly a graph output.
    graph = helper.make_graph(nodes, "passthrough", [x], [x, _vi("Y", ["N", 4])])
    return _model(graph, 17, 8)


def int4_initializer():
    # Wq is also listed in graph.input so that it is an overridable initializer
    # of a dtype Triton cannot map. io_pooling_test.py recomputes its values.
    q = [((i * 5 + j * 3) % 16) - 8 for i in range(16) for j in range(16)]
    wq = helper.make_tensor("Wq", TensorProto.INT4, [16, 16], q)
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5])
    nodes = [
        helper.make_node("Constant", [], ["scale"], value=scale),
        helper.make_node("DequantizeLinear", ["Wq", "scale"], ["Wf"]),
        helper.make_node("MatMul", ["X", "Wf"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "int4_initializer",
        [
            _vi("X", ["N", 16]),
            helper.make_tensor_value_info("Wq", TensorProto.INT4, [16, 16]),
        ],
        [_vi("Y", ["N", 16])],
        initializer=[wq],
    )
    return _model(graph, 21, 10)


def shape_mismatch():
    # Y is declared [N, 16] but execution yields Tile(X, [1, 2]) = [N, 32]. The
    # repeats are derived from the values of X (0 * max|X| + [1, 2]) so that
    # shape inference cannot see the real shape: an inferred [N, 32] would be
    # merged with the declaration into [N, ?] and the model would fail to load.
    rep = helper.make_tensor("rep", TensorProto.INT64, [2], [1, 2])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    nodes = [
        helper.make_node("Constant", [], ["rep"], value=rep),
        helper.make_node("Constant", [], ["zero"], value=zero),
        helper.make_node("Abs", ["X"], ["absx"]),
        helper.make_node("ReduceMax", ["absx"], ["maxx"], keepdims=0),
        helper.make_node("Mul", ["maxx", "zero"], ["zf"]),
        helper.make_node("Cast", ["zf"], ["zi"], to=TensorProto.INT64),
        helper.make_node("Add", ["rep", "zi"], ["repeats"]),
        helper.make_node("Tile", ["X", "repeats"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "shape_mismatch", [_vi("X", ["N", 16])], [_vi("Y", ["N", 16])]
    )
    return _model(graph, 17, 8)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    models = {
        "add_sub_static": add_sub(16),
        "add_sub_dynamic": add_sub("K"),
        "passthrough": passthrough(),
        "int4_initializer": int4_initializer(),
        "shape_mismatch": shape_mismatch(),
    }
    for base, model in models.items():
        os.makedirs(os.path.join(out_dir, base, "1"), exist_ok=True)
        onnx.save(model, os.path.join(out_dir, base, "1", "model.onnx"))
