# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import onnx

# Reference script on how the model used in this test is created
if __name__ == "__main__":
    values = np.ones((5, 5)).astype(np.float32)
    onnx_dtype = onnx.TensorProto.FLOAT
    initialized_input = onnx.helper.make_tensor(
        name="INITIALIZER",
        data_type=onnx_dtype,
        dims=values.shape,
        vals=values.flatten().astype(float),
    )
    add = onnx.helper.make_node("Add", ["INPUT", "INITIALIZER"], ["OUTPUT"])

    input = onnx.helper.make_tensor_value_info("INPUT", onnx_dtype, values.shape)
    initializer = onnx.helper.make_tensor_value_info(
        "INITIALIZER", onnx_dtype, values.shape
    )
    output = onnx.helper.make_tensor_value_info("OUTPUT", onnx_dtype, values.shape)

    graph_proto = onnx.helper.make_graph(
        [add],
        "init_input",
        [input, initializer],
        [output],
        initializer=[initialized_input],
    )
    model_def = onnx.helper.make_model(graph_proto, producer_name="triton")
    onnx.save(model_def, "model.onnx")
