#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info, make_tensor
)
from onnx.checker import check_model

def create_simple_cnn_model():
    """Create a simple CNN model for testing MPS backend"""
    
    # Input
    input_tensor = make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 3, 224, 224]
    )
    
    # Output 
    output_tensor = make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 64, 56, 56]
    )
    
    # Create weight tensors
    # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    conv_weight = make_tensor(
        'conv_weight',
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten()
    )
    
    # Conv bias shape: [out_channels]
    conv_bias = make_tensor(
        'conv_bias',
        TensorProto.FLOAT,
        [64],
        np.zeros(64).astype(np.float32)
    )
    
    # Create nodes
    conv_node = make_node(
        'Conv',
        inputs=['input', 'conv_weight', 'conv_bias'],
        outputs=['conv_output'],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
        name='conv1'
    )
    
    relu_node = make_node(
        'Relu',
        inputs=['conv_output'],
        outputs=['relu_output'],
        name='relu1'
    )
    
    maxpool_node = make_node(
        'MaxPool',
        inputs=['relu_output'],
        outputs=['output'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        name='maxpool1'
    )
    
    # Create graph
    graph = make_graph(
        [conv_node, relu_node, maxpool_node],
        'simple_cnn',
        [input_tensor],
        [output_tensor],
        [conv_weight, conv_bias]
    )
    
    # Create model
    model = make_model(graph)
    model.opset_import[0].version = 11
    
    # Check model
    check_model(model)
    
    return model

def create_resnet_block_model():
    """Create a ResNet-style block for testing"""
    
    # Input
    input_tensor = make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 64, 56, 56]
    )
    
    # Output
    output_tensor = make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 64, 56, 56]
    )
    
    # Create weight tensors for two 3x3 convolutions
    conv1_weight = make_tensor(
        'conv1_weight',
        TensorProto.FLOAT,
        [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).astype(np.float32).flatten() * 0.01
    )
    
    conv1_bias = make_tensor(
        'conv1_bias',
        TensorProto.FLOAT,
        [64],
        np.zeros(64).astype(np.float32)
    )
    
    conv2_weight = make_tensor(
        'conv2_weight',
        TensorProto.FLOAT,
        [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).astype(np.float32).flatten() * 0.01
    )
    
    conv2_bias = make_tensor(
        'conv2_bias',
        TensorProto.FLOAT,
        [64],
        np.zeros(64).astype(np.float32)
    )
    
    # Create nodes
    conv1_node = make_node(
        'Conv',
        inputs=['input', 'conv1_weight', 'conv1_bias'],
        outputs=['conv1_output'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name='conv1'
    )
    
    relu1_node = make_node(
        'Relu',
        inputs=['conv1_output'],
        outputs=['relu1_output'],
        name='relu1'
    )
    
    conv2_node = make_node(
        'Conv',
        inputs=['relu1_output', 'conv2_weight', 'conv2_bias'],
        outputs=['conv2_output'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name='conv2'
    )
    
    # Residual connection
    add_node = make_node(
        'Add',
        inputs=['conv2_output', 'input'],
        outputs=['add_output'],
        name='residual_add'
    )
    
    relu2_node = make_node(
        'Relu',
        inputs=['add_output'],
        outputs=['output'],
        name='relu2'
    )
    
    # Create graph
    graph = make_graph(
        [conv1_node, relu1_node, conv2_node, add_node, relu2_node],
        'resnet_block',
        [input_tensor],
        [output_tensor],
        [conv1_weight, conv1_bias, conv2_weight, conv2_bias]
    )
    
    # Create model
    model = make_model(graph)
    model.opset_import[0].version = 11
    
    # Check model
    check_model(model)
    
    return model

def main():
    """Generate test models for MPS backend"""
    
    # Create directories
    os.makedirs('models/mps_simple/1', exist_ok=True)
    os.makedirs('models/mps_resnet_block/1', exist_ok=True)
    
    # Generate simple CNN model
    simple_model = create_simple_cnn_model()
    onnx.save(simple_model, 'models/mps_simple/1/model.onnx')
    print("Created simple CNN model: models/mps_simple/1/model.onnx")
    
    # Generate ResNet block model
    resnet_model = create_resnet_block_model()
    onnx.save(resnet_model, 'models/mps_resnet_block/1/model.onnx')
    print("Created ResNet block model: models/mps_resnet_block/1/model.onnx")
    
    # Create config for ResNet block
    resnet_config = """name: "mps_resnet_block"
backend: "metal_mps"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 64, 56, 56 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 64, 56, 56 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
"""
    
    with open('models/mps_resnet_block/config.pbtxt', 'w') as f:
        f.write(resnet_config)
    print("Created ResNet block config: models/mps_resnet_block/config.pbtxt")

if __name__ == '__main__':
    main()