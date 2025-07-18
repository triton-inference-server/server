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

"""
Create a simple CoreML model for testing the Triton CoreML backend.
"""

import os
import numpy as np
import coremltools as ct
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    """A simple neural network for testing."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_simple_coreml_model():
    """Create a simple CoreML model for testing."""
    print("Creating simple CoreML test model...")
    
    # Define model parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    
    # Create PyTorch model
    model = SimpleModel(input_size, hidden_size, output_size)
    model.eval()
    
    # Create example input
    example_input = torch.rand(1, input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, input_size))],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS11
    )
    
    # Create model directory
    model_dir = "models/coreml_simple/1"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.mlmodel")
    coreml_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Test the model
    print("\nTesting model...")
    test_input = {"input": example_input.numpy()}
    predictions = coreml_model.predict(test_input)
    print(f"Input shape: {example_input.shape}")
    print(f"Output shape: {predictions['output'].shape}")
    print(f"Sample output: {predictions['output'][0][:5]}...")
    
    return model_path

def create_image_classification_model():
    """Create a simple image classification CoreML model."""
    print("\nCreating image classification CoreML test model...")
    
    class SimpleImageClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleImageClassifier, self).__init__()
            # Simple CNN for 32x32 RGB images
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x
    
    # Create model
    model = SimpleImageClassifier(num_classes=10)
    model.eval()
    
    # Create example input (batch_size=1, channels=3, height=32, width=32)
    example_input = torch.rand(1, 3, 32, 32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 32, 32))],
        outputs=[ct.TensorType(name="scores")],
        minimum_deployment_target=ct.target.macOS11
    )
    
    # Create model directory
    model_dir = "models/coreml_image_classifier/1"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.mlmodel")
    coreml_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Test the model
    print("\nTesting image classification model...")
    test_input = {"image": example_input.numpy()}
    predictions = coreml_model.predict(test_input)
    print(f"Input shape: {example_input.shape}")
    print(f"Output shape: {predictions['scores'].shape}")
    print(f"Sample scores: {predictions['scores'][0][:5]}...")
    
    return model_path

def main():
    """Main function to create test models."""
    print("CoreML Test Model Generator")
    print("===========================")
    
    # Check if coremltools is installed
    try:
        import coremltools
        print(f"CoreML Tools version: {coremltools.__version__}")
    except ImportError:
        print("Error: coremltools is not installed.")
        print("Install with: pip install coremltools")
        return
    
    # Create models
    try:
        # Simple model
        simple_model_path = create_simple_coreml_model()
        
        # Image classification model
        image_model_path = create_image_classification_model()
        
        print("\nModels created successfully!")
        print("\nTo test with Triton, create config.pbtxt files in the model directories.")
        
    except Exception as e:
        print(f"Error creating models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()