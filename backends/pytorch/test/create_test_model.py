#!/usr/bin/env python3
"""
Create a simple PyTorch model for testing the backend
"""

import torch
import torch.nn as nn
import os

class SimpleAddSubModel(nn.Module):
    """
    Simple model that returns (input1 + input2, input1 - input2)
    """
    def __init__(self):
        super(SimpleAddSubModel, self).__init__()
    
    def forward(self, input1, input2):
        return input1 + input2, input1 - input2

def main():
    # Create the model
    model = SimpleAddSubModel()
    model.eval()
    
    # Create example inputs for tracing
    example_input1 = torch.randn(1, 3, 224, 224)
    example_input2 = torch.randn(1, 3, 224, 224)
    
    # Trace the model
    traced_model = torch.jit.trace(model, (example_input1, example_input2))
    
    # Save the model
    output_path = os.path.join(os.path.dirname(__file__), "models/pytorch_simple/1/model.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model.save(output_path)
    print(f"Model saved to: {output_path}")
    
    # Test the saved model
    loaded_model = torch.jit.load(output_path)
    test_input1 = torch.tensor([[1.0, 2.0, 3.0]])
    test_input2 = torch.tensor([[0.5, 1.0, 1.5]])
    
    output = loaded_model(test_input1, test_input2)
    print(f"Test input1: {test_input1}")
    print(f"Test input2: {test_input2}")
    print(f"Output (add): {output[0]}")
    print(f"Output (sub): {output[1]}")

if __name__ == "__main__":
    main()