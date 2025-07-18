#!/usr/bin/env python3
"""
Run transformer model demo using Python API directly
This bypasses the need for the tritonserver binary
"""

import os
import sys
import time
import numpy as np

# First, let's run our standalone transformer test
print("🚀 Running Apple Silicon Transformer Demo")
print("=" * 50)

# Check if models exist
if not os.path.exists("models/bert_ane/1/model.mlpackage"):
    print("❌ Models not found. Running model conversion first...")
    os.system("python3 convert_bert_to_coreml.py")
else:
    print("✅ Models already converted")

print("\n📊 Running performance benchmark...")
print("-" * 50)

# Run the transformer test script
os.system("python3 test_transformer.py")

print("\n📈 Generating performance charts...")
print("-" * 50)

# Generate performance visualization
os.system("python3 generate_performance_charts.py")

print("\n✅ Demo complete! Check the generated PNG/PDF files for performance charts.")
print("\n💡 To monitor real-time performance, run in another terminal:")
print("   ./scripts/apple-silicon/monitor_apple_silicon.sh")