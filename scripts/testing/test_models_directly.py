#!/usr/bin/env python3
import torch
import coremltools as ct
import numpy as np
from transformers import BertTokenizer
import time

print("🧪 Testing Models Directly")
print("=" * 50)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("models/tokenizer")
test_text = "Apple Silicon optimizations make machine learning incredibly fast!"

# Tokenize
inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
input_ids = inputs["input_ids"].numpy().astype(np.int32)
attention_mask = inputs["attention_mask"].numpy().astype(np.int32)

print(f"Input text: {test_text}")
print(f"Input shape: {input_ids.shape}")

# Test PyTorch model
print("\n1️⃣ Testing PyTorch Model...")
try:
    model_pt = torch.jit.load("models/bert_pytorch/1/model.pt")
    model_pt.eval()
    
    start = time.time()
    with torch.no_grad():
        outputs = model_pt(torch.tensor(input_ids), torch.tensor(attention_mask))
    pt_time = time.time() - start
    
    print(f"✅ PyTorch inference time: {pt_time*1000:.2f}ms")
    print(f"   Output shapes: {outputs[0].shape}, {outputs[1].shape}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

# Test CoreML ANE model
print("\n2️⃣ Testing CoreML ANE Model...")
try:
    mlmodel_ane = ct.models.MLModel("models/bert_ane/1/model.mlpackage")
    
    # Prepare inputs for CoreML
    coreml_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    start = time.time()
    outputs = mlmodel_ane.predict(coreml_inputs)
    ane_time = time.time() - start
    
    print(f"✅ ANE inference time: {ane_time*1000:.2f}ms")
    print(f"   Output keys: {list(outputs.keys())}")
    
    # Check if ANE was used
    compute_unit = mlmodel_ane.get_spec().description.metadata.userDefined.get("ComputeUnit", "Unknown")
    print(f"   Compute unit: {compute_unit}")
except Exception as e:
    print(f"❌ CoreML ANE error: {e}")

# Test CoreML Metal model (same model, different runtime config)
print("\n3️⃣ Testing CoreML Metal Model...")
try:
    # Force GPU usage
    mlmodel_gpu = ct.models.MLModel("models/bert_metal/1/model.mlpackage", 
                                    compute_units=ct.ComputeUnit.CPU_AND_GPU)
    
    start = time.time()
    outputs = mlmodel_gpu.predict(coreml_inputs)
    gpu_time = time.time() - start
    
    print(f"✅ Metal GPU inference time: {gpu_time*1000:.2f}ms")
except Exception as e:
    print(f"❌ CoreML Metal error: {e}")

# Performance comparison
print("\n📊 Performance Summary:")
print("=" * 50)
if 'pt_time' in locals() and 'ane_time' in locals():
    speedup = pt_time / ane_time
    print(f"ANE speedup over CPU: {speedup:.2f}x")
if 'pt_time' in locals() and 'gpu_time' in locals():
    speedup = pt_time / gpu_time  
    print(f"Metal speedup over CPU: {speedup:.2f}x")
if 'ane_time' in locals() and 'gpu_time' in locals():
    ratio = ane_time / gpu_time
    print(f"ANE vs Metal ratio: {ratio:.2f}x")