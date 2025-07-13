#!/usr/bin/env python3
import os
import sys
import numpy as np

print("📦 Installing required packages...")
import subprocess

packages = ["transformers", "torch", "coremltools", "tokenizers"]
for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import torch
from transformers import BertModel, BertTokenizer
import coremltools as ct

print("📥 Downloading BERT model...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Create example input
max_length = 128
example_text = "Hello, this is a test of Apple Silicon optimizations!"
inputs = tokenizer(example_text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)

print("🔄 Converting to CoreML (for ANE)...")

# Wrap model to handle dictionary output
class BERTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output

wrapped_model = BERTWrapper(model)
wrapped_model.eval()

# Trace the model
traced_model = torch.jit.trace(wrapped_model, (inputs["input_ids"], inputs["attention_mask"]))

# Convert to CoreML with ANE optimization
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32)
    ],
    outputs=[
        ct.TensorType(name="last_hidden_state"),
        ct.TensorType(name="pooler_output")
    ],
    compute_units=ct.ComputeUnit.ALL,  # Use ANE when available
    convert_to="mlprogram"
)

# Save the model
os.makedirs("models/bert_ane/1", exist_ok=True)
mlmodel.save("models/bert_ane/1/model.mlpackage")

print("✅ CoreML model saved (optimized for ANE)")

# Also save PyTorch version
print("💾 Saving PyTorch version...")
torch.jit.save(traced_model, "models/bert_pytorch/1/model.pt")

# Save tokenizer
tokenizer.save_pretrained("models/tokenizer")

print("✅ Models ready!")
