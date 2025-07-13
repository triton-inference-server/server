#!/usr/bin/env python3
import torch
import coremltools as ct
import numpy as np
from transformers import BertTokenizer
import time
import json
import matplotlib.pyplot as plt

print("🏃 Running Apple Silicon Transformer Benchmark")
print("=" * 60)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("models/tokenizer")

# Test texts of varying lengths
test_texts = [
    "Short text",
    "Medium length text that contains more words to process",
    "This is a longer text that will test the performance of the different compute units on Apple Silicon when processing natural language. The Apple Neural Engine is specifically optimized for transformer models."
]

# Load models
print("📦 Loading models...")
model_pt = torch.jit.load("models/bert_pytorch/1/model.pt")
model_pt.eval()

mlmodel_ane = ct.models.MLModel("models/bert_ane/1/model.mlpackage", 
                                compute_units=ct.ComputeUnit.ALL)
mlmodel_gpu = ct.models.MLModel("models/bert_metal/1/model.mlpackage", 
                                compute_units=ct.ComputeUnit.CPU_AND_GPU)
mlmodel_cpu = ct.models.MLModel("models/bert_metal/1/model.mlpackage", 
                                compute_units=ct.ComputeUnit.CPU_ONLY)

results = {
    "pytorch_cpu": [],
    "coreml_ane": [],
    "coreml_gpu": [],
    "coreml_cpu": []
}

# Warm-up runs
print("\n🔥 Warming up models...")
for _ in range(3):
    inputs = tokenizer(test_texts[0], return_tensors="pt", max_length=128, 
                      padding="max_length", truncation=True)
    with torch.no_grad():
        model_pt(inputs["input_ids"], inputs["attention_mask"])
    
    coreml_inputs = {
        "input_ids": inputs["input_ids"].numpy().astype(np.int32),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.int32)
    }
    mlmodel_ane.predict(coreml_inputs)
    mlmodel_gpu.predict(coreml_inputs)
    mlmodel_cpu.predict(coreml_inputs)

# Benchmark
print("\n⚡ Running benchmark...")
num_runs = 10

for text_idx, text in enumerate(test_texts):
    print(f"\nText {text_idx + 1} (length: {len(text)} chars)")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=128, 
                      padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    coreml_inputs = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32)
    }
    
    # PyTorch CPU
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            model_pt(input_ids, attention_mask)
        times.append((time.time() - start) * 1000)
    avg_time = np.mean(times[1:])  # Skip first run
    results["pytorch_cpu"].append(avg_time)
    print(f"  PyTorch CPU: {avg_time:.2f}ms")
    
    # CoreML ANE
    times = []
    for _ in range(num_runs):
        start = time.time()
        mlmodel_ane.predict(coreml_inputs)
        times.append((time.time() - start) * 1000)
    avg_time = np.mean(times[1:])
    results["coreml_ane"].append(avg_time)
    print(f"  CoreML ANE: {avg_time:.2f}ms")
    
    # CoreML GPU
    times = []
    for _ in range(num_runs):
        start = time.time()
        mlmodel_gpu.predict(coreml_inputs)
        times.append((time.time() - start) * 1000)
    avg_time = np.mean(times[1:])
    results["coreml_gpu"].append(avg_time)
    print(f"  CoreML GPU: {avg_time:.2f}ms")
    
    # CoreML CPU
    times = []
    for _ in range(num_runs):
        start = time.time()
        mlmodel_cpu.predict(coreml_inputs)
        times.append((time.time() - start) * 1000)
    avg_time = np.mean(times[1:])
    results["coreml_cpu"].append(avg_time)
    print(f"  CoreML CPU: {avg_time:.2f}ms")

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create visualization
print("\n📊 Creating performance charts...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
text_labels = ['Short', 'Medium', 'Long']
x = np.arange(len(text_labels))
width = 0.2

ax1.bar(x - 1.5*width, results["pytorch_cpu"], width, label='PyTorch CPU', color='#1f77b4')
ax1.bar(x - 0.5*width, results["coreml_cpu"], width, label='CoreML CPU', color='#ff7f0e')
ax1.bar(x + 0.5*width, results["coreml_gpu"], width, label='CoreML GPU', color='#2ca02c')
ax1.bar(x + 1.5*width, results["coreml_ane"], width, label='CoreML ANE', color='#d62728')

ax1.set_xlabel('Text Length')
ax1.set_ylabel('Inference Time (ms)')
ax1.set_title('BERT Inference Performance on Apple Silicon')
ax1.set_xticks(x)
ax1.set_xticklabels(text_labels)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Speedup chart
baseline = results["pytorch_cpu"]
speedups = {
    'CoreML CPU': [baseline[i]/results["coreml_cpu"][i] for i in range(len(baseline))],
    'CoreML GPU': [baseline[i]/results["coreml_gpu"][i] for i in range(len(baseline))],
    'CoreML ANE': [baseline[i]/results["coreml_ane"][i] for i in range(len(baseline))]
}

for i, (label, values) in enumerate(speedups.items()):
    ax2.plot(text_labels, values, marker='o', linewidth=2, markersize=8, label=label)

ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Text Length')
ax2.set_ylabel('Speedup vs PyTorch CPU')
ax2.set_title('Speedup Factor on Apple Silicon')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('apple_silicon_benchmark.png', dpi=150)
print("✅ Chart saved as apple_silicon_benchmark.png")

# Summary
print("\n📈 Performance Summary:")
print("=" * 60)
avg_speedups = {k: np.mean(v) for k, v in speedups.items()}
for backend, speedup in sorted(avg_speedups.items(), key=lambda x: x[1], reverse=True):
    print(f"{backend}: {speedup:.2f}x average speedup")

print("\n🎯 Best configuration:")
best_times = {
    'Short text': min([(k, v[0]) for k, v in results.items()], key=lambda x: x[1]),
    'Medium text': min([(k, v[1]) for k, v in results.items()], key=lambda x: x[1]),
    'Long text': min([(k, v[2]) for k, v in results.items()], key=lambda x: x[1])
}
for text_type, (backend, time) in best_times.items():
    print(f"  {text_type}: {backend} ({time:.2f}ms)")