#!/bin/bash
#
# Build Debian 12 + ROCm 7.2 + python 3.11 + vLLM image chain:
#   1. localhost/debian12_rocm7.2          (minimal ROCm base)
#   2. localhost/debian12_rocm7.2_vllm_base (PyTorch, Flash Attention, aiter, etc.)
#   3. localhost/debian12_rocm7.2_vllm     (vLLM wheel and install)
#

set -e

# Get script directory and run from it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Step 1: Build localhost/debian12_rocm7.2 ---
echo "========================================"
echo "Step 1/3: Building localhost/debian12_rocm7.2"
echo "========================================"
docker build -t localhost/debian12_rocm7.2 -f Dockerfile.debian12_rocm7.2 .
echo ""

# --- Step 2: Build localhost/debian12_rocm7.2_vllm_base ---
echo "========================================"
echo "Step 2/3: Building localhost/debian12_rocm7.2_vllm_base"
echo "========================================"
docker build -t localhost/debian12_rocm7.2_vllm_base -f Dockerfile.debian12_rocm7.2_vllm_base .
echo ""

# --- Step 3: Build localhost/debian12_rocm7.2_vllm (vLLM from git) ---
echo "========================================"
echo "Step 3/3: Building localhost/debian12_rocm7.2_vllm (REMOTE_VLLM=1)"
echo "========================================"
docker build -t localhost/debian12_rocm7.2_vllm \
  --build-arg REMOTE_VLLM=1 \
  --build-arg BASE_IMAGE=localhost/debian12_rocm7.2_vllm_base \
  -f Dockerfile.debian12_rocm7.2_vllm \
  .
echo ""

echo "========================================"
echo "✓ All builds complete!"
echo "========================================"
echo "Images:"
echo "  localhost/debian12_rocm7.2"
echo "  localhost/debian12_rocm7.2_vllm_base"
echo "  localhost/debian12_rocm7.2_vllm"
echo ""
docker images | grep -E "localhost/debian12_rocm7.2|localhost/debian12_rocm7.2" || true