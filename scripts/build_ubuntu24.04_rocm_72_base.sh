#!/bin/bash
#
# Build Ubuntu 24.04 + ROCm 7.2 base image

set -e

echo "========================================"
echo "Building localhost/ubuntu24.04_rocm7.2"
echo "========================================"
docker build -t localhost/ubuntu24.04_rocm7.2 -f Dockerfile.ubuntu24.04_rocm7.2 .
echo ""

docker images | grep -E "localhost/ubuntu24.04_rocm7.2" || true