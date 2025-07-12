#!/bin/bash

echo "Building private buffer operations test..."

# Compile with clang++ directly for simplicity
clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
  -I. -I.. \
  test_private_buffer_ops.mm \
  metal_memory_manager.mm \
  metal_device.mm \
  metal_command.mm \
  -o test_private_buffer_ops

if [ $? -eq 0 ]; then
  echo "Build successful!"
  echo "Running test..."
  echo
  ./test_private_buffer_ops
else
  echo "Build failed!"
  exit 1
fi