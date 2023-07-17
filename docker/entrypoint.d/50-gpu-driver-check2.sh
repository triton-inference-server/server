#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if [[ "${NVIDIA_CPU_ONLY:-0}" == "1" ]]; then
  export TRITON_SERVER_CPU_ONLY=1
fi
