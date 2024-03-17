#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

prodname_uc=$(echo "${NVIDIA_PRODUCT_NAME}" | tr [:lower:] [:upper:] | sed 's/ /_/g' | sed 's/^NVIDIA_//')  # Product name
_prodver="NVIDIA_${prodname_uc}_VERSION" # Container product version variable name
_compver="${prodname_uc}_VERSION"        # Upstream component version variable name

echo
echo "NVIDIA Release ${!_prodver} (build ${NVIDIA_BUILD_ID})"
[ -n "${!_compver}" ] && echo "${NVIDIA_PRODUCT_NAME} Version ${!_compver}"
