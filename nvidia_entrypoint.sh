#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e
cat <<EOF

=============================
== Triton Inference Server ==
=============================

NVIDIA Release ${NVIDIA_TRITON_SERVER_VERSION} (build ${NVIDIA_BUILD_ID})

Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
EOF

if [[ "$(find -L /usr -name libcuda.so.1 | grep -v "compat") " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use Docker with NVIDIA Container Toolkit to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker."
  export TRITON_SERVER_CPU_ONLY=1
else
  ( /usr/local/bin/checkSMVER.sh )
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*(.[0-9]*)?$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%%.*}" -lt "${CUDA_DRIVER_VERSION%%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected.  Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      sleep 2
    fi
  fi
fi

avx=1
if ! cat /proc/cpuinfo | grep flags | sort -u | grep avx >& /dev/null; then
  avx=0
fi

isarm=$(uname -m | grep -c aarch64 >& /dev/null)

if [ ! avx ] && [ ! isarm ]; then
  echo
  echo "ERROR: This container was built for CPUs supporting at least the AVX instruction set, but"
  echo "       the CPU detected was $(cat /proc/cpuinfo |grep "model name" | sed 's/^.*: //' | sort -u), which does not report"
  echo "       support for AVX.  An Illegal Instrution exception at runtime is likely to result."
  echo "       See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX ."
  sleep 2
fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
