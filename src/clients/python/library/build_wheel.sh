#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

function main() {
  if [[ $# -lt 2 ]] ; then
    echo "usage: $0 <destination dir> <linux-specific> <perf-client-binary>"
    exit 1
  fi

  if [[ ! -f "VERSION" ]]; then
    echo "Could not find VERSION"
    exit 1
  fi

  VERSION=`cat VERSION`
  DEST="$1"
  WHLDIR="$DEST/wheel"

  echo $(date) : "=== Using builddir: ${WHLDIR}"

  echo "Adding package files"
  mkdir -p ${WHLDIR}/tritonclient
  touch ${WHLDIR}/tritonclient/__init__.py

  # Needed for the backwards-compatibility
  # Remove when moving completely to the
  # new structure.
  if [ -d tritonclientutils ]; then
    cp -r tritonclientutils ${WHLDIR}/
  fi
  if [ -d tritonhttpclient ]; then
    cp -r tritonhttpclient ${WHLDIR}/
  fi
  if [ -d tritongrpcclient ]; then
    cp -r tritongrpcclient ${WHLDIR}/
  fi
  if [ "$2" = true ] ; then
    if [ -d tritonshmutils ]; then
      cp -r tritonshmutils ${WHLDIR}/
      touch ${WHLDIR}/tritonshmutils/__init__.py
    fi
  fi
  ####################################

  if [ -f tritonclient/grpc.py ]; then
    cp tritonclient/grpc.py \
      "${WHLDIR}/tritonclient/."
    cp ../../../core/*_pb2.py \
      "${WHLDIR}/tritonclient/."
    cp ../../../core/*_grpc.py \
      "${WHLDIR}/tritonclient/."
    # Use 'sed' command to fix protoc compiled imports (see
    # https://github.com/google/protobuf/issues/1491).
    sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tritonclient import \1_pb2 as \2/" \
      ${WHLDIR}/tritonclient/*_pb2.py
    sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tritonclient import \1_pb2 as \2/" \
     ${WHLDIR}/tritonclient/*_pb2_grpc.py
  fi

  cp tritonclient/utils.py \
      "${WHLDIR}/tritonclient/."

  if [ -f tritonclient/http.py ]; then
    cp tritonclient/http.py \
      "${WHLDIR}/tritonclient/."
  fi

  if [ "$2" = true ] ; then
    mkdir -p ${WHLDIR}/tritonclient/shared_memory/
    cp tritonclient/shared_memory/__init__.py \
        "${WHLDIR}/tritonclient/shared_memory/."
    cp tritonclient/libcshm.so \
      "${WHLDIR}/tritonclient/shared_memory/."

    if [ -f tritonclient/libccudashm.so ] && [ -f tritonclient/cuda_shared_memory/__init__.py ]; then
      mkdir -p ${WHLDIR}/tritonclient/cuda_shared_memory
      cp tritonclient/cuda_shared_memory/__init__.py \
        "${WHLDIR}/tritonclient/cuda_shared_memory/."
      cp tritonclient/libccudashm.so \
        "${WHLDIR}/tritonclient/cuda_shared_memory/."
    fi
  
    # Copies the pre-compiled perf_client binary
    if [ -f $3 ]; then
      cp $3 "${WHLDIR}"
    fi
  fi
  
  cp LICENSE.txt "${WHLDIR}"
  cp README.md "${WHLDIR}"
  cp -r requirements "${WHLDIR}"
  cp setup.py "${WHLDIR}"

  pushd "${WHLDIR}"
  echo $(date) : "=== Building wheel"
  if [ "$2" = true ] ; then
    VERSION=$VERSION python${PYVER} setup.py bdist_wheel --plat-name=manylinux1_x86_64
  else
    VERSION=$VERSION python${PYVER} setup.py bdist_wheel
  fi
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  echo $(date) : "=== Output wheel file is in: ${DEST}"

  touch ${DEST}/stamp.whl
}

main "$@"
