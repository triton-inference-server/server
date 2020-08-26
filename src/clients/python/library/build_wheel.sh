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
  mkdir -p ${WHLDIR}/tritonclient/
  touch ${WHLDIR}/tritonclient/__init__.py

  # grpcclient module
  if [ -f grpcclient.py ]; then
    mkdir -p ${WHLDIR}/tritonclient/grpcclient
    cp ../../../core/*_pb2.py \
      "${WHLDIR}/tritonclient/grpcclient/."
    cp ../../../core/*_grpc.py \
      "${WHLDIR}/tritonclient/grpcclient/."
    cp grpcclient.py \
      "${WHLDIR}/tritonclient/grpcclient/__init__.py"
    # Use 'sed' command to fix protoc compiled imports (see
    # https://github.com/google/protobuf/issues/1491).
    sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tritonclient.grpcclient import \1_pb2 as \2/" \
      ${WHLDIR}/tritonclient/grpcclient/*_pb2.py
    sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tritonclient.grpcclient import \1_pb2 as \2/" \
     ${WHLDIR}/tritonclient/grpcclient/*_pb2_grpc.py
  fi

  # httpclient module
  if [ -f httpclient.py ]; then
    mkdir -p ${WHLDIR}/tritonclient/httpclient
    cp httpclient.py \
      "${WHLDIR}/tritonclient/httpclient/__init__.py"
  fi

  # utility module
  mkdir -p ${WHLDIR}/tritonclient/utils
  cp utils.py \
   "${WHLDIR}/tritonclient/utils/__init__.py"

  if [ "$2" = true ] ; then
    # shared_memory
    mkdir -p ${WHLDIR}/tritonclient/shared_memory
    cp libcshm.so \
      "${WHLDIR}/tritonclient/shared_memory/."
    cp shared_memory/__init__.py \
      "${WHLDIR}/tritonclient/shared_memory/."

    if [ -f libccudashm.so ] && [ -f cuda_shared_memory/__init__.py ]; then
      mkdir -p ${WHLDIR}/tritonclient/cuda_shared_memory
      cp libccudashm.so \
        "${WHLDIR}/tritonclient/cuda_shared_memory/."
      cp cuda_shared_memory/__init__.py \
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
  if [ "$2" = true ] ; then
    cp x86_linux_setup.py "${WHLDIR}"
  else
    cp setup.py "${WHLDIR}"
  fi

  pushd "${WHLDIR}"
  echo $(date) : "=== Building wheel"
  if [ "$2" = true ] ; then
    VERSION=$VERSION python${PYVER} x86_linux_setup.py bdist_wheel
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
