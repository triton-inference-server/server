#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
  if [[ $# -lt 1 ]] ; then
    echo "usage: $0 <destination dir>"
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
  mkdir -p ${WHLDIR}/tensorrtserver/api

  echo "Adding package files"
  cp ../../core/*_pb2.py \
    "${WHLDIR}/tensorrtserver/api/."

  cp ../../core/*_grpc.py \
    "${WHLDIR}/tensorrtserver/api/."

  # library are compiled in different location and extension based on OS
  if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    cp libcrequest.so \
      "${WHLDIR}/tensorrtserver/api/."
    cp ../c++/librequest.so \
      "${WHLDIR}/tensorrtserver/api/."
  else
    cp Release/crequest.dll \
      "${WHLDIR}/tensorrtserver/api/."
    cp ../c++/Release/request.dll \
      "${WHLDIR}/tensorrtserver/api/."
  fi

  cp __init__.py \
    "${WHLDIR}/tensorrtserver/api/."

  cp setup.py "${WHLDIR}"
	touch ${WHLDIR}/tensorrtserver/__init__.py

  # Use 'sed' command to fix protoc compiled imports (see
  # https://github.com/google/protobuf/issues/1491).
	sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tensorrtserver.api import \1_pb2 as \2/" \
    ${WHLDIR}/tensorrtserver/api/*_pb2.py
	sed -i "s/^import \([^ ]*\)_pb2 as \([^ ]*\)$/from tensorrtserver.api import \1_pb2 as \2/" \
    ${WHLDIR}/tensorrtserver/api/*_pb2_grpc.py

  pushd "${WHLDIR}"
  echo $(date) : "=== Building wheel"
  VERSION=$VERSION python${PYVER} setup.py bdist_wheel
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  echo $(date) : "=== Output wheel file is in: ${DEST}"

	touch ${DEST}/stamp.whl
}

main "$@"
