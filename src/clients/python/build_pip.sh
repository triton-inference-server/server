#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

function main() {
  if [[ $# -lt 1 ]] ; then
    echo "usage: $0 <destination dir>"
    exit 1
  fi

  if [[ ! -d "bazel-bin/src/clients/python" ]]; then
    echo "Could not find bazel-bin/src/clients/python"
    exit 1
  fi

  if [[ ! -f "VERSION" ]]; then
    echo "Could not find VERSION"
    exit 1
  fi

  VERSION=`cat VERSION`
  DEST="$1"
  TMPDIR="$(mktemp -d)"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  mkdir -p ${TMPDIR}/tensorrtserver/api

  echo "Adding package files"
  cp bazel-genfiles/src/core/*_pb2.py \
    "${TMPDIR}/tensorrtserver/api/."

  cp bazel-genfiles/src/core/*_grpc.py \
    "${TMPDIR}/tensorrtserver/api/."

  cp bazel-bin/src/clients/python/libcrequest.so \
    "${TMPDIR}/tensorrtserver/api/."

  cp src/clients/python/__init__.py \
    "${TMPDIR}/tensorrtserver/api/."

  cp src/clients/python/setup.py "${TMPDIR}"
	touch ${TMPDIR}/tensorrtserver/__init__.py

  # Use 'sed' command to fix protoc compiled imports (see
  # https://github.com/google/protobuf/issues/1491).
	sed -i "s/^from src\.core import \([^ ]*\)_pb2 as \([^ ]*\)$/from tensorrtserver.api import \1_pb2 as \2/" \
    ${TMPDIR}/tensorrtserver/api/*_pb2.py
	sed -i "s/^from src\.core import \([^ ]*\)_pb2 as \([^ ]*\)$/from tensorrtserver.api import \1_pb2 as \2/" \
    ${TMPDIR}/tensorrtserver/api/*_pb2_grpc.py

  pushd "${TMPDIR}"
  echo $(date) : "=== Building wheel"
  VERSION=$VERSION python${PYVER} setup.py bdist_wheel # >/dev/null
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  rm -rf "${TMPDIR}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
