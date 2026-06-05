#!/bin/bash
# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

get_shm_pages() {
  shm_pages=(`ls /dev/shm`)
  echo ${#shm_pages[@]}
}

install_conda() {
  rm -rf ./miniconda
  file_name="Miniconda3-py312_24.9.2-0-Linux-x86_64.sh"
  wget https://repo.anaconda.com/miniconda/$file_name

  # install miniconda in silent mode
  bash $file_name -p ./miniconda -b

  # activate conda
  eval "$(./miniconda/bin/conda shell.bash hook)"
}

install_build_deps_apt() {
  apt update && apt install software-properties-common rapidjson-dev -y
  # Using CMAKE installation instruction from:: https://apt.kitware.com/
  apt update -q=2 \
    && apt install -y gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -q=2 \
    && apt-get install -y --no-install-recommends cmake=3.28.3* cmake-data=3.28.3*
}

install_build_deps_yum() {
  yum install rapidjson-devel -y
}

install_build_deps() {
  if [[ ${TRITON_RHEL} -eq "1" ]]; then
    install_build_deps_yum
  else
    install_build_deps_apt
  fi
}

create_conda_env() {
  local python_version=$1
  local env_name=$2
  conda create -n $env_name python=$python_version -y
  conda activate $env_name
  conda install -c conda-forge conda-pack -y
}

create_conda_env_with_specified_path() {
  local python_version=$1
  local env_path=$2
  conda create -p $env_path python=$python_version -y
  conda activate $env_path
  conda install -c conda-forge conda-pack -y
}

create_python_backend_stub() {
  rm -rf python_backend
  git clone ${TRITON_REPO_ORGANIZATION}/python_backend -b $PYTHON_BACKEND_REPO_TAG
  (cd python_backend/ && mkdir builddir && cd builddir && \
  cmake -DTRITON_ENABLE_GPU=ON -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION} -DTRITON_BACKEND_REPO_TAG=$TRITON_BACKEND_REPO_TAG -DTRITON_COMMON_REPO_TAG=$TRITON_COMMON_REPO_TAG -DTRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG -DPYBIND11_PYTHON_VERSION=$PY_VERSION ../ && \
  make -j18 triton-python-backend-stub)
}
