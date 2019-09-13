..
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

|License|

Example Go Client
=================

This sample script assumes that the protobuf stubs will be compiled
local to the client, under the nvidia_inferenceserver directory. This
can be easily tweaked by uncommenting gen_go_stubs.sh to generate the
stubs in ${GOPATH}/src/nvidia_inferenceserver to have something more
global instead.

Usage::

  # Clone repo
  git clone https://github.com/NVIDIA/tensorrt-inference-server.git

  # Setup "simple" model from example model_repository
  cd tensorrt-inference-server/docs/examples
  ./fetch_models.sh

  # Launch (detached) TRTIS
  docker run -d -p8000:8000 -p8001:8001 -p8002:8002 -it -v $(pwd)/model_repository:/models nvcr.io/nvidia/tensorrtserver:19.07-py3 trtserver --model-store=/models

  # Use client
  cd ../../src/clients/go
  # Compiles *.proto to *.pb.go
  ./gen_go_stubs.sh
  go run grpc_simple_client.go

Sample Output::

  $ go run grpc_simple_client.go
  TRTIS Health - Live: true
  TRTIS Health - Ready: true
  request_status:<code:SUCCESS server_id:"inference:0" request_id:3 > server_status:<id:"inference:0" version:"1.4.0" ready_state:SERVER_READY uptime_ns:39273850004 model_status:<key:"simple" value:<config:<name:"simple" platform:"tensorflow_graphdef" version_policy:<latest:<num_versions:1 > > max_batch_size:8 input:<name:"INPUT0" data_type:TYPE_INT32 dims:16 > input:<name:"INPUT1" data_type:TYPE_INT32 dims:16 > output:<name:"OUTPUT0" data_type:TYPE_INT32 dims:16 > output:<name:"OUTPUT1" data_type:TYPE_INT32 dims:16 > instance_group:<name:"simple" kind:KIND_CPU count:1 > default_model_filename:"model.graphdef" > version_status:<key:1 value:<ready_state:MODEL_READY > > > > >

  Checking Inference Outputs
  --------------------------
  0 + 1 = 1
  0 - 1 = -1
  1 + 1 = 2
  1 - 1 = 0
  2 + 1 = 3
  2 - 1 = 1
  3 + 1 = 4
  3 - 1 = 2
  4 + 1 = 5
  4 - 1 = 3
  5 + 1 = 6
  5 - 1 = 4
  6 + 1 = 7
  6 - 1 = 5
  7 + 1 = 8
  7 - 1 = 6
  8 + 1 = 9
  8 - 1 = 7
  9 + 1 = 10
  9 - 1 = 8
  10 + 1 = 11
  10 - 1 = 9
  11 + 1 = 12
  11 - 1 = 10
  12 + 1 = 13
  12 - 1 = 11
  13 + 1 = 14
  13 - 1 = 12
  14 + 1 = 15
  14 - 1 = 13
  15 + 1 = 16
  15 - 1 = 14

.. |License| image:: https://img.shields.io/badge/License-BSD3-lightgrey.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
