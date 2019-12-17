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

Example Java/Kotlin Client
=================


Usage::
  
  # Clone repo
  git clone https://github.com/NVIDIA/tensorrt-inference-server.git

  # Setup "simple" model from example model_repository
  cd tensorrt-inference-server/docs/examples
  ./fetch_models.sh

  # Launch (detached) TRTIS
  docker run -d -p8000:8000 -p8001:8001 -p8002:8002 -it -v $(pwd)/model_repository:/models nvcr.io/nvidia/tensorrtserver:19.07-py3 trtserver --model-store=/models

  # Use client
  cd tensorrt-inference-server/src/clients/java
  ./gradlew run --args="--model simple --server-url localhost:8001 --batch-size 1"


Sample Output::

  [main] INFO nvidia.inferenceserver.client.Main - is live: true, is ready: true, contains model: true
  [main] INFO nvidia.inferenceserver.client.Main - Model inputs' sizes: [InputMeta(name=INPUT0, dims=[16], type=int (Kotlin reflection is not available)), InputMeta(name=INPUT1, dims=[16], type=int (Kotlin reflection is not available))]
  [main] INFO nvidia.inferenceserver.client.Main - input INPUT0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  [main] INFO nvidia.inferenceserver.client.Main - input INPUT1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  [main] INFO nvidia.inferenceserver.client.Main - result for output OUTPUT0: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  [main] INFO nvidia.inferenceserver.client.Main - result for output OUTPUT1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


.. |License| image:: https://img.shields.io/badge/License-BSD3-lightgrey.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
