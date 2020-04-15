..
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

.. _section-client-experimental:

Experimental Client
===================

Alpha versions of two Python client libraries is available that use
the new HTTP/REST and GRPC protocol based on the `community standard
inference protocols
<https://github.com/kubeflow/kfserving/docs/predict-api/v2>`_ that
have been proposed by the `KFServing project
<https://github.com/kubeflow/kfserving>`_. These Python libraries also
expose all the functionality expressed in the Triton `protocol
extensions
<https://docs.nvidia.com/deeplearning/sdk/triton-inference-server-master-branch-guide/docs/protocol>`_.

To try the new client libraries, first get the Python client library
that uses the new protocol following directions in
:ref:`section-getting-the-client-libraries`.

The Python GRPC client interface documentation is available at
`src/clients/python/experimental\_api\_v2/library/grpcclient.py
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/clients/python/experimental_api_v2/library/grpcclient.py>`_,
`src/clients/python/experimental\_api\_v2/library/httpclient.py
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/clients/python/experimental_api_v2/library/httpclient.py>`_
and in the API Reference.

Examples are available in
`src/clients/python/experimental\_api\_v2/examples
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/clients/python/experimental_api_v2/examples>`_. The
examples that start with **grpc_** use the `protoc compiler to
generate the GRPC client stub <https://grpc.io/docs/guides/>`_, the
examples that start with **simple_grpc_** use the Python GRPC client
library, and the examples that start with **simple_http_** use the
Python HTTP/REST client library.

For Triton to support the new HTTP/REST and GRPC protocols the server
must be run with the -\\-api-version=2 flag.
