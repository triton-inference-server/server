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

Triton includes beta versions of the version 2 Python and C++ client
libraries and examples. The libraries use the new HTTP/REST and GRPC
`KFServing protocols
<https://github.com/kubeflow/kfserving/docs/predict-api/v2>`_ and also
expose all the functionality expressed in the Triton `protocol
extensions
<https://github.com/NVIDIA/triton-inference-server/tree/master/docs/protocol>`_.

To try the new client libraries and examples, first follow directions
in :ref:`section-getting-the-client-libraries`.

Several `examples
<https://github.com/NVIDIA/triton-inference-server/tree/r20.03.1/src/clients/python/experimental_api_v2/examples>`_
demonstrate the new Python client library and the code is documented
in `grpcclient.py
<https://github.com/NVIDIA/triton-inference-server/blob/r20.03.1/src/clients/python/experimental_api_v2/library/grpcclient.py>`_
and `httpclient.py
<https://github.com/NVIDIA/triton-inference-server/blob/r20.03.1/src/clients/python/experimental_api_v2/library/httpclient.py>`_.

Similarly there are `C++ client examples
<https://github.com/NVIDIA/triton-inference-server/tree/r20.03.1/src/clients/c%2B%2B/experimental_api_v2/examples>`_
and documentation in `grpc_client.h and http_client.h
<https://github.com/NVIDIA/triton-inference-server/tree/r20.03.1/src/clients/c%2B%2B/experimental_api_v2/library>`_.

The examples that start with **grpc_** use the `protoc compiler to
generate the GRPC client stub <https://grpc.io/docs/guides/>`_, the
examples that start with **simple_grpc_** use the GRPC client library,
and the examples that start with **simple_http_** use the HTTP/REST
client library.

For the 20.03.1 release, for Triton to support the new HTTP/REST and
GRPC protocols the server must be run with the -\\-api-version=2
flag. This requirement will be removed in the 20.06 and later
releases.
