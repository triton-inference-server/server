..
  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

Installing the Server
=====================

The TensorRT Inference Server is available as a pre-built Docker
container or you can :ref:`build it from source
<section-building>`.

.. _section-installing-prebuilt-containers:

Installing Prebuilt Containers
------------------------------

The inference server is provided as a pre-built container on the
`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_.  Before pulling the
container you must have access and be logged into the NGC container
registry as explained in the `NGC Getting Started Guide
<http://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html>`_.

Before you can pull a container from the NGC container registry, you
must have Docker and nvidia-docker installed. For DGX users, this is
explained in `Preparing to use NVIDIA Containers Getting Started Guide
<http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html>`_.
For users other than DGX, follow the `nvidia-docker installation
documentation <https://github.com/NVIDIA/nvidia-docker>`_ to install
the most recent version of CUDA, Docker, and nvidia-docker.

After performing the above setup, you can pull the TensorRT Inference
Server container using the following command::

  docker pull nvcr.io/nvidia/tensorrtserver:19.09-py3

Replace *19.09* with the version of inference server that you want to
pull.
