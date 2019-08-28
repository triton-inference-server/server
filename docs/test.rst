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

Testing
=======

Currently there is no CI testing enabled for the open-source version
of the TensorRT Inference Server. We will enable CI testing in a
future update.

However, there is a set of tests in the qa/ directory that can be run
manually to provide extensive testing. Before running these tests you
must first generate a few model repositories containing the models
needed by the tests.

Generate QA Model Repositories
------------------------------

The QA model repositories contain some simple models that are used to
verify the correctness of the inference server. To generate the QA
model repositories::

  $ cd qa/common
  $ ./gen_qa_model_repository
  $ ./gen_qa_custom_ops

This will create multiple model repositories in /tmp/qa_* (for example
/tmp/qa_model_repository).  The TensorRT models will be created for
the GPU on the system that CUDA considers device 0 (zero). If you have
multiple GPUs on your system see the documentation in the scripts for
how to target a specific GPU.

Build QA Container
------------------

Next you need to build a QA version of the inference server
container. This container will contain the inference server, the QA
tests, and all the dependencies needed to run the QA tests. You must
first build the tensorrtserver_client, tensorrtserver_cbe,
tensorrtserver_build and tensorrtserver containers as described in
:ref:`section-getting-the-client-libraries-and-examples` and
:ref:`section-building` and then build the QA container::

  $ docker build -t tensorrtserver_qa -f Dockerfile.QA .

Run QA Container
----------------

Now run the QA container and mount the QA model repositories into the
container so the tests will be able to access them::

  $ nvidia-docker run -it --rm -v/tmp:/data/inferenceserver tensorrtserver_qa

Within the container the QA tests are in /opt/tensorrtserver/qa. To run a test::

  $ cd <test directory>
  $ ./test.sh
