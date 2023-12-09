<!--
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Testing Triton

Currently there is no CI testing enabled for Triton repositories. We
will enable CI testing in a future update.

However, there is a set of tests in the qa/ directory that can be run
manually to provide extensive testing. Before running these tests you
must first generate a few model repositories containing the models
needed by the tests.

## Generate QA Model Repositories

The QA model repositories contain some simple models that are used to
verify the correctness of Triton. To generate the QA model
repositories:

```
$ cd qa/common
$ ./gen_qa_model_repository
$ ./gen_qa_custom_ops
```

This will create multiple model repositories in /tmp/<version>/qa_*
(for example /tmp/23.12/qa_model_repository).  The TensorRT models
will be created for the GPU on the system that CUDA considers device 0
(zero). If you have multiple GPUs on your system see the documentation
in the scripts for how to target a specific GPU.

## Build SDK Image

Build the *tritonserver_sdk* image that contains the client
libraries, model analyzer, and examples using the following
commands. You must first checkout the <client branch> branch of the
*client* repo into the clientrepo/ subdirectory. Typically you want to
set <client branch> to be the same as your current server branch.

```
$ cd <server repo root>
$ git clone --single-branch --depth=1 -b <client branch> https://github.com/triton-inference-server/client.git clientrepo
$ docker build -t tritonserver_sdk -f Dockerfile.sdk .
```

## Build QA Image

Next you need to build a QA version of the Triton Docker image. This
image will contain Triton, the QA tests, and all the dependencies
needed to run the QA tests. First do a [Docker image
build](build.md#building-with-docker) to produce the
*tritonserver_cibase* and *tritonserver* images.

Then, build the actual QA image.

```
$ docker build -t tritonserver_qa -f Dockerfile.QA .
```

## Run QA Tests

Now run the QA image and mount the QA model repositories into the
container so the tests will be able to access them.

```
$ docker run --gpus=all -it --rm -v/tmp:/data/inferenceserver tritonserver_qa
```

Within the container the QA tests are in /opt/tritonserver/qa. To run
a test, change directory to the test and run the test.sh script.

```
$ cd <test directory>
$ bash -x ./test.sh
```

### Sanity Tests

Many tests require that you use a complete Triton build, with all
backends and other features enabled. There are three sanity tests that
are parameterized so that you can run them even if you have built a
Triton that contains only a subset of all supported Triton
backends. These tests are L0_infer, L0_batcher and
L0_sequence_batcher. For these tests the following envvars are
available to control how the tests behave:

* BACKENDS: Control which backends are tested. Look in the test.sh
  file of the test to see the default and allowed values.

* ENSEMBLES: Enable testing of ensembles. Set to "0" to disable, set
  to "1" to enable. If enabled you must have the *identity* backend
  included in your Triton build.

* EXPECTED_NUM_TESTS: The tests perform a check of the total number of
  test sub-cases. The exact number of sub-cases that run will depend
  on the values you use for BACKENDS and ENSEMBLES. So you will need
  to adjust this as appropriate for your testing.

For example, if you build a Triton that has only the TensorRT backend
you can run L0_infer as follows:

```
$ BACKENDS="plan" ENSEMBLES=0 EXPECTED_NUM_TESTS=<expected> bash -x ./test.sh
```

Where '\<expected\>' is the number of sub-tests expected to be run for
just TensorRT testing and no ensembles. Depending on which backend(s)
you are testing you will need to experiment and determine the correct
value for '\<expected\>'.
