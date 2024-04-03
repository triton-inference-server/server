<!--
# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Using Triton Inference Server as a shared library for execution on Jetson

## Overview
This project demonstrates how to run C API applications using Triton Inference Server as a shared library. We also show how to build and execute such applications on Jetson.

### Prerequisites

* JetPack >= 4.6
* OpenCV >= 4.1.1
* TensorRT >= 8.0.1.6

### Installation

Follow the installation instructions from the GitHub release page ([https://github.com/triton-inference-server/server/releases/](https://github.com/triton-inference-server/server/releases/)).

In our example, we placed the contents of downloaded release directory under `/opt/tritonserver`.

## Part 1. Concurrent inference and dynamic batching

The purpose of the sample located under [concurrency_and_dynamic_batching](concurrency_and_dynamic_batching/README.md)
is to demonstrate the important features of Triton Inference Server such as concurrent model execution and
dynamic batching. In order to do that, we implemented a people detection application using C API and Triton
Inference Server as a shared library.

## Part 2. Analyzing model performance with perf_analyzer

To analyze model performance on Jetson,
[perf_analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
tool is used. The `perf_analyzer` is included in the release tar file or can be
compiled from source.

From this directory of the repository, execute the following to evaluate model performance:

```shell
./perf_analyzer -m peoplenet -b 2 --service-kind=triton_c_api --model-repo=$(pwd)/concurrency_and_dynamic_batching/trtis_model_repo_sample_1 --triton-server-directory=/opt/tritonserver --concurrency-range 1:6 -f perf_c_api.csv
```

In the example above we saved the results as a `.csv` file. To visualize these
results, follow the steps described
[here](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md).
