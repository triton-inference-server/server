<!--
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Triton Python Client Library

This package holds the triton python client library. The generic
installation is different from the linux-specific installation. The components
included within generic tritonclient package are:
- http
- grpc [ `service_pb2`, `service_pb2_grpc`, `model_config_pb2` ]
- utils [ x86-linux distribution will include `shared_memory` and `cuda_shared_memory`]

Apart from the above library, x86-linux distribution will include a
pre-compiled perf_analyzer binary For further information regarding
perf_analyzer, refer to docs
[here](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.rst).

Note the following packages are deprecated and will be removed in future releases:
- tritongrpcclient
- tritonhttpclient
- tritonclientutils
- tritonshmutils


## Installation

Install dependencies needed by the perf_analyzer binary shipped with the
package. This step can be skipped if perf_analyzer is not needed.

```bash
sudo apt update
sudo apt install libb64-dev
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to
install triton client library. A recent version of pip is required for
the install.

```bash
pip install nvidia-pyindex
pip install tritonclient[all]
```

There are two optional packages available, namely, `grpc` and
`http`. [`all`] installs dependencies for both the protocols.

The perf_analyzer is installed in the system configuration specific
`bin` directory.

## Usage

Refer to the examples
[here](https://github.com/triton-inference-server/server/tree/master/src/clients/python/examples)
for the usage of library.

Use the following command to get the detailed usage for perf_analyzer
binary.

```bash
perf_analyzer -h
```
