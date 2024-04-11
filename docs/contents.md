<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

```{toctree}
:maxdepth: 1
:caption: Getting Started

getting_started/quickstart
```

```{toctree}
:maxdepth: 1
:caption: User Guide

user_guide/performance_tuning
user_guide/architecture
user_guide/model_repository
customization_guide/repository_agents
user_guide/model_configuration
user_guide/request_cancellation
user_guide/optimization
user_guide/ragged_batching
user_guide/rate_limiter
user_guide/model_analyzer
user_guide/model_management
user_guide/custom_operations
user_guide/decoupled_models
user_guide/response_cache
user_guide/metrics
user_guide/trace
user_guide/jetson
user_guide/v1_to_v2
customization_guide/deploy
```

```{toctree}
:maxdepth: 1
:caption: Debugging

user_guide/debugging_guide
user_guide/faq
```

```{toctree}
:maxdepth: 1
:caption: Protocol Guides

protocol/README
customization_guide/inference_protocols
protocol/extension_binary_data
protocol/extension_classification
protocol/extension_generate
protocol/extension_logging
protocol/extension_model_configuration
protocol/extension_model_repository
protocol/extension_schedule_policy
protocol/extension_sequence
protocol/extension_shared_memory
protocol/extension_statistics
protocol/extension_trace
protocol/extension_parameters
```

```{toctree}
:maxdepth: 1
:caption: Customization Guide

customization_guide/build
customization_guide/compose
customization_guide/test
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/jetson/README
examples/jetson/concurrency_and_dynamic_batching/README
```

```{toctree}
:maxdepth: 1
:caption: Client

client/README
_reference/tritonclient_api.rst
client/src/java/README
client/src/grpc_generated/go/README
client/src/grpc_generated/javascript/README
client/src/grpc_generated/java/README
```

```{toctree}
:maxdepth: 1
:caption: Performance Analyzer

client/src/c++/perf_analyzer/README
client/src/c++/perf_analyzer/docs/README
client/src/c++/perf_analyzer/docs/install
client/src/c++/perf_analyzer/docs/quick_start
client/src/c++/perf_analyzer/docs/cli
client/src/c++/perf_analyzer/docs/inference_load_modes
client/src/c++/perf_analyzer/docs/input_data
client/src/c++/perf_analyzer/docs/measurements_metrics
client/src/c++/perf_analyzer/docs/benchmarking
client/src/c++/perf_analyzer/genai-perf/README
client/src/c++/perf_analyzer/genai-perf/examples/tutorial
```

```{toctree}
:maxdepth: 1
:caption: Python Backend

python_backend/README
python_backend/inferentia/README
python_backend/examples/auto_complete/README
python_backend/examples/bls/README
python_backend/examples/bls_decoupled/README
python_backend/examples/custom_metrics/README
python_backend/examples/decoupled/README
python_backend/examples/instance_kind/README
python_backend/examples/jax/README
python_backend/examples/preprocessing/README
```
