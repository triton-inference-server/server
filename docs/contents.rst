..
.. Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions
.. are met:
..  * Redistributions of source code must retain the above copyright
..    notice, this list of conditions and the following disclaimer.
..  * Redistributions in binary form must reproduce the above copyright
..    notice, this list of conditions and the following disclaimer in the
..    documentation and/or other materials provided with the distribution.
..  * Neither the name of NVIDIA CORPORATION nor the names of its
..    contributors may be used to endorse or promote products derived
..    from this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
.. EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
.. PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
.. OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
.. (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
.. OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

.. toctree::
   :hidden:

   Home <introduction/index.md>
   Release notes <introduction/release_notes.md>
   Compatibility matrix <introduction/compatibility.md>

.. toctree::
   :hidden:
   :caption: Getting Started

   getting_started/quick_deployment_by_backend
   LLM With TRT-LLM <getting_started/trtllm_user_guide.md>
   Multimodal model <../tutorials/Popular_Models_Guide/Llava1.5/llava_trtllm_guide.md>
   Stable diffusion <../tutorials/Popular_Models_Guide/StableDiffusion/README.md>

.. toctree::
   :hidden:
   :caption: Scaling guide

   Multi-Node (AWS) <../tutorials/Deployment/Kubernetes/EKS_Multinode_Triton_TRTLLM/README.md>
   Multi-Instance <../tutorials/Deployment/Kubernetes/TensorRT-LLM_Autoscaling_and_Load_Balancing/README.md>

.. toctree::
   :hidden:
   :caption: LLM Features

   Constrained Decoding <../tutorials/Feature_Guide/Constrained_Decoding/README.md>
   Function Calling <../tutorials/Feature_Guide/Function_Calling/README.md>
   llm_features/speculative_decoding_by_backend_type

.. toctree::
   :hidden:
   :caption: Client

   client_guide/api_reference
   client_guide/in_process
   Client Libraries <client/README>
   _reference/tritonclient_api.rst

.. toctree::
   :hidden:
   :caption: Server

   Model_execution <user_guide/model_execution.md>
   Scheduler <user_guide/scheduler.md>
   Batcher <user_guide/batcher.md>
   server_guide/model_pipelines
   server_guide/state_management
   Request Cancellation <user_guide/request_cancellation.md>
   Rate Limiter <user_guide/rate_limiter.md>
   Caching <user_guide/response_cache.md>
   Metrics <user_guide/metrics.md>
   Tracing <user_guide/trace.md>

.. toctree::
   :hidden:
   :caption: Model Management


   Repository <user_guide/model_repository>
   Configuration <user_guide/model_configuration>
   Optimization <user_guide/optimization>
   Controls <user_guide/model_management>
   Decoupled models <user_guide/decoupled_models>
   Custom operators <user_guide/custom_operations>

.. toctree::
   :hidden:
   :caption: Backends

   TRT-LLM <tensorrtllm_backend/README>
   vLLM <backend_guide/vllm>
   Python <python_backend/README>
   Pytorch <pytorch_backend/README>
   ONNX Runtime <onnxruntime_backend/README>
   TensorFlow <tensorflow_backend/README>
   TensorRT <tensorrt_backend/README>
   FIL <fil_backend/README>
   DALI <dali_backend/README>
   Custom <backend/README>

.. toctree::
   :hidden:
   :caption: Perf benchmarking and tuning

   GenAI Perf Analyzer <perf_benchmark/genai_perf>
   Performance Analyzer <perf_benchmark/perf_analyzer>
   Model Analyzer <perf_benchmark/model_analyzer>
   Model Navigator <model_navigator/README>

.. toctree::
   :hidden:
   :caption: Debugging

   Guide <user_guide/debugging_guide>
