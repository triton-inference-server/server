<!--
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
# [Triton Inference Server Release 25.01](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-01.html#rel-25-01)

The Triton Inference Server container image, release 25.01, is available
on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) and
is open source
on [GitHub](https://github.com/triton-inference-server/server).

## Contents of the Triton Inference Server container

The [Triton Inference
Server](https://github.com/triton-inference-server/server) Docker image
contains the inference server executable and related shared libraries
in /opt/tritonserver.

For a complete list of what the container includes, refer to [Deep
Learning Frameworks Support
Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

The container also includes the following:

-   [Ubuntu 24.04](https://releases.ubuntu.com/24.04/) including [Python
    3.12](https://www.python.org/downloads/release/python-3120/)

-   [NVIDIA CUDA
    12.8.0.038](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

-   [NVIDIA cuBLAS
    12.8.3.14](https://docs.nvidia.com/cuda/cublas/index.html)

-   [cuDNN
    9.7.0.66](https://docs.nvidia.com/deeplearning/cudnn/release-notes/)

-   [NVIDIA NCCL
    2.25.1](https://docs.nvidia.com/deeplearning/nccl/release-notes/) (optimized
    for [NVIDIA NVLink](http://www.nvidia.com/object/nvlink.html)®)

-   [NVIDIA TensorRT™
    10.8.0.43](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)

-   OpenUCX 1.15.0

-   GDRCopy 2.4.1

-   NVIDIA HPC-X 2.21

-   OpenMPI 4.1.7

-   [FIL](https://github.com/triton-inference-server/fil_backend)

-   [NVIDIA DALI®
    1.45](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html)

-   [nvImageCodec
    0.2.0.7](https://docs.nvidia.com/cuda/nvimagecodec/release_notes_v0.2.0.html)

-   ONNX Runtime 1.20.1

-   Intel[ OpenVINO ](https://github.com/openvinotoolkit/openvino/tree/2022.1.0)2024.05.0

-   DCGM 3.3.6

-   [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) version [release/0.17.0](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.17.0)

-   [vLLM](https://github.com/vllm-project/vllm) version 0.6.3.1

## Driver Requirements

Release 25.01 is based on [CUDA
12.8.0.038](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) which
requires [NVIDIA
Driver](http://www.nvidia.com/Download/index.aspx?lang=en-us) release
560 or later. However, if you are running on a data center GPU (for
example, T4 or any other data center GPU), you can use NVIDIA driver
release 470.57 (or later R470), 525.85 (or later R525), 535.86 (or later
R535), or 545.23 (or later R545).

The CUDA driver\'s compatibility package only supports particular
drivers. Thus, users should upgrade from all R418, R440, R450, R460,
R510, R520, R530, R545 and R555 drivers, which are not
forward-compatible with CUDA 12.6. For a complete list of supported
drivers, see the [CUDA Application
Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package) topic.
For more information, see [CUDA Compatibility and
Upgrades](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-and-upgrades).

## GPU Requirements

Release 25.01 supports CUDA compute capability 7.5 and later. This
corresponds to GPUs in the NVIDIA Turing™, NVIDIA Ampere architecture,
NVIDIA Hopper™, NVIDIA Ada Lovelace, and NVIDIA Blackwell architecture
families. For a list of GPUs to which this compute capability
corresponds, see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus).
For additional support details, see [Deep Learning Frameworks Support
Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Key Features and Enhancements

This Inference Server release includes the following key features and
enhancements.

-   Starting with the 25.01 release, Triton Inference Server supports
    Blackwell GPU architectures.

-   Fixed a bug when passing the correlation ID of string type to
    python_backend. Added datatype checks to correlation ID values.

-   vLLM backend can now take advantage of the [vLLM
    v0.6](https://blog.vllm.ai/2024/09/05/perf-update.html) performance
    improvement by communicating with the vLLM engine via ZMQ.

-   GenAI-Perf now provides the exact input sequence length requested
    for synthetic text generation.

-   GenAI-Perf supports the creation of a prefix pool to emulate system
    prompts via \--num-system-prompts and \--system-prompt-length.

-   GenAI-Perf improved error visibility via returning more detailed
    errors when OpenAI frontends return an error or metric generation
    fails.

-   GenAI-Perf reports time-to-second-token and request count in its
    metrics.

-   GenAI-Perf allows the use of a custom tokenizer in its "compare"
    subcommand for comparing multiple profiles.

-   GenAI-Perf natively supports \--request-count for sending a specific
    number of requests and \--header for sending a list of headers with
    every request.

-   Model Analyzer functionality has been migrated to GenAI-Perf via the
    "analyze" subcommand, enabling the tool to sweep and find the
    optimal model configuration.

-   A bytes appending bug was fixed in GenAI-Perf, resulting in more
    accurate output sequence lengths for Triton.


## Known Issues

-   A segmentation fault related to DCGM and NSCQ may be encountered
    during server shutdown on NVSwitch systems. A possible workaround
    for this issue is to disable the collection of GPU metrics
    \`tritonserver \--allow-gpu-metrics false \...\`

-   vLLM backend currently does not take advantage of the [vLLM
    v0.6](https://blog.vllm.ai/2024/09/05/perf-update.html) performance
    improvement when metrics are enabled.

-   Please note, that the vllm version provided in 25.01 container is
    0.6.3.post1. Due to some issues with vllm library versioning,
    \`vllm.\_\_version\_\_\` displays \`0.6.3\`.

-   Incorrect results are known to occur when using TensorRT (TRT)
    Backend for inference using int8 data type for I/O on the Blackwell
    GPU architecture.

-   When running Torch TRT models, the output may differ from running
    the same model on a previous release.

-   When using TensorRT models, if auto-complete configuration is
    disabled and is_non_linear_format_io:true for reformat-free tensors
    is not provided in the model configuration, the model may not load
    successfully.

-   When using Python models in[decoupled
    mode](https://github.com/triton-inference-server/python_backend/tree/main?tab=readme-ov-file#decoupled-mode),
    users need to ensure that the ResponseSender goes out of scope or is
    properly cleaned up before unloading the model to guarantee that the
    unloading process executes correctly.

-   Restart support was temporarily removed for Python models.

-   The Triton Inference Server with vLLM backend currently does not
    support running vLLM models with tensor parallelism sizes greater
    than 1 and default \"distributed_executor_backend\" setting when
    using explicit model control mode. In attempt to load a vllm model
    (tp \> 1) in explicit mode, users could potentially see failure at
    the \`initialize\` step: \`could not acquire lock for
    \<\_io.BufferedWriter name=\'\<stdout\>\'\> at interpreter shutdown,
    possibly due to daemon threads\`. For the default model control
    mode, after server shutdown, vllm related sub-processes are not
    killed. Related vllm
    issue: <https://github.com/vllm-project/vllm/issues/6766> . Please
    specify distributed_executor_backend:ray in the model.json when
    deploying vllm models with tensor parallelism \> 1.

-   When loading models with file override, multiple model configuration
    files are not supported. Users must provide the model configuration
    by setting parameter config : \<JSON\> instead of custom
    configuration file in the following
    format: file:configs/\<model-config-name\>.pbtxt :
    \<base64-encoded-file-content\>.

-   TensorRT-LLM [backend](https://github.com/triton-inference-server/tensorrtllm_backend) provides
    limited support of Triton extensions and features.

-   The TensorRT-LLM backend may core dump on server shutdown. This
    impacts server teardown only and will not impact inferencing.

-   The Java CAPI is known to have intermittent segfaults.

-   Some systems which implement malloc() may not release memory back to
    the operating system right away causing a false memory leak. This
    can be mitigated by using a different malloc implementation.
    Tcmalloc and jemalloc are installed in the Triton container and can
    be [used by specifying the library in
    LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md).
    NVIDIA recommends experimenting with both tcmalloc and jemalloc to
    determine which one works better for your use case.

-   Auto-complete may cause an increase in server start time. To avoid a
    start time increase, users can provide the full model configuration
    and launch the server with \--disable-auto-complete-config.

-   Auto-complete does not support PyTorch models due to lack of
    metadata in the model. It can only verify that the number of inputs
    and the input names matches what is specified in the model
    configuration. There is no model metadata about the number of
    outputs and datatypes. Related PyTorch
    bug:<https://github.com/pytorch/pytorch/issues/38273>.

-   Triton Client PIP wheels for ARM SBSA are not available from PyPI
    and pip will install an incorrect Jetson version of Triton Client
    library for Arm SBSA. The correct client wheel file can be pulled
    directly from the Arm SBSA SDK image and manually installed.

-   Traced models in PyTorch seem to create overflows when int8 tensor
    values are transformed to int32 on the GPU. Refer
    to [pytorch/pytorch#66930](https://github.com/pytorch/pytorch/issues/66930) for
    more information.

-   Triton cannot retrieve GPU metrics with [MIG-enabled GPU
    devices](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#supported-gpus).

-   Triton metrics might not work if the host machine is running a
    separate DCGM agent on bare-metal or in a container.

-   When cloud storage (AWS, GCS, AZURE) is used as a model repository
    and a model has multiple versions, Triton creates an extra local
    copy of the cloud model's folder in the temporary directory, which
    is deleted upon server's shutdown.

-   Python backend support for Windows is limited and does not currently
    support the following features:

    -   GPU tensors

    -   CPU and GPU-related metrics

    -   Custom execution environments

    -   The model load/unload APIs
