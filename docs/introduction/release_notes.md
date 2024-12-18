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
# [Triton Inference Server Release 24.10](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-10.html#rel-24-10)

The Triton Inference Server container image, release 24.10, is available on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) and is open source on [GitHub](https://github.com/triton-inference-server/server).


## **Contents of the Triton Inference Server container**

The [Triton Inference Server](https://github.com/triton-inference-server/server) Docker image contains the inference server executable and related shared libraries in `/opt/tritonserver`.

For a complete list of what the container includes, refer to [Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

The container also includes the following:

- [Ubuntu 22.04](http://releases.ubuntu.com/22.04/) including [Python 3.10](https://www.python.org/downloads/release/python-3100/)

- [NVIDIA CUDA 12.6.2](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

- [NVIDIA cuBLAS 12.6.3.3](https://docs.nvidia.com/cuda/cublas/index.html)

- [cuDNN 9.5.0.50](https://docs.nvidia.com/deeplearning/cudnn/release-notes/)

- [NVIDIA NCCL 2.22.3](https://docs.nvidia.com/deeplearning/nccl/release-notes/) (optimized for [NVIDIA NVLink](http://www.nvidia.com/object/nvlink.html)®)

- [NVIDIA TensorRT™ 10.5.0.18](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)

- OpenUCX 1.15.0

- GDRCopy 2.3

- NVIDIA HPC-X 2.20

- OpenMPI 4.1.7

- [FIL](https://github.com/triton-inference-server/fil_backend)

- [NVIDIA DALI® 1.42](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html)

- [nvImageCodec 0.2.0.7](https://docs.nvidia.com/cuda/nvimagecodec/release_notes_v0.2.0.html)

- ONNX Runtime 1.19.2

- Intel[ OpenVINO ](https://github.com/openvinotoolkit/openvino/tree/2022.1.0)2024.0.0

- DCGM 3.2.6

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) version [release/0.13.0](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.13.0)

- [vLLM](https://github.com/vllm-project/vllm) version 0.5.3 post 1


## **Driver Requirements**

Release 24.10 is based on [CUDA 12.6.2](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) which requires [NVIDIA Driver](http://www.nvidia.com/Download/index.aspx?lang=en-us) release 560 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 470.57 (or later R470), 525.85 (or later R525), 535.86 (or later R535), or 545.23 (or later R545).

The CUDA driver's compatibility package only supports particular drivers. Thus, users should upgrade from all R418, R440, R450, R460, R510, R520, R530, R545 and R555 drivers, which are not forward-compatible with CUDA 12.6. For a complete list of supported drivers, see the [CUDA Application Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package) topic. For more information, see [CUDA Compatibility and Upgrades](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-and-upgrades).


## **GPU Requirements**

Release 24.10 supports CUDA compute capability 6.0 and later. This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, NVIDIA Hopper™, and NVIDIA Ada Lovelace architecture families. For a list of GPUs to which this compute capability corresponds, see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus). For additional support details, see [Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).


## **Key Features and Enhancements**

This Inference Server release includes the following key features and enhancements.

- Optimized vLLM performance with custom metrics.

## **Known Issues**
- Numpy 2.x is not currently supported for Python Backend models and may cause them to return empty tensors unxpectedly, please use Numpy 1.x until support is added.
- To build the Llama 3.1 engine inside the 24.09-trtllm-python-py3 image, make sure to upgrade the transformer library to 4.43+ due to the bug in 4.43.x. One option to do so is to run `pip install -U transformers`. For more information, please refer to the discussion: https://github.com/NVIDIA/TensorRT-LLM/issues/2121.
- Triton vLLM container comes with the vLLM version, which has a known vulnerability: https://github.com/advisories/GHSA-w2r7-9579-27hf. Note, that the affected code is not invoked at runtime, therefore the Triton vLLM container is not affected by this issue.
- When running Torch TRT models, the output may differ from running the same model on a previous release.
- When using TensorRT models, if auto-complete configuration is disabled and is_non_linear_format_io:true for reformat-free tensors is not provided in the model configuration, the model may not load successfully.
- When using Python models indecoupled mode, users need to ensure that the ResponseSender goes out of scope or is properly cleaned up before unloading the model to guarantee that the unloading process executes correctly.
- Restart support was temporarily removed for Python models.
- Triton TensorRT-LLM Backend container image uses TensorRT-LLM version 0.14.0 and built out of nvcr.io/nvidia/tritonserver:24.07-py3-min. Please refer to the Triton TRT-LLM - - Container Support Matrix section in the GitHub release note for more details.
- The Triton Inference Server with vLLM backend currently does not support running vLLM models with tensor parallelism sizes greater than 1 and default "distributed_executor_backend" setting when using explicit model control mode. In attempt to load a vllm model (tp > 1) in explicit mode, users could potentially see failure at the `initialize` step: `could not acquire lock for <_io.BufferedWriter name='<stdout>'> at interpreter shutdown, possibly due to daemon threads`. For the default model control mode, after server shutdown, vllm related sub-processes are not killed. Related vllm issue: https://github.com/vllm-project/vllm/issues/6766 . Please specify distributed_executor_backend:ray in the model.json when deploying vllm models with tensor parallelism > 1.

- When loading models with file override, multiple model configuration files are not supported. Users must provide the model configuration by setting parameter config : <JSON> instead of custom configuration file in the following format: file:configs/<model-config-name>.pbtxt : <base64-encoded-file-content>.
- Perf Analyzer no longer supports the --trace-file option.
- TensorRT-LLM backend provides limited support of Triton extensions and features.
- The TensorRT-LLM backend may core dump on server shutdown. This impacts server teardown only and will not impact inferencing.
- The Java CAPI is known to have intermittent segfaults.
- Some systems which implement malloc() may not release memory back to the operating system right away causing a false memory leak. This can be mitigated by using a different malloc implementation. Tcmalloc and jemalloc are installed in the Triton container and can be used by specifying the library in LD_PRELOAD. NVIDIA recommends experimenting with both tcmalloc and jemalloc to determine which one works better for your use case.
- Auto-complete may cause an increase in server start time. To avoid a start time increase, users can provide the full model configuration and launch the server with --disable-auto-complete-config.
- Auto-complete does not support PyTorch models due to lack of metadata in the model. It can only verify that the number of inputs and the input names matches what is specified in the model configuration. There is no model metadata about the number of outputs and datatypes. Related PyTorch bug:https://github.com/pytorch/pytorch/issues/38273.
- Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip will install an incorrect Jetson version of Triton Client library for Arm SBSA. The correct client wheel file can be pulled directly from the Arm SBSA SDK image and manually installed.
- Traced models in PyTorch seem to create overflows when int8 tensor values are transformed to int32 on the GPU. Refer to pytorch/pytorch#66930 for more information.
- Triton cannot retrieve GPU metrics with MIG-enabled GPU devices.
- Triton metrics might not work if the host machine is running a separate DCGM agent on bare-metal or in a container.
- When cloud storage (AWS, GCS, AZURE) is used as a model repository and a model has multiple versions, Triton creates an extra local copy of the cloud model’s folder in the temporary directory, which is deleted upon server’s shutdown.
- Python backend support for Windows is limited and does not currently support the following features:
  - GPU tensors
  - CPU and GPU-related metrics
  - Custom execution environments
  - The model load/unload APIs