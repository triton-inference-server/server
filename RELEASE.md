<!--
 # Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 # Release Notes for 2.40.0

 ## New Features and Improvements

 * Implicit state management has been enhanced to
[support growing buffers](https://github.com/triton-inference-server/common/blob/a8a7341ff15bb6faddde2d11035c895476516a96/protobuf/model_config.proto#L1405)
and [use a single buffer](https://github.com/triton-inference-server/common/blob/a8a7341ff15bb6faddde2d11035c895476516a96/protobuf/model_config.proto#L1386C3-L1386C3) for both input and output states.

 * Sequence batcher has been enhanced to support
[iterative scheduling](https://github.com/triton-inference-server/server/blob/r23.11/docs/user_guide/model_configuration.md#iterative-sequences).
The backend API has been enhanced to support rescheduling a request.
Currently, only [Python
backend](https://github.com/triton-inference-server/python_backend/tree/r23.11#request-rescheduling) and Custom C++ backends support request rescheduling.

 * TRT-LLM backend now supports request cancellation.

 * Configuration of a vLLM backend model can now be auto-completed by Triton. The
 user just needs to pass backend: "vllm" to leverage the auto-complete feature.

 * Python backend now supports parameters in BLS requests.
 * Python backend GPU tensor support has been improved to provide better
 performance.
 * A [new tutorial](https://github.com/triton-inference-server/tutorials/blob/r23.11/Popular_Models_Guide/Llama2/trtllm_guide.md)
demonstrating how to deploy LLaMa2 using TRT-LLM has been added.

 * The HTTP endpoint has been enhanced to support [access restriction](https://github.com/triton-inference-server/server/blob/r23.11/docs/customization_guide/inference_protocols.md#limit-endpoint-access-beta).
 * [Secure Deployment Guide](https://github.com/triton-inference-server/server/blob/r23.11/docs/customization_guide/deploy.md)
has been added to provide guidance on deploying Triton securely.
 * The client model loading API no longer allows uploading files outside the model
 repository.
 * DCGM version has been upgraded to 3.2.6.
 * The [Kubernetes Deploy example](https://github.com/triton-inference-server/server/tree/r23.11/deploy/k8s-onprem)
now supports Kubernetes’ new StartupProbe to allow
 Triton pods time to finish startup before running health probes.

 * Refer to the 23.11 column of the [Frameworks Support
   Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
for container image versions on which the 23.10 inference server container is
based.

 ## Known Issues
* When using the generate streaming endpoint, Triton will segfault
  if the client closes the connection before all responses have been generated.
  The [fix](https://github.com/triton-inference-server/server/pull/6591)
  will be available in the next release.
  
* Reuse-grpc-port and reuse-http-port are now properly parsed as booleans. 0 and 1
will continue to work as values. Any other integers will throw an error.

 * The TensorRT-LLM
   [backend](https://github.com/triton-inference-server/tensorrtllm_backend)
provides limited support of Triton extensions and features.

 * The TensorRT-LLM backend may core dump on server shutdown. This impacts
   server teardown only and will not impact inferencing.

 * When a model uses a backend which is not found, Triton would reference the
   missing  backend as `backend_name /model.py” in the error message. This is
already fixed for future releases.

 * When using decoupled models, there is a possibility that response order as
   sent from the backend may not match with the order in which these responses
are received by the streaming gRPC client. Note that this only applies to
responses from different requests. Any responses corresponding to the same
request will still be received in their expected order, relative to each other.

 * The FasterTransformer backend is only officially supported for 22.12, though
   it can be built for Triton container versions up to 23.07.

 * The Java CAPI is known to have intermittent segfaults we’re looking for a
   root cause.

 * Some systems which implement malloc() may not release memory back to the
   operating system right away causing a false memory leak. This can be
mitigated by using a different malloc implementation. Tcmalloc and jemalloc are
installed in the Triton container and can be [used by specifying the library in
LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md).
We recommend experimenting with both tcmalloc and jemalloc to determine which
one works better for your use case.

 * Auto-complete may cause an increase in server start time. To avoid a start
   time increase, users can provide the full model configuration and launch the
server with `--disable-auto-complete-config`.

 * Auto-complete does not support PyTorch models due to lack of metadata in the
   model. It can only verify that the number of inputs and the input names
matches what is specified in the model configuration. There is no model metadata
about the number of outputs and datatypes. Related PyTorch bug:
https://github.com/pytorch/pytorch/issues/38273

 * Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip
   will install an incorrect Jetson version of Triton Client library for Arm
SBSA. The correct client wheel file can be pulled directly from the Arm SBSA SDK
image and manually installed.

 * Traced models in PyTorch seem to create overflows when int8 tensor values are
   transformed to int32 on the GPU. Refer to
https://github.com/pytorch/pytorch/issues/66930 for more information.

 * Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and
   A30).

 * Triton metrics might not work if the host machine is running a separate DCGM
   agent on bare-metal or in a container.

 * When cloud storage (AWS, GCS, AZURE) is used as a model repository and a
   model has multiple versions, Triton creates an extra local copy of the cloud
model’s folder in the temporary directory, which is deleted upon server’s
shutdown.
