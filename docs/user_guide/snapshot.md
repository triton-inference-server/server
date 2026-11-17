<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Snapshot

NVIDIA's [Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) supports Triton Inference Server.

Snapshot provides the checkpoint and restore primitives for GPU pods.
Orchestration (which pods to checkpoint, when, and how the checkpoints are restored) is left to the systems that integrate it.

## What Does Snapshot Provide?

In inference serving, a replica cannot answer a single request until it is fully initialized: model weights loaded into GPU memory, CUDA and runtime libraries initialized, execution kernels warmed up, and computation graphs compiled.
For large models, this cold start takes minutes.
This cost is paid over and over.

Snapshot checkpoints a fully initialized pod once and restores it on demand, so a new replica comes online in seconds instead of minutes.

> [!NOTE]
> Whereas Triton Server is hosting environment agnostic, Snapshot is intended for use in Kubernetes clusters and requires supported NVIDIA hardware and software packages.

## Triton Inference Server and Dynamo Snapshot

Using Snapshot to accelerate your deployment of Triton Server requires the following steps:

1. [Start Triton Server](#start-triton-server)
2. [Isolate Triton Server](#isolate-triton-server)
3. [Checkpoint and Restoration](#checkpoint-and-restoration)

### Start Triton Server

In order to successfully create and/or restore a checkpoint using Snapshot, the Triton Sever process must **NOT** have GPU metrics collection enabled.
This is due to a known issue related to [NVIDIA Management Library (NVML)](https://developer.nvidia.com/management-library-nvml) state capture and restoration.
To disable GPU metrics collection, the `--allow-gpu-metric=false` option must be provided when starting Triton Server.

> [!IMPORTANT]
> Any Triton Server process must be initialized with the `--allow-gpu-metrics=false` option to provided;
> otherwise, attempts to use Snapshot to checkpoint or restore will fail.

```bash
# Start Triton Server with GPU metrics collection disabled.
tritonserver --allow-gpu-metrics=false --model-repository=$MODEL_REPOSITORY_PATH
```

### Isolate Triton Server

When the checkpoint of Triton Server is created, any in-flight or enqueued inference requests will result in leaked or corrupted memory.
There should be zero pending, active, or otherwise "in-flight" inference requests being handled by the Triton Server process from which a checkpoint will be created.
This includes, but is not limited to, synchronous, asynchronous, streaming, and sequenced inference requests.

> [!IMPORTANT]
> Ensure that the Triton Server process has zero in-flight inference requests while Snapshot creates a checkpoint.

For best results, a Snapshot checkpoints should be created from Triton Server processes intended for the sole purpose of checkpoint creation, and not intended to be used to serve inference request traffic either before or after the checkpoint creation process.
For models which benefit from warm-up requests, we recommend limiting all inference request traffic to only the required warm-up requests, ensuring all requests complete, and there is zero inference request traffic being served by the server prior to the checkpoint being created.

### Checkpoint and Restoration

To checkpoint the Triton Server pod, follow the Snapshot documentation on how to [Checkpoint a Pod](https://github.com/ai-dynamo/snapshot/blob/main/docs/using-snapshot/checkpoint.md). Similarly, use the documentation on how to [Restore a Checkpoint](https://github.com/ai-dynamo/snapshot/blob/main/docs/using-snapshot/restore.md) provided by Snapshot.

Once a Triton Server checkpoint has been restored, the server should be immediately available to serve inference request traffic.
