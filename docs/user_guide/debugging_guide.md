<!--
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Debugging Guide
This guide goes over first-step troubleshooting for common scenarios in which Triton is behaving unexpectedly or failing. Below, we break down the issues into these categories:

- **[Configuration](#configuration-issues)**: Triton reports an error with your configuration file.
- **[Model](#model-issues)**: Your model fails to load or perform inference.
- Server: The server is crashing or unavailable.
- Client: The client is failing in sending and receiving data to the server.
- Performance: Triton is not achieving optimal performance.

Regardless of the category of your issue, it is worthwhile to try running in the latest Triton container, whenever possible. While we provide support to older containers, fixes get merged into the next release. By checking the latest release, you can spot whether this issue has already been resolved.

You can also search [Triton’s GitHub issues](https://github.com/triton-inference-server/server/issues) to see if someone previously asked about your issue. If you received an error, you can use a few keywords from the error as a search term.

Triton provides different types of errors and statuses, relevant across a wide swath of issues. Here is an overview of them:

| Error | Definition | Example |
| ----- | ---------- | ------- |
|Already Exists | Returned when an action cannot be done because there is already an existing item. | A registered model fails to be registered again.|
| Internal | Returned when there is an unexpected failure within the Triton code. | A memory allocation fails. |
| Invalid Arg | Returned when an invalid argument is provided to a function | A model config has an invalid parameter |
| Not Found | Returned when a requested resource is unable to be found | A shared library is unable to be found |
| Unavailable | Returned when a requested resource is found but unavailable | A requested model is not ready for inference |
| Unknown | Returned for cases where the reason for the error is unknown | This error code should not be used |
| Unsupported | Returned when an option is unsupported | A model config includes a parameter that is not yet supported for that backend |

## Configuration Issues

Before proceeding, please see if the model configuration documentation [here](./model_configuration.md) resolves your question. Beyond that, the best places to find a sample model configuration for your use cases are:

- The server [qa folder](https://github.com/triton-inference-server/server/tree/main/qa). You can find test scripts covering most features, including some which update the model config files to do so.
    - [Custom_models](https://github.com/triton-inference-server/server/tree/main/qa/custom_models), [ensemble_models](https://github.com/triton-inference-server/server/tree/main/qa/ensemble_models), and [python_models](https://github.com/triton-inference-server/server/tree/main/qa/python_models) include examples of configs for their respective use cases.
    - [L0_model_config](https://github.com/triton-inference-server/server/tree/main/qa/L0_model_config) tests many types of incomplete model configs.

Note that if you are running into an issue with [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md) or [Model Analyzer](https://github.com/triton-inference-server/model_analyzer), try loading the model onto Triton directly. This checks if the configuration is incorrect or the perf_analyzer or Model Analyzer options need to be updated.

## Model Issues
**Step 1. Run Models Outside of Triton**

If you are running into an issue with loading or running a model, the first step is to ensure your model runs in its framework outside of Triton. For example, you can run ONNX models in ONNX Runtime and TensorRT models in trtexec. If this check fails, the issue is happening within the framework and not within Triton.

**Step 2. Find the Error Message**

If you receive an error message, you may be able to find where it was generated by searching the code. GitHub provides instructions for searching code [here](https://docs.github.com/en/search-github/searching-on-github/searching-code). A generic search through the Triton organization is available at [this link](https://github.com/search?q=org%3Atriton-inference-server&type=Code).

If your error message only occurs in one or a few places in the Triton code, you may be able to see what’s going wrong pretty quickly. Even if not, it’s good to save this link to provide to us when asking for help with your issue. This is often the first thing we look for.

**Step 3. Build with Debug Flags**

The next step is building with debug flags. We unfortunately don’t provide a debug container, so you’d need to follow the [build guide](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md) to build the container, which includes a [section on adding debug symbols](https://github.com/triton-inference-server/server/blob/main/docs/build.md#building-with-debug-symbols). Once you do so, you can install GDB (`apt-get install gdb`) in the container and run Triton in GDB (`gdb --args tritonserver…`). If needed, you can open a second terminal to run a script in another container. If the server segfaults, you can enter `backtrace`, which will provide you a call stack that lets you know where the error got generated. You should then be able to trace the source of the error. If the bug still exists after debugging, we’ll need this to expedite our work.

Advanced GDB users can also examine variable values, add breakpoints, and more to find the cause of their issue.

### Specific Issues
**Undefined Symbols**

There are a few options here:
- This often means a version mismatch between the version of a framework used by Triton and the one used to create the model. Check the version of the framework used in the Triton container and compare against the version used to generate the model.
- If you are loading a shared library used by a backend, don’t forget to include LD_PRELOAD before the command to run Tritonserver. 
    - `LD_PRELOAD=<name_of_so_file.so> tritonserver --model-repository…`
If you built the backend yourself, this could be a linking error. If you are confident the backends and server were built correctly, double check that the server is loading the correct backend.

## Server Issues

You generally should not run into errors with the server itself. If the server goes down, it’s usually because something went wrong during model loading or inference and you can use the above section to debug. It’s particularly useful to work through the [Building with Debug Flags](https://github.com/triton-inference-server/server/blob/main/docs/build.md#building-with-debug-symbols) section above to resolve those sorts of issues. However, this section will go through some specific cases that may occur.

### No Connection to Server

If you are having trouble connecting to the server or getting its health via the health endpoint (`curl -v localhost:8000/v2/health/ready`), make sure you are able to reach the network your server is running on from where you are running your command. Most commonly, we see that when separate Docker containers are started for the client and server, they are not started with [--net=host](https://docs.docker.com/network/host/) to share the network.

### Intermittent Failure

This is going to be one of the hardest things to debug. If possible, you want to build your server with debug flags to get a backtrace of what is happening specifically. You would also want to keep notes to see how often this happens and whether that is a common cause. The server itself should not fail while idling, so see if a certain action (loading/unloading a model, running a model inference, etc.) is triggering it.

### Server Failure Due to Individual Models

If you want the server to start up even when models fail, use the `exit-on-error=false` option. If you want the server health endpoint to show ready even when specific models fail, use the `--strict-readiness=false` flag.

### Deadlock

Some useful steps for debugging a deadlock with `gdb`:
1. Use `$info threads` to see which threads are waiting.
2. Go to a thread: `$thread 4`.
3. Print the backtrace: `$bt`.
4. Go to the frame with the lock: `$f 1`.
5. Print the memory of the mutex being held: `$p *mutex`.
6. You can now see the owner of the mutex under `owner`.

## Client Issues

For working with different client cases, the best resources are the [client repo’s](https://github.com/triton-inference-server/client) examples. You can see clients written in Python, Java, and C++ with running examples across many common use cases. You can review the main functions of these clients to get a sense of the flow of the code.

We often get performance optimization questions around the clients. Triton clients send input tensors as raw binary. However, GRPC uses protobuf which has some serialization and deserialization overhead. For those looking for the lowest-latency solution, C API eliminates the latency associated with GRPC/HTTP. Shared memory is also a good option to reduce data movement when the client and server are on the same system.

## Performance Issues

This section goes over debugging unexpected performance. If you are looking to optimize performance, please see the [Optimization](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md) and [Performance Tuning](https://github.com/triton-inference-server/server/blob/main/docs/performance_tuning.md) guides.

The easiest step to start with is running perf_analyzer to get a breakdown of the request lifecycle, throughput, and latency for each individual model. For a more detailed view, you can [enable tracing](https://github.com/triton-inference-server/server/blob/main/docs/trace.md) when running the server. This will provide exact timestamps to drill down into what is happening. You can also enable tracing with perf_analyzer for the GRPC and HTTP clients by using the tracing flags. Note that enabling tracing can impact Triton’s performance, but it can be helpful to examine the timestamps throughout a request’s lifecycle.

### Performance Profiling

The next step would be to use a performance profiler. One profiler we recommend is [Nsight Systems](https://developer.nvidia.com/nsight-systems) (nsys), optionally including NVIDIA Tools Extension (NVTX) markers to profile Triton.

The Triton server container already has nsys installed. However, Triton does not build with the NVTX markers by default. If you want to use NVTX markers, you should build Triton with build.py, using the “--enable-nvtx” flag. This will provide details around some phases of processing a request, such as queueing, running inference, and handling outputs.

You can profile Triton by running `nsys profile tritonserver --model-repository …`. The [nsys documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) provides more options and details for getting a thorough overview of what is going on.

## Submitting an Issue

If you’ve done the initial debugging steps with no results, the next step is to submit the issue to us. Before you do so, please answer these questions:
- Is this reproducible with multiple models and/or our example models? Or is the issue unique to your model?
- Is the bug reproducible with any protocol (ex: HTTP vs GRPC)? Or only one protocol?

The answers to the above should inform what you submit. If you find that this issue only happens under specific circumstances, please include this in your report. If the issue still exists, please submit **all** of the below:

- The commands or script used to build/pull Triton and run your models.
    - If building Triton, please provide the version or branch you are building from.
- Your model configuration file.
- The error received, plus any logs.
    - If your issue involves the server crashing, a backtrace of the dump would be helpful.
    - Please enable verbose logging (--verbose-log=1) to get the most detailed logs.
- If this issue is unique to your model, your model or a toy model that reproduces the issue.
- Anything else that would expedite our investigation.
