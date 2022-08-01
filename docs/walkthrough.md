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

# From Model to Deployment

Given a trained model, how do I deploy it at-scale with an
optimal configuration using Triton Inference Server? This document is
here to help answer that.

For those who like a high level overview, below is the common flow for most use cases. For those who wish to jump right in, skip to the [end-to-end example](#end-to-end-example).

1. Is my model compatible with Triton?
    - If your model falls under one of Triton's [supported backends](https://github.com/triton-inference-server/backend), then we can simply try to deploy the model as described in the [Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md) guide. For the ONNXRuntime, TensorFlow SavedModel, and TensorRT backends, the model configuration can be completely inferred from the model using Triton's [AutoComplete](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration) feature, so no `config.pbtxt` file is required. Additionally, by enabling verbose logging via `--log-verbose=1`, you can see the complete config that Triton sees internally in the server log output. For other backends, refer to the [Minimal Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#minimal-model-configuration) required to get started.
    - If your model does not come directly from a supported backend, but is executable through Python, you may be interested in writing a simple python script for running your model through the [Python Backend](https://github.com/triton-inference-server/python_backend).
    - Otherwise, you can create a [Custom Backend](https://github.com/triton-inference-server/backend/blob/main/examples/README.md) for executing your model.

2. Can I run inference on my served model?
    - Assuming you were able to load your model on Triton, the next step is to verify that we can run inference requests and get a baseline performance benchmark of your model. Triton's [Perf Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) tool specifically fits this purpose. Here is a simplified output for demonstration purposes:

    ```
    # NOTE: "my_model" represents the model currently being served by 
    #       Triton in the background from the previous step.

    $ perf_analyzer -m my_model

    ...

    Inferences/Second vs. Client Average Batch Latency
    Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
    ```

    - This gives us a sanity test that we are able to successfully form input requests and receive output responses to communicate with the model backend via Triton APIs. If Perf Analyzer fails to send requests and it is unclear from the error how to proceed, then you may want to sanity check that your model `config.pbtxt` inputs/outputs match what the model expects. If the config is correct, check that the model runs successfully using its original framework directly. If you don't have your own script or tool to do so, [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) is a useful tool to run sample inferences on a model via various frameworks.
    - The definition of "performing well" is subject to change for each use case. Some common metrics are throughput, latency, and GPU utilization. There are many variables that can be tweaked just within your Triton configuration (`config.pbtxt`) to obtain different results.
    - As your model, config, or use case evolves, [Perf Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) is a great tool to quickly verify model functionality and performance.

3. How can I improve my model performance?
    - To further understand the best model configuration you can provide to Triton for your use case, Triton's [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) tool can help. After running Model Analyzer to find the optimal configurations for your model/use case, you can transfer the generated config files to your [Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md). Model Analyzer provides a [Quickstart](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/quick_start.md) guide with some examples to walk through.
    - Upon serving the model with the newly optimized configuration file found by Model Analyzer and running Perf Analyzer again, you should expect to find better performance numbers in many cases compared to a default config.
    - Some parameters that can be tuned for a model may not be exposed to Model Analyzer since they don't apply to all models. For instance, [backends](https://github.com/triton-inference-server/backend) can expose backend-specific configuration options that can be tuned as well. The [ONNXRuntime Backend](https://github.com/triton-inference-server/onnxruntime_backend), for example, has several [parameters](https://github.com/triton-inference-server/onnxruntime_backend#model-config-options) that affect the level of parallelization when executing inference on a model. These backend-specific options may be worth investigating if the defaults are not providing sufficient performance. 
    - For manual tuning of your model configuration, there is further information in our [Optimization](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md) docs.

## Other Areas of Interest

1. My model performs slowly when it is first loaded by Triton (cold-start penalty), what do I do?
    - Triton exposes the ability to run [ModelWarmup](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#model-warmup) requests when first loading the model to ensure that the model is sufficiently initialized before being marked "READY" for inference.

2. Why doesn't my model perform significantly faster on GPU?
    - Most official backends supported by Triton are optimized for GPU inference and should perform well on GPU out of the box.
    - Triton exposes options for you to optimize your model further on the GPU. Triton's [Framework Specific Optimizations](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#framework-specific-optimization) goes into further detail on this topic.
    - Complete conversion of your model to a backend fully optimized for GPU inference such as [TensorRT](https://developer.nvidia.com/tensorrt) may provide even better results. You may find more Triton-specific details about TensorRT in the [TensorRT Backend](https://github.com/triton-inference-server/tensorrt_backend).
    - If none of the above can help get sufficient GPU-accelerated performance for your model, the model may simply be better designed for CPU execution and the [OpenVINO Backend](https://github.com/triton-inference-server/openvino_backend) may help further optimize your CPU execution.

## End-to-end Example

> NOTE: If you have never worked with Triton before, you may be interested in first checking out the [Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md) example. Some basic understanding of Triton may be useful for the following section, but this example is meant to be straightforward enough without prior experience.

Let's take an ONNX model as our example since ONNX is designed to be a format that can be [easily exported](https://github.com/onnx/tutorials#converting-to-onnx-format) from most other frameworks.

1. Download our example Alexnet model from the [ONNX Model Zoo](https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx).

```bash
# Create model repository
mkdir -p ./models/alexnet/1

# Download model and place it in model repository
wget -O models/alexnet/1/model.onnx https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx?raw=true
```

2. Create a minimal [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) at `./models/alexnet/config.pbtxt`
```protobuf
name: "alexnet"
backend: "onnxruntime"
max_batch_size: 0
input: [
  {
    name: "data_0",
    data_type: TYPE_FP32,
    dims: [ 1, 3, 224, 224]
  }
]
output: [
  {
    name: "prob_1",
    data_type: TYPE_FP32,
    dims: [ 1, 1000 ]
  }
]
```

> NOTE: As of the 22.07 release, both Triton and Model Analyzer support fully auto-completing the config file for [backends that support it](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration). So for an ONNX model for example, this step can be skipped unless you want to explicitly set certain parameters.

3. Start server and SDK containers in the background

These containers can be started interactively instead, but for the sake of demonstration it is more clear to start these containers in the background and `docker exec` into them as needed for the following steps.

```bash
# Start server container in the background
docker run -d --gpus=all --network=host -v $PWD:/mnt --name triton-server nvcr.io/nvidia/tritonserver:22.07-py3 

# Start SDK container in the background
docker run -d --gpus=all --network=host -v $PWD:/mnt --name triton-sdk nvcr.io/nvidia/tritonserver:22.07-py3-sdk
```



4. Serve the model with Triton

To serve our model, we will use the `server` container that we already started which comes pre-installed with a `tritonserver` binary.

```bash
# Enter server container interactively
docker exec -ti triton-server bash

# Start serving your models
tritonserver --model-repository=/mnt/models
```

> NOTE: The `-v $PWD:/mnt` is mounting your current directory on the host into the `/mnt` directory inside the container. So if you created your model repository in `$PWD/models`, you will find it inside the container at `/mnt/models`. You can change these paths as needed. See [docker volume](https://docs.docker.com/storage/volumes/) docs for more information on how this works.

5. Verify the model can run inference

To verify our model can perform inference, we will use the `triton-sdk` (client) container that we already started which comes with `perf_analyzer`  pre-installed.

In a separate shell, we use Perf Analyzer to sanity check that we can run inference and get a baseline for the kind of performance we expect from this model.

In the example below, Perf Analyzer is sending requests to models served on the same machine (`localhost` from the server container via `--network=host`). However, you may also test models being served remotely at some `<IP>:<PORT>`  by setting the `-u` flag, such as `perf_analyzer -m alexnet -u 127.0.0.1:8000`.

```bash
# Enter SDK container interactively
docker exec -ti triton-sdk bash

# Benchmark model being served from step 3
perf_analyzer -m alexnet --concurrency-range 1:4
```

```
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 265.147 infer/sec, latency 3769 usec
Concurrency: 2, throughput: 890.793 infer/sec, latency 2243 usec
Concurrency: 3, throughput: 937.036 infer/sec, latency 3199 usec
Concurrency: 4, throughput: 965.21 infer/sec, latency 4142 usec
```

6. Run Model Analyzer to find the best configurations for our model

While Model Analyzer comes pre-installed in the SDK container and supports various modes of connecting to a Triton server,
for simplicity we will use install Model Analyzer in our `server` container to use the `--triton-launch-mode=local` flag.
To learn more about other methods of connecting Model Analyzer to a running Triton Server, see the `--triton-launch-mode`
Model Analyzer flag.

```bash
# Enter server container interactively
docker exec -ti triton-server bash

# Stop existing tritonserver process if still running
# because model-analyzer will start its own server
SERVER_PID=`ps | grep tritonserver | awk '{ printf $1 }'`
kill ${SERVER_PID}

# Install model analyzer
pip install --upgrade pip 
pip install triton-model-analyzer wkhtmltopdf

# Profile the model
# NOTE: This may take some time, in this example it took ~10 minutes
model-analyzer profile \
  --model-repository=/mnt/models \
  --profile-models=alexnet \
  --triton-launch-mode=local \
  --output-model-repository-path=results

# Summarize the profiling results
model-analyzer analyze --analysis-models=alexnet
```

Example Model Analyzer output summary:

> In 41 measurements across 6 configurations, `alexnet_config_4` provides the best throughput: 1676 infer/sec.
>
> **This is a 67% gain over the default configuration (1002 infer/sec), under the given constraints.**

| Model Config Name | Max Batch Size | Dynamic Batching | Instance Count | p99 Latency (ms) | Throughput (infer/sec) | Max GPU Memory Usage (MB) | Average GPU Utilization (%) |
|---|---|---|---|---|---|---|---|
| alexnet_config_4 | 0 | Enabled | 5/GPU | 22.333 | 1675.98 | 6437 | 31.7 |
| alexnet_config_3 | 0 | Enabled | 4/GPU | 43.26 | 1612.95 | 5809 | 53.4 |
| alexnet_config_2 | 0 | Enabled | 3/GPU | 44.403 | 1563.16 | 5117 | 55.1 |
| alexnet_config_default | 0 | Disabled | 1/GPU | 8.274 | 1001.77 | 3786 | 24.6 |

In the table above, we see that setting our GPU [Instance Count](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#instance-groups) to 5 allows us to achieve the highest throughput and lowest latency for this example. Note that our this "alexnet" model has a fixed batch-size that is explicitly specified in the first dimension of the Input/Output `dims`, therefore the `max_batch_size` parameter is set to 0 as described [here](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#maximum-batch-size). For models that support dynamic batch size, Model Analyzer would also tune the `max_batch_size` parameter.

7. Extract optimal config from Model Analyzer results

In our example above, `alexnet_config_4` was the optimal configuration. So let's extract that `config.pbtxt` and put it back in our model repository for future use.

```bash
# (optional) Backup our original config.pbtxt (if any) to another directory
cp /mnt/models/alexnet/config.pbtxt /tmp/original_config.pbtxt

# Copy over the optimal config.pbtxt from Model Analyzer results to our model repository
cp ./results/alexnet_config_4/config.pbtxt /mnt/models/alexnet/
```

Now that we have an optimized Model Configuration, we are ready to take our model to deployment. For further manual tuning, read the [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) and [Optimization](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#) docs to learn more about Triton's complete set of capabilities.

In this example, we happened to get both the highest throughput and lowest latency from the same configuration, but in some cases this is a tradeoff that must be made. Certain models or configurations may achieve a higher throughput but also incur a higher latency in return. It is worthwhile to fully inspect the reports generated by Model Analyzer to ensure your model performance meets your requirements.
