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

Given a trained model, how do I take it and deploy it at-scale with an
optimal configuration using Triton Inference Server? This document is
here to help answer that.

There are a few steps in this process, and there is no single answer for
everything, but here is the common flow:

1. Is my model compatible with Triton?
    - If your model falls under one of Triton's [supported backends](https://github.com/triton-inference-server/backend), then we can simply try to deploy the model as described in the [Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md) guide. For the ONNXRuntime, TensorFlow SavedModel, and TensorRT backends, the model configuration can be completely inferred from the model using Triton's [AutoComplete](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration) feature, so no `config.pbtxt` file is required. For other backends, refer to the [Minimal Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#minimal-model-configuration) required to get started.
    - If your model does not come directly from a supported backend, but is executable through Python; you may be interested in writing a simple python script for running your model through the [Python Backend](https://github.com/triton-inference-server/python_backend).
    - Otherwise, you can create a [Custom Backend](https://github.com/triton-inference-server/backend/blob/main/examples/README.md) for executing your model.

2. Does my model perform well?
    - Assuming you were able to deploy your model on Triton, the next step is to benchmark the performance of your model. Triton's [Perf Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) tool specifically fits this purpose. Here is a simplified output for demonstration purposes:

    ```
    # NOTE: "my_model" represents the model currently being served by 
    #       Triton in the background from the previous step.

    $ perf_analyzer -m my_model --concurrency-range 1:2

    ...

    Inferences/Second vs. Client Average Batch Latency
    Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
    Concurrency: 2, throughput: 765.2 infer/sec, latency 18191 usec
    ```

    - The defintion of "performing well" is subject to change for each use case. There are many variables that can be tweaked just within your Triton configuration (`config.pbtxt`) to obtain different results.

3. How can I improve my model performance?
    - To further understand the best model configuration you can provide to Triton for your use case, Triton's [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) tool can help. After running Model Analyzer to find the optimal configurations for your model/use case, you can transfer the generated config files to your [Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md). Model Analyzer provides a [Quickstart](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/quick_start.md) guide with some examples to walk through.
    - Upon serving the model with the newly optimized configuration file found by Model Analyzer and running Perf Analyzer again, you should expect to find better performance numbers in many cases compared to a default config.
    - Some parameters that can be tuned for a model may not be exposed to Model Analyzer since they don't apply to all models. For instance, [backends](https://github.com/triton-inference-server/backend) can expose backend-specific configuration options that can be tuned as well. The [ONNXRuntime Backend](https://github.com/triton-inference-server/onnxruntime_backend) for example has several [parameters](https://github.com/triton-inference-server/onnxruntime_backend#model-config-options) that affect the level of parallelization used when executing inference on a model and may be worth investigating if the defaults are not providing sufficient performance. 
    - For manual tuning of your model configuration, there is further information in our [Optimization](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md) docs.

## Other Areas of Interest

1. My model performs slowly when it is first loaded by Triton (cold-start penalty)
    - Triton exposes the ability to run [ModelWarmup](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#model-warmup) requests when first loading the model to ensure that the model is sufficiently initialized before being marked "READY" for inference.

2. Why doesn't my model perform significantly faster on GPU?
    - Most official backends supported by Triton are optimized for GPU inference and should perform well on GPU out of the box.
    - However, Triton expose options to further optimize your model for GPU execution for you. Triton's [Framework Specific Optimizations](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#framework-specific-optimization) goes into further detail on this topic.
    - Lastly, complete conversion of your model to a backend fully optimized for GPU inference such as [TensorRT](https://developer.nvidia.com/tensorrt) may provide even better results. You may find more Triton-specific details about TensorRT in the [TensorRT Backend](https://github.com/triton-inference-server/tensorrt_backend).
    - If none of the above can help get sufficient GPU-accelerated performance for your model, the model may simply be better designed for CPU execution and the [OpenVINO Backend](https://github.com/triton-inference-server/openvino_backend) may help further optimize your CPU execution.

## End-to-end Example

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

> NOTE: As of the 22.07 release, this step is optional by default.  Both Triton and Model Analyzer support fully auto-completing the configuration file for ONNX models.

3. Start server and SDK containers in the background

```bash
# Start server container in the background
docker run -d --gpus=all --network=host -v $PWD:/mnt --name triton-server nvcr.io/nvidia/tritonserver:22.07-py3 

# Start SDK container in the background
docker run -d --gpus=all --network=host -v $PWD:/mnt --name triton-sdk nvcr.io/nvidia/tritonserver:22.07-py3-sdk
```

> NOTE: These containers can be started interactively instead, but for the sake of demonstration it is more clear to start these containers in the background and `docker exec` into them as needed for the following steps.

4. Serve the model with Triton
```bash
# Enter server container interactively
docker exec -ti triton-server bash

# Start serving your models
tritonserver --model-repository=/mnt/models
```

> NOTE: The `-v $PWD:/mnt` is mounting your current directory on the host into the `/mnt` directory inside the container. So if you created your model repository in `$PWD/models`, you will find it inside the container at `/mnt/models`; You can change these paths as needed. See [docker volume](https://docs.docker.com/storage/volumes/) docs for more information on how this works.

5. Run Perf Analyzer on the model (in a separate shell)
```bash
# Enter SDK container interactively
docker exec -ti triton-sdk bash

# Benchmark model being served from step 3
perf_analyzer -m alexnet --concurrency 1:4
```

```
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 265.147 infer/sec, latency 3769 usec
Concurrency: 2, throughput: 890.793 infer/sec, latency 2243 usec
Concurrency: 3, throughput: 937.036 infer/sec, latency 3199 usec
Concurrency: 4, throughput: 965.21 infer/sec, latency 4142 usec
```

6. Run Model Analyzer to find the best configurations
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

7. Extract optimal `config.pbtxt` from Model Analyzer results

In our example above, `alexnet_config_4` was the optimal configuration. So let's extract that and put it back in our model repository for future use.

```bash
# (optional) Backup our original config.pbtxt (if any) to another directory
cp /mnt/models/alexnet/config.pbtxt /tmp/original_config.pbtxt

# Copy over the optimal config.pbtxt from model-analyzer results to our model repository
cp ./results/alexnet_config_4/config.pbtxt /mnt/models/alexnet/
```