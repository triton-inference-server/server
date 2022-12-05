# Dynamic Batching & Concurrent Model Execution

Part-1 of this series introduced the mechanisms to setup a Triton Inference Server. This iteration discusses the concept of dynamic batching and concurrent model execution. These are important features which can be used to reduce the latency as well as increase throughput via higher resource utilization. 

## What is Dynamic Batching?

Dynamic batching, in reference to the Triton Inference Server, refers to the functionality which allows the combining of one or more infernce requests into a single batch (which has to be created dynamically) to maximize throughput. 

Dynamic batching can be enabled and configured on per model basis by specifying selections in the model's `config.pbtxt`. Dynamic Batching can be enabled with its default settings by adding the following to the `config.pbtxt` file:
```
dynamic_batching { }
```
While Triton batches these incoming requests without any delay, users can choose to allocate a limited delay for the scheduler to collect more inference requests to be used by the dynamic batcher.

```
dynamic_batching {
    max_queue_delay_microseconds: 100
}
```
Let's discuss a sample scenario(refer the diagram below). Say there are 5 inference requests, `A`, `B`, `C`, `D`, and `E`, with batch sizes of `4`, `2`, `2`, `6`, and `2` respecitvely. Each batch requires time `X ms` to be processed by the model. The maximum batch size supported by the model is `8`. `A` and `C` arrive at time `T = 0`, `B` arrives at time `T = X/3`, and `D` and `E` arrive at time `T = 2*X/3`.

![Dynamic Batching Sample](./img/dynamic_batching.PNG)

In case where no dynamic batching is used, all requests are processed sequentially, which means that it takes `5X ms` to process all the requests. This process is quite wastefull as each batch processing could have processed more batches than it did in sequential execution.

Using Dynamic batcher in this case leads to more efficient packing of requests into the GPU memory resulting in a considerably faster `3X ms`. It also reduces the latency of responses as more queries can be processed in fewer cycles. If the use of `delay` is considered, `A`, `B`, `C` and `D`, `E` can be batched together to get even better utilization of resources. 

**Note:** The above is an extreme version of an ideal case scenerio. In practice, not all elements of exectution can be perfectly parallized, resulting in longer exectution time for larger batches.

As observed from the above, use of Dynamic Batching can lead to improvements in both latency and throughput while serving models. This batching feature is mainly focused on providing a solution for statless models(models which do not maintain a state between execution, like object detection models). Triton's [sequence batcher](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#sequence-batcher) can be used to manage multiple inference requests for stateful models. For more information and configurations regarding dynamic batching, refer to the Triton Inference Server [documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#scheduling-and-batching).

## Concurrent model execution

The Triton Inference Server can spin up multiple instance of the same model which can process queries in parallel. Triton can spawn instances on the same device (GPU), or a different device on the same node as per the user's specifications. This customizablity is especially useful when considering ensembles which have models with different throughput. Multiple copies of slowers models can be spawned on a separate GPU to allow for more parallel processing. This is enabled via the use of `instance groups` option in a model's configuration.

```
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
] 
```

Let's take the previous example and discuss the effect of adding multiple model for parallel execution. In this example, instead of having a single model process five queries, two models are spawned. ![Multiple Model Instances](./img/2.PNG)

For a "no dynamic batching" case, as there are model models to execute, the queries are distributed equally. Users can also add [priorities](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#priority) to prioritize or de-prioritize any specific instance group.

When considering the case of multiple instance with dynamic batches enabled, owing to the availablity of another instance, query `B` which arrives with some delay can be executed using the second instance. With some delay allocated, instance 1 gets filled and launched by time `T = X/2` and since queries `D` and `E` stack up to fill up to the maximum batch size, the second model can start inference without any delay. 

The key take away with the above examples is that the Triton Inference Server provides flexiblity with respect to policies related to creating more efficient batching, thus enabling better resource utilzation, resulting in reduced latency and increased throughput.

## Demonstration

This section showcases the use of dynamic batching and concurrent model execution using the example for Part 1 of this series.

### Getting access to the model

Let's use the `text recognition` used in part 1. We do need to make some minor changes in the model, namely making the 0th axes of dynamic shape to enable batching. Step 1, download the Text Recognition model weights. Use the NGC PyTorch container as the environment for the following.
```
docker run -it --gpus all -v ${PWD}:/scratch nvcr.io/nvidia/pytorch:<yy.mm>-py3
cd /scratch
wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth
```
Export the models as .onnx using the file in the utils folder. This file is adapted from [Baek et. al. 2019](https://github.com/clovaai/deep-text-recognition-benchmark).
```
import torch
from utils.model import STRModel

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create ONNX file by tracing model
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, trace_input, "str.onnx", verbose=True, dynamic_axes={'input.1':[0],'308':[0]})
```

### Launching the server

As discussed in `Part 1`, a model repository is a filesystem based repository of models and configuration schema used by the Triton Inference Server (refer to `Part 1` for a more detailed explanation for model repositories). For this example the model repository structure would need to be set up in the following manner:
```
model_repository
|   
|-- text_recognition
    |
    |-- config.pbtxt
    |-- 1
        |
        |-- model.onnx
```
This repository is a subset from the previous example. The key difference in this setup is the use of `instance_group`(s) and `dynamic_batching` in the model configuration. The additions are as follows:

```
instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]
dynamic_batching { }
```
With `instance_group` users can primarily tweak two things. First, the number of instances on a that model deployed on each GPU. The above example will deploy `2` intances of the model `per GPU`. Secondly, the target GPUs for this group can be specified with `gpus: [ <device number>, ... <device number> ]`.

Adding `dynamic_batching {}` will enable the use of dynamic batches. Users can also add `preferred_batch_size` and `max_queue_delay_microseconds` in the body of dynamic batching to manage more efficient batching per their use case. Explore the [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#model-configuration) documentation for more information

As the model repository setup, the Triton Inference Server can be launched.
```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 bash

tritonserver --model-repository=/models
```

### Measuring Performance

Having made some improvements to the model's serving capablities with enabling `dynamic batching` and the use of `multiple model instances`, the next step is to make observations these changes. To that end, the Triton Inference Server comes packaged with the [Performance Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) which is a tool specifically designed to measure perfomance for Triton Inference Servers. For ease of use, it is recommended that users run this inside the same container used to run client code in Part 1 of this series.
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.08-py3-sdk bash
```
On a third terminal, it is advisable to monitor the GPU Utilization to see if the deployment is saturating GPU resources.
```
watch -n0.1 nvidia-smi
```

To measure the performance gain, let's run performance analyzer on the following configurations:

* **No Dynamic Batching, single model instance**: This configuration will be the base line measurement. To setup the Triton Server in this configuration, do not add `instance_group` or `dynamic_batching` in `config.pbtxt` and make sure to include `--gpus=1` in the `docker run` command to setup the server.

```
# perf_analyzer -m <model name> -b <batch size> --shape <input layer>:<input shape> --concurrency-range <lower number of request>:<higher number of request>:<step>

# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Summarized Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 955.708 infer/sec, latency 4311 usec
Concurrency: 4, throughput: 977.314 infer/sec, latency 8497 usec
Concurrency: 6, throughput: 973.367 infer/sec, latency 12799 usec
Concurrency: 8, throughput: 974.623 infer/sec, latency 16977 usec
Concurrency: 10, throughput: 975.859 infer/sec, latency 21199 usec
Concurrency: 12, throughput: 976.191 infer/sec, latency 25519 usec
Concurrency: 14, throughput: 966.07 infer/sec, latency 29913 usec
Concurrency: 16, throughput: 975.048 infer/sec, latency 34035 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 8777
    Throughput: 975.048 infer/sec
    p50 latency: 32566 usec
    p90 latency: 33897 usec
    p95 latency: 34035 usec
    p99 latency: 34241 usec
    Avg HTTP time: 32805 usec (send/recv 43 usec + response wait 32762 usec)
  Server:
    Inference count: 143606
    Execution count: 71803
    Successful request count: 71803
    Avg request latency: 17937 usec (overhead 14 usec + queue 15854 usec + compute input 20 usec + compute infer 2040 usec + compute output 7 usec)
```

* **Just Dynamic Batching**: To setup the Triton Server in this configuration, add `dynamic_batching` in `config.pbtxt`.
```
# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 998.141 infer/sec, latency 4140 usec
Concurrency: 4, throughput: 1765.66 infer/sec, latency 4750 usec
Concurrency: 6, throughput: 2518.48 infer/sec, latency 5148 usec
Concurrency: 8, throughput: 3095.85 infer/sec, latency 5565 usec
Concurrency: 10, throughput: 3182.83 infer/sec, latency 7632 usec
Concurrency: 12, throughput: 3181.3 infer/sec, latency 7956 usec
Concurrency: 14, throughput: 3184.54 infer/sec, latency 10357 usec
Concurrency: 16, throughput: 3187.76 infer/sec, latency 10567 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 28696
    Throughput: 3187.76 infer/sec
    p50 latency: 10030 usec
    p90 latency: 10418 usec
    p95 latency: 10567 usec
    p99 latency: 10713 usec
    Avg HTTP time: 10030 usec (send/recv 54 usec + response wait 9976 usec)
  Server:
    Inference count: 393140
    Execution count: 64217
    Successful request count: 196570
    Avg request latency: 6231 usec (overhead 31 usec + queue 3758 usec + compute input 35 usec + compute infer 2396 usec + compute output 11 usec)
```
As each of the requests had a batchsize of 2, while the maximum batch size of the model was 8, dynamically batching these requests resulted in considerably improved throughput. Another conquence is a reduction in the latency. This reduction can be primarily attributed to reduced wait time in queue wait time. As the requests are batched together, multiple requests can be processed in parallel.

* **Dynamic Batching with multiple model instances**: To setup the Triton Server in this configuration, add `instance_group` in `config.pbtxt` and make sure to include `--gpus=1` and make sure to include `--gpus=1` in the `docker run` command to setup the server. Include `dynamic_batching` per instructions of the previous section in the model configuration. A point to note is that peak GPU utilization on the GPU shot up to 74% (A100 in this case) while just using a single model instance with dynamic batching. Adding one more instance will definitely improve performance but linear perf scaling will not be achieved in this case.

```
# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 1446.26 infer/sec, latency 3108 usec
Concurrency: 4, throughput: 1926.1 infer/sec, latency 5491 usec
Concurrency: 6, throughput: 2695.12 infer/sec, latency 5710 usec
Concurrency: 8, throughput: 3224.69 infer/sec, latency 6268 usec
Concurrency: 10, throughput: 3380.49 infer/sec, latency 6932 usec
Concurrency: 12, throughput: 3982.13 infer/sec, latency 7233 usec
Concurrency: 14, throughput: 4027.74 infer/sec, latency 7879 usec
Concurrency: 16, throughput: 4134.09 infer/sec, latency 8244 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 37218
    Throughput: 4134.09 infer/sec
    p50 latency: 7742 usec
    p90 latency: 8022 usec
    p95 latency: 8244 usec
    p99 latency: 8563 usec
    Avg HTTP time: 7734 usec (send/recv 54 usec + response wait 7680 usec)
  Server:
    Inference count: 490626
    Execution count: 101509
    Successful request count: 245313
    Avg request latency: 5287 usec (overhead 29 usec + queue 1878 usec + compute input 36 usec + compute infer 3332 usec + compute output 11 usec)
```

This is a perfect example of "simply enabling all the features" isn't a one-size fits all solution. A point to note is that this experiment was conducted by caping the maximum batch size of the model to `8`, while having a single GPU setup. Each production environment is different. Models, hardware, business level SLAs, costs, are all variables which need to be taken into account while selecting appropriate deployment configurations. Running through a grid search for each and every deployment isn't a feasible strategy. To solve this challenge, Triton users can make use of the Model Analyzer covered in Part 3 of this tutorial! Checkout [this section of the documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#optimization) for another example of dynamic batching and multiple model instance.

# What's next?

In this tutorial, we covered the very basics two key concepts, `dynamic batching` and `concurrent model execution` which can be used improve the resource utilization. This is Part 2 of a 10 part tutorial series which covers the challenges faced in deploying Deep Learning models to production. As you may have figured, there are many possible combinations to use the features discussed in this tutorial, especially with nodes having multiple GPUs. Part 3 covers `Model Analyzer`, a tool which helps to find the best possible deployment configuration.
