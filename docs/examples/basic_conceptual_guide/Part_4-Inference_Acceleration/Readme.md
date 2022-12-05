# Accelerating Inference for Deep Learning Models

Model acceleration is a complex nuanced topic. The viablity of techniques like graph optimizations for models, pruning, knowledge distalation, quantization, and more, highly depend on the structure of the model. Each of these topics are vast fields of research in their own right and building custom tools requires massive engineering investment. It is reasonable to assume that for challenges as complex and important as the aforementioned, a plethora of tools and SDKs would be available. A discussion of the merits and challenges of each tool comes with many "in-general"(s) and caviats. Rather than having an exhaustive outline of the ecosystem, for brevity and objectivity, this disucssion will be focused on the tools and features which are recommended to use while deploying models using the Triton Inference Server.

![Triton Flow](./img/1.PNG)

Triton Inference Server has a concept called "Triton Backend" or "Backend". Backends are essentially the implementation that executes the model. A backend can be a wrapper around a popular deep learning framework like PyTorch, TensorFlow, TensorRT or ONNX Runtime, or users can choose to build their own backends customized to their models and usecase. Each of these backends have their own specific options for acceleration.

Acceleration recommendations depend on two main factors:
* **Type of Hardware**: Triton users can choose to run models on GPU or CPU. Owing to the parallelism they provide, GPUs provide many avenues of performance acceleration. Models using PyTorch, TensorFlow, ONNX runtime, and TensorRT can utilize these benefits. For CPUs Triton 
* **Type of the model**: Usually users leverage one or more of three different classes of model: `Shallow models` like Random Forests, `Neural Networks` like BERT or CNNs, and lastly, `Large Transformer Models` which are usually too big to fit in a single GPU's memory. Each of these category of models leverage different optimizations to accelerate models. 

![Decision Tree](./img/2.PNG)

With these broad categories in considered, let's drill down into the specific scenrios and decision making process to pick the most appropriate Triton Backend for the use case along with a brief discussion about possible optimizations. 

## GPU Based Acceleration

As mentioned before, acceleration for deep learning models can be achived with a wide variety of methods. Graph level optimizations like fusing layers can reduce the number of GPU kernels that are needed to be launched for execution. Fusing them together also makes this process more memory efficient and increases the density of operations. Once fused, we can utilize a kernel auto tuner to pick the correct combination of kernels to maximize utilization of GPU resources. Use of lower precision with techniques like quantization can drastically reduce memory requirements and increase througput. The exact nature of the tactics which can be used to optimize performance differ from GPU to GPU owing to different hardware design. These are a few of many challenges we solve for Deep Learning Practicioners with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) which is an SDK focused on deep learning inference optimizations.

While TensorRT works with popular deep learning with PyTorch, TensorFlow, MxNET, ONNX Runtime and more it also has framework level integrations with PyTorch([Torch-TensorRT](https://github.com/pytorch/TensorRT)) and TensorFlow([TensorFlow-TensorRT](https://github.com/tensorflow/tensorrt)) to provide their respective developers with the flexibility and fallback mechanisms. 

### Using TensorRT directly

There are three routes for users to use to convert their models to TensorRT: the C++ API, Python API, and trtexec (TensorRT's command line tool). [Refer this guide for a fleshed out example](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton). That said there are two main steps needed. First, convert the model to a TensorRT Engine. It is recommended to use the [TensorRT Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt) to run the command.

```
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --explicitBatch \
        --useCudaGraph
```

Once converted, place the model in the `model.plan` in the model repository (as described in part 1) and use `tensorrt_plan` as the `platform` in the `config.pbtxt`

Apart from just the conversion to TensorRT, users can also leverage some [cuda specfic optimizations](https://github.com/triton-inference-server/common/blob/d4017443199e4f19462360789f5c80b0eb1e4738/protobuf/model_config.proto#L823). 

For cases where users run into a situation where some of the operators in their models aren't supported by TensorRT there are three possible options:
* **Use one of the framework integrations**: TensorRT has two integrations with Frameworks: Torch-TensorRT (PyTorch), and TensorFlow-TensorRT (TensorFlow). These integrations have a fallback mechanism built in to use the framework backend in cases where TensorRT doesn't directly support the graph.

* **Use the ONNX Runtime with TensorRT**: Triton users can also leverage this fallback mechanism with the ONNX Runtime (more in the following section).

* **Build a plugin**: TensorRT allows for building plugins and implementing custom ops. Users can write their own [TensorRT plugins](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending) to implement unsupported ops(Recommended for expert users). It is highly encoraged to report said ops to have them inately supported by TensorRT. 

### Using TensorRT's integration with PyTorch/TensorFlow
In the case of **PyTorch**, Torch-TensorRT is an Ahead of Time Compiler which converts TorchScript/Torch FX to a module targeting a TensorRT Engine. Post compilation, users can use the optimized model in the same manner as they would use a TorchScript model. Check out the [getting started](https://www.youtube.com/watch?v=TU5BMU6iYZ0) with Torch TensorRT to learn more. [Refer this guide for a fleshed out example](https://pytorch.org/TensorRT/tutorials/serving_torch_tensorrt_with_triton.html) demonstrating compilation of PyTorch model with Torch TensorRT and deploying it on Triton.  

**TensorFlow** users can make use of TensorFlow TensorRT, which segments the graph into subgraphs which are supported and not supported by TensorRT. The supported subgraphs are then replaced by a TensorRT optimized node producing a graph which has both TensorFlow and TensorRT components. [Refer to this tutorial](https://github.com/tensorflow/tensorrt/tree/master/tftrt/triton) explaining the exact steps required to accelerate a model with TensorFlow-TensorRT and deploy it on Triton Inference Server.

![Flow](./img/3.PNG)

### Using TensorRT's integration with ONNX RunTime

There are three options to accelerate the ONNX runtime: with `TensorRT` and `CUDA` execution providers for GPU and with `OpenVINO`(discussed in later section) for CPU. 

In general TensorRT will provide better optimizations than the CUDA execution provider however, this depends on the exact structure of the model, more precisely, it depends in the operators used in the network being accelerated. If all the operators are supported, conversion to TensorRT will yield better performance. When `TensorRT` is selected as the accelerater, all supported subgraphs are accelerated by TensorRT and the rest of the graph runs on the CUDA execution provider. Users can achieve this with the following additions to the config file.

**TensorRT acceleration**
```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
  }
}
```
That said, users can also choose to run models without TensorRT optimization, in which case the CUDA EP is the default Execution Provider. More details can be found [here](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-tensorrt-optimization). Refer to the `onnx_tensorrt_config.pbtxt` here for a sample configuration file for the `Text Recognition` model used in Part 1-3 of this series.

There are a few other ONNX runtime specific optimizations. Refer to this section of our [ONNX backend docs](https://github.com/triton-inference-server/onnxruntime_backend#other-optimization-options-with-onnx-runtime) for more information.

## CPU Based Acceleration
Triton Inference Server also supports acceleration for CPU only model with [OpenVINO](https://docs.openvino.ai/latest/index.html). In confugration file, users can add the following to enable CPU acceleration.
```
optimization { 
  execution_accelerators {
    cpu_execution_accelerator : [{
      name : "openvino"
    }]
  }
}
```
While OpenVINO provides software level optimizations, it is also important to consider the CPU hardware being used. CPUs comprise multiple cores, memory resources, and interconnects. With multiple CPUs these resources can be shared with NUMA (Non uniform memory access).
Refer this [section of the Triton Documentation](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#numa-optimization) for more.

## Accelerating Shallow models
Shallow models like Gradient Boosted Decision Trees are often used in many pipelines. These models are typically built with libraries like [XGBoost](https://xgboost.readthedocs.io/en/stable/), [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/), [Scikit-learn](https://scikit-learn.org/stable/), [cuML](https://github.com/rapidsai/cuml) and more. These models can be deployed on the Triton Inference Server via the Forest Infernce Library backend. Check out [these examples](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks) for more information.

## Accelerating Large Tranformer Models
On the other end of the spectrum, Deep Learning practicioners are drawn to Large Transformer based models with billions of parameters. With models at that scale, often times they need different types of optimization or to be parallelized across GPUs. This parallization across GPUs(as they may not fit on 1 GPU) can be achieved either via Tensor parallelism or Pipeline parallelism. To solve this issue, users can use the [Faster Transformer Library](https://github.com/NVIDIA/FasterTransformer/) and Triton's [Faster Transformer Backend](https://github.com/triton-inference-server/fastertransformer_backend). [Check out this blog](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/) for more information!

## Working Example
Before proceeding, please set up a model repository for the Text Recognition model being used in Part 1-3 of this series. Then, navigate to the model repository and launch two containers:

```
# Server Container
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.11-py3 bash

# Client Container (on a different terminal)
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.11-py3-sdk bash
```

Since this is a model we converted to ONNX, and TensorRT acceleration examples are linked throughout the explanation, we will explore the ONNX pathway. There are three cases to consider with ONNX backend:
* Accelerated ONNX RT execution on GPU w. CUDA execution provider: `ORT_cuda_ep_config.pbtxt`
* ONNX RT execution on GPU w. TRT acceleration: `ORT_TRT_config.pbtxt`
* ONNX RT execution on CPU w. OpenVINO acceleration: `ORT_openvino_config.pbtxt`

While using ONNX RT there are some [general optimizations](https://github.com/triton-inference-server/onnxruntime_backend#other-optimization-options-with-onnx-runtime) to consider, irrespective of the Execution provider. These can be graph level optimizations, or selecting the number and behavior of the threads used to parallelize the execution or some memory usage optimzations. The use of each of these options is highly dependent on the model being deployed. 

With this context, let's launch the Triton Inference Server with the appropriate configuration file.

```
tritonserver --model-repository=/models
```
**NOTE: These benchmarks are just to illustrate the general curve of the performance gain. This is not the highest throughput obtainable via Triton as resource utilization features haven't been enabled (eg. Dynamic Batching). Refer to the Model Analyzer tuturial for the best deployment configuration once model optimization are done.**

**NOTE**: These settings are to maximize throughput. Refer to the Model Analyzer tutorial which covers managing latency requirements.

For reference, the baseline performance is as follows:
```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 4191.7 infer/sec, latency 7633 usec
```

### ONNX RT execution on GPU w. CUDA execution provider

For this model, an exhaustive search for the best convolution algorithm is enabled. [Learn about more options](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-cuda-execution-provider-optimization).

```
## Additions to Config
parameters { key: "cudnn_conv_algo_search" value: { string_value: "0" } }
parameters { key: "gpu_mem_limit" value: { string_value: "4294967200" } }

## Perf Analyzer Query
perf_analyzer -m text_recognition -b 16 --shape input.1:1,32,100 --concurrency-range 64
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 4257.9 infer/sec, latency 7672 usec
```

### ONNX RT execution on GPU w. TRT acceleration
While specifying the use of TensorRT Execution Provider, the CUDA Execution provider is used as a fallback for operators not supported by TensorRT. It is recommended to use TensorRT natively if all operators are supported as the performance boost and optimization options are considerably better. In this case, TensorRT accelerator has been used with lower `FP16` precision. 

```
## Additions to Config
optimization {
  graph : {
    level : 1
  }
 execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt",
      parameters { key: "precision_mode" value: "FP16" },
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
  }
}

## Perf Analyzer Query
perf_analyzer -m text_recognition -b 16 --shape input.1:1,32,100 --concurrency-range 2
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 11820.2 infer/sec, latency 2706 usec
```

### ONNX RT execution on CPU w. OpenVINO acceleration

Triton users can also use OpenVINO for CPU deployment. This can be enabled via the following: 

```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "openvino"
  } ]
}}
```
As compare 1 CPU with 1 GPU is not an apples to apples comparision for most cases, we encorage benchmarking on user's local CPU hardware. [Learn more](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-openvino-optimization)

There are many other features that for each backend which can be enabled depending on the needs of specific models. Refer to [this protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) for the complete list of possible features and optimizations.

## Model Navigator

The sections above describe converting models and using different accelerators and provide a "general guideline" to build an intution about which "path" to take while considering optimizations. These are manual explorations that consume considerable time. To check the conversion coverage and explore a subset of the optimization possible, users can make use of the [Model Navigator Tool](https://github.com/triton-inference-server/model_navigator). 

# What's next?

In this tutorial, we covered a plethora of optimization options available to accelerate models while using the Triton Inference Server. This is Part 4 of a 10 part tutorial series which covers the challenges faced in deploying Deep Learning models to production. Part 5 covers `Building a model ensemble`. Part 3 and Part 4 focus on two different aspects, resource utilizations and framework level model acceleration respectively. Using both of these techniques in conjuntion will lead to the best performance possible. Since the specific selections are highly dependent on workloads, models, SLAs, and hardware resources, this process varies for each user. We highly encorage users to experiment with all these features to find our the best deployment configuration for their usecase.