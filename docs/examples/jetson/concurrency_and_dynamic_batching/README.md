<!--
# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Concurrent inference and dynamic batching

The purpose of this sample is to demonstrate the important features of Triton Inference Server such as concurrent model execution and dynamic batching.

We will be using a purpose built deployable people detection model, which we download from [Nvidia GPU Cloud (NGC)](https://ngc.nvidia.com/).

## Acquiring the model

Download the pruned [PeopleNet](https://ngc.nvidia.com/catalog/models/nvidia:tlt_peoplenet) model from the NGC. This model is available as a ready-to-use model, and you can download it from NGC using either `wget` method:

```shell
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_v2.1/zip -O pruned_v2.1.zip
```

or via CLI command:

```shell
ngc registry model download-version "nvidia/tao/peoplenet:pruned_v2.1"
```

For latter you need to setup the [NGC CLI](https://ngc.nvidia.com/setup).

Having downloaded the model from the NGC, unzip the archive `peoplenet_pruned_v2.1.zip` into `concurrency_and_dynamic_batching/tao/models/peoplenet`.

If you have the zip archive in the `concurrency_and_dynamic_batching` directory, the following will automatically place the model to the correct location:

```shell
unzip pruned_v2.1.zip -d $(pwd)/tao/models/peoplenet
```

Verify that you can see the model file `resnet34_peoplenet_pruned.etlt` under

```
concurrency_and_dynamic_batching
└── tao
       └── models
           └── peoplenet
               ├── labels.txt
               └── resnet34_peoplenet_pruned.etlt
```

## Converting the model to TensorRT

After you have acquired the model file in `.etlt` format, you will need to convert the model to [TensorRT](https://developer.nvidia.com/tensorrt) format. NVIDIA TensorRT is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications. The latest versions of JetPack include TensorRT.

In order to convert an `.etlt` model to TensorRT format, you need to use the `tao-converter` tool.

The `tao-converter` tool is available as a compiled release file for different platforms. The download links corresponding to your deployment system are provided among the [TLT Getting Started resources](https://developer.nvidia.com/tlt-get-started).

After you have downloaded `tao-converter`, you might need to execute

```shell
chmod 777 tao-converter
```

in the directory with the tool.

We provide a conversion script `tao/convert_peoplenet.sh` which expects the model to be present at the location.

```shell
tao
└──  models
   └── peoplenet
```

To execute it, you can place the `tao-converter` executable to the `tao` directory of the project and in the same directory run

```shell
bash convert_peoplenet.sh
```

After you execute it, verify that a `model.plan` file was placed to to the directories `/trtis_model_repo_sample_1/peoplenet/1` and `/trtis_model_repo_sample_2/peoplenet/1`. Note that we have two slightly different repositories for the same model to demonstrate different features of Triton.

Also note that this step has to be performed on the target hardware: if you are planning to execute this application on Jetson, the conversion has to be performed on Jetson.

To learn more about `tao-converter`parameters, run:

```shell
./tao-converter -h
```

## Building the app

To compile the sample, pull the following repositories:
* [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)
* [https://github.com/triton-inference-server/core](https://github.com/triton-inference-server/core)

Make sure you copied the contents of the release you downloaded to `$HOME`

```shell
sudo cp -rf tritonserver2.x.y-jetpack4.6 $HOME/tritonserver
```

Open the terminal in `concurrency_and_dynamic_batching` and build the app executing

```shell
make
```

An example Makefile is provided for Jetson.

## Demonstration  case 1: Concurrent model execution

With Triton Inference Server, multiple models (or multiple instances of the same model) can run simultaneously on the same GPU or on multiple GPUs. In this example, we are demonstrating how to run multiple instances of the same model on a single Jetson GPU.

### Running the sample

To execute from the terminal, run from the `concurrency_and_dynamic_batching` directory:

```shell
LD_LIBRARY_PATH=$HOME/tritonserver/lib ./people_detection -m system -v -r $(pwd)/trtis_model_repo_sample_1 -t 6 -s false -p $HOME/tritonserver
```

The parameter `-t` controls the number of concurrent inference calls we want to execute. We will be executing the same model on the same sample image with the purpose of demonstrating how setting different concurrency options affects the performance.

You can enable saving detected bounding boxes in the project directory in form of overlays over the original image for each execution thread. You can turn the visualization on by setting the parameter `-s` to `true` upon execution (`-s` is set to `false` by default).

### Expected output

Upon execution, in the terminal log you will see _Model 'peoplenet' Stats_ in json format reflecting the inference performance. We also output _TOTAL INFERENCE TIME_ which simply reflects the elapsed time required to run the application including data loading, pre-processing and post-processing.

A typical output in the log for _Model 'peoplenet' Stats_ looks as follows:

```json
{
   "model_stats":[
      {
         "name":"peoplenet",
         "version":"1",
         "last_inference":1626448309997,
         "inference_count":6,
         "execution_count":6,
         "inference_stats":{
            "success":{
               "count":6,
               "ns":574589968
            },
            "fail":{
               "count":0,
               "ns":0
            },
            "queue":{
               "count":6,
               "ns":234669630
            },
            "compute_input":{
               "count":6,
               "ns":194884512
            },
            "compute_infer":{
               "count":6,
               "ns":97322636
            },
            "compute_output":{
               "count":6,
               "ns":47700806
            }
         },
         "batch_stats":[
            {
               "batch_size":1,
               "compute_input":{
                  "count":6,
                  "ns":194884512
               },
               "compute_infer":{
                  "count":6,
                  "ns":97322636
               },
               "compute_output":{
                  "count":6,
                  "ns":47700806
               }
            }
         ]
      }
   ]
}

"TOTAL INFERENCE TIME: 174ms"
```

To learn about different statistics check out the [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md#statistics-extension).

To see how setting different values for concurrency affects total execution time and its components reflected in the model stats, you need to modify a single parameter in the model config file.

To enable concurrent model execution support for a model, corresponding model config file `trtis_model_repo_sample_1/peoplenet/config.pbtxt` includes the following:

```
instance_group [
  {
    count: 3
    kind: KIND_GPU
  }
]
```

You can change the count of allowed inferences for the same model instance and observe how it affects performance in _Model 'peoplenet' Stats_ and _TOTAL INFERENCE TIME_. Note that on Jetson we dont recommend setting values too high: for instance, on a device like a Jetson Xavier AGX we don't recommend setting the number larger than 6. The values in the range 1-3 are optimal.

While trying out different values, note how it affects total inference time as well as some inference statistics (like queue and compute times)

## Demonstration case 2: Dynamic batching

For models that support batching, Triton implements multiple scheduling and batching algorithms that combine individual inference requests together to improve inference throughput. In this example, we want to demonstrate how enbling automatic dynamic batching affects inference performance.

### Running the sample

To observe the effect of dynamic batching, from the `concurrency_and_dynamic_batching` directory execute:

```shell
LD_LIBRARY_PATH=$HOME/tritonserver/lib ./people_detection -m system -v -r $(pwd)/trtis_model_repo_sample_2 -t 6 -s false -p $HOME/tritonserver
```

### Expected output

Take a look at _Model 'peoplenet' Stats_ and _TOTAL INFERENCE TIME_ to see the effect of dynamic batching. A possible outcome should look like that:

```json
{
   "model_stats":[
      {
         "name":"peoplenet",
         "version":"1",
         "last_inference":1626447787832,
         "inference_count":6,
         "execution_count":2,
         "inference_stats":{
            "success":{
               "count":6,
               "ns":558981051
            },
            "fail":{
               "count":0,
               "ns":0
            },
            "queue":{
               "count":6,
               "ns":49271380
            },
            "compute_input":{
               "count":6,
               "ns":170634044
            },
            "compute_infer":{
               "count":6,
               "ns":338079193
            },
            "compute_output":{
               "count":6,
               "ns":950544
            }
         },
         "batch_stats":[
            {
               "batch_size":1,
               "compute_input":{
                  "count":1,
                  "ns":15955684
               },
               "compute_infer":{
                  "count":1,
                  "ns":29917093
               },
               "compute_output":{
                  "count":1,
                  "ns":152264
               }
            },
            {
               "batch_size":5,
               "compute_input":{
                  "count":1,
                  "ns":30935672
               },
               "compute_infer":{
                  "count":1,
                  "ns":61632420
               },
               "compute_output":{
                  "count":1,
                  "ns":159656
               }
            }
         ]
      }
   ]
}

"TOTAL INFERENCE TIME: 162ms"
```

Notice that this time the model was executed only twice (as indicated by `execution_count`). Also, unlike in the previous example, the `batch_stats` part of the statitstics looks different: we see that our model was executed one time with `batch = 1` and the second time with `batch = 5`. It helped to decrease the total inference time.

In order to enable dynamic batching, the following is present in the model config `trtis_model_repo_sample_2/peoplenet/config.pbtxt`:

```
dynamic_batching {
}
```

To try further options of dynamic batcher see the [documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher).

You can also try enabling both concurrent model execution and dynamic batching.