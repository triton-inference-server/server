<!--
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Using Triton with Inferentia 1

Starting from 21.11 release, Triton supports
[AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)
and the [Neuron Runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html).

## Table of Contents

- [Using Triton with Inferentia 1](#using-triton-with-inferentia-1)
  - [Table of Contents](#table-of-contents)
  - [Inferentia setup](#inferentia-setup)
  - [Setting up the Inferentia model](#setting-up-the-inferentia-model)
    - [PyTorch](#pytorch)
    - [TensorFlow](#tensorflow)
  - [Serving Inferentia model in Triton](#serving-inferentia-model-in-triton)
    - [Using Triton's Dynamic Batching](#using-tritons-dynamic-batching)
  - [Testing Inferentia Setup for Accuracy](#testing-inferentia-setup-for-accuracy)

## Inferentia setup

First step of running Triton with Inferentia is to create an AWS Inferentia
 instance with Deep Learning AMI (tested with Ubuntu 18.04).
`ssh -i <private-key-name>.pem ubuntu@<instance address>`
Note: It is recommended to set your storage space to greater than default value
of 110 GiB. The current version of Triton has been tested
with storage of 500 GiB.

After logging into the inf1* instance, you will need to clone
[this current Github repo](https://github.com/triton-inference-server/python_backend).
 Follow [steps on Github to set up ssh access](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
or simply clone with https.
Clone this repo with Github to home repo `/home/ubuntu`.

```
 chmod 777 /home/ubuntu/python_backend/inferentia/scripts/setup-pre-container.sh
 sudo /home/ubuntu/python_backend/inferentia/scripts/setup-pre-container.sh
```

Then, start the Triton instance with:
```
 docker run --device /dev/neuron0 <more neuron devices> -v /home/ubuntu/python_backend:/home/ubuntu/python_backend -v /lib/udev:/mylib/udev --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```
Note 1: The user would need to list any neuron device to run during container initialization.
For example, to use 4 neuron devices on an instance, the user would need to run with:
```
 docker run --device /dev/neuron0 --device /dev/neuron1 --device /dev/neuron2 --device /dev/neuron3 ...`
```
Note 2: `/mylib/udev` is used for Neuron parameter passing.

Note 3: For Triton container version xx.yy, please refer to
[Triton Inference Server Container Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html).
 The current build script has been tested with container version `21.10`.

After starting the Triton container, go into the `python_backend` folder and run the setup script.
```
 source /home/ubuntu/python_backend/inferentia/scripts/setup.sh
```
This script will:
1. Install necessary dependencies
2. Install [neuron-cc](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-cc/index.html), the Neuron compiler.
3. Install neuron framework packages as per your preference e.g., either pytorch, or tensorflow or both.

There are user configurable options available for the script as well.
Please use the `-h` or `--help` options to learn about more configurable options.

## Setting up the Inferentia model

Currently, we only support [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html)
and [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/index.html)
workflows for execution on inferentia.

The user is required to create their own `*.pt` (for pytorch) or `*.savedmodels`
(for tensorflow) models. This is a critical step since Inferentia will need
the underlying `.NEFF` graph to execute the inference request. Please refer to:

- [Neuron compiler CLI Reference Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-cc/command-line-reference.html)
- [PyTorch-Neuron trace python API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.html)
- [PyTorch Tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/index.html)
- [TensorFlow Tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/index.html)
for guidance on how to compile models.

### PyTorch

For PyTorch, we support models traced by [PyTorch-Neuron trace python API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.html)
for execution on Inferentia.
Once the TorchScript model supporting Inferentia is obtained, use the
[gen_triton_model.py](scripts/gen_triton_model.py) script to generate
triton python model directory.

An example invocation for the `gen_triton_model.py` for PyTorch model can look like:

```
 python3 inferentia/scripts/gen_triton_model.py --model_type pytorch --triton_input INPUT__0,INT64,4x384 INPUT__1,INT64,4x384 INPUT__2,INT64,4x384 --triton_output OUTPUT__0,INT64,4x384 OUTPUT__1,INT64,4x384 --compiled_model /home/ubuntu/bert_large_mlperf_neuron_hack_bs1_dynamic.pt --neuron_core_range 0:3 --triton_model_dir bert-large-mlperf-bs1x4
```

In order for the script to treat the compiled model as TorchScript
model, `--model_type pytorch` needs to be provided.

NOTE: Due to the absence of metadata for inputs and outputs in a
TorchScript model - name, datatype and shape of tensor of
both the inputs and outputs must be provided to the above script
and the name must follow a specific naming convention i.e.
`<name>__<index>`. Where `<name>` can be any string and `<index>`
refers to the position of the corresponding input/output. This
means if there are two inputs and two outputs they must be named
as: "INPUT__0", "INPUT__1" and "OUTPUT__0", "OUTPUT__1" such
that "INPUT__0" refers to first input and INPUT__1 refers to the
second input, etc.

Additionally, `--neuron_core_range` specifies the neuron cores to
be used while serving this models. Currently, only
`torch.neuron.DataParallel()` mode is supported. See
[Data Parallel Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/appnotes/perf/torch-neuron-dataparallel-app-note.html)
for more information. Triton model instance count can be specified
by using  `--triton_model_instance_count` option. The neuron
cores will be equally distributed among all instances. For example,
in case of two triton model instances and 4 neuron cores, the first
instance will be loaded on on cores 0-1 and second instance will be
loaded on cores 2-3. To best engage inferentia device, try setting
the number of neuron cores to be a proper multiple of the instance
count.

### TensorFlow

For TensorFlow, the model must be compiled for AWS Neuron. See
[AWS Neuron TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/index.html)
tutorials to learn how to get a compiled model that uses Neuron
cores. Currently, the code is tested only on `tensorflow==1.15`.

Once the compiled model is obtained use [gen_triton_model.py](scripts/gen_triton_model.py)
script to generate triton python model directory.

An example invocation for the `gen_triton_model.py` for TensorFlow model can look like:

```
 python3 gen_triton_model.py --model_type tensorflow --compiled_model /home/ubuntu/inferentia-poc-2.0/scripts-rn50-tf-native/resnet50_mlperf_opt_fp16_compiled_b5_nc1/1 --neuron_core_range 0:3  --triton_model_dir rn50-1neuroncores-bs1x1
```

NOTE: Unlike TorchScript model, TensorFlow SavedModel stores sufficient
metadata to detect the name, datatype and shape of the input and output
tensors for the model. By default, the script will assume the compiled
model to be torchscript. In order for it to treat the compiled model
as TF savedmodel, `--model_type tensorflow` needs to be provided.
The input and output details are read from the model itself. The user
must have [`tensorflow`](https://www.tensorflow.org/install/pip) python
module installed in order to use this script for tensorflow models.

Similar to PyTorch, `--neuron_core_range` and `--triton_model_instance_count`
can be used to specify the neuron core range and number of triton model
instances. However, the neuron core indices don't point to a specific
neuron core in the chip. For TensorFlow, we use deprecated feature of
`NEURONCORE_GROUP_SIZES` to load model. The model in this case will be loaded on
next available Neuron cores and not specific ones. See
[Parallel Execution using NEURONCORE_GROUP_SIZES](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/appnotes/perf/parallel-ncgs.html?highlight=NEURONCORE_GROUP_SIZES)
for more information.

Another note, since Neuron-Tensorflow(unlike Neuron-Python) does not have
built-in functions for running a model for multiple cores, `model.py` will
distribute the workload by splitting the input tensor across available cores.
It is recommended the first dimension for the inputs be `None` if the user enables
processing across multiple cores.

Please use the `-h` or `--help` options in `gen_triton_model.py` to
learn about more configurable options.

## Serving Inferentia model in Triton

The `gen_triton_model.py` should create a triton model directory with following
structutre:

```
bert-large-mlperf-bs1x4
 |
 |- 1
 |  |- model.py
 |
 |- config.pbtxt
```

Look at the usage message of the script to understand each option.

The script will generate a model directory with the user-provided
name. Move that model directory to Triton's model repository.
Ensure the compiled model path provided to the script points to
a valid torchscript file or tensorflow savedmodel.

Now, the server can be launched with the model as below:

```
 tritonserver --model-repository <path_to_model_repository>
```

Note:

1. The `config.pbtxt` and `model.py` should be treated as
starting point. The users can customize these files as per
their need.
2. Triton Inferentia is currently tested with a **single** model.

### Using Triton's Dynamic Batching

To enable dynamic batching, `--enable_dynamic_batching`
flag needs to be specified. `gen_triton_model.py` supports following three
options for configuring [Triton's dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md):

1. `--preferred_batch_size`: Please refer to [model configuration documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#preferred-batch-sizes) for details on preferred batch size. To optimize
   performance, this is recommended to be multiples of engaged neuron cores.
   For example, if each instance is using 2 neuron cores, `preferred_batch_size`
   could be 2, 4 or 6.
2. `--max_queue_delay_microseconds`: Please refer to
   [model configuration documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching) for details.
3. `--disable_batch_requests_to_neuron`: Enable the non-default way for Triton to
   handle batched requests. Triton backend will send each request to neuron
   separately, irrespective of if the Triton server requests are batched.
   This flag is recommended when users want to optimize performance with models
   that do not perform well with batching without the flag.

Additionally, `--max_batch_size` will affect the maximum batching limit. Please
refer to the
[model configuration documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size)
for details.

## Testing Inferentia Setup for Accuracy

The [qa folder](https://github.com/triton-inference-server/python_backend/tree/main/inferentia/qa)
contains the necessary files to set up testing with a simple add_sub model. The test
requires an instance with more than 8 inferentia cores to run, eg:`inf1.6xlarge`.
start the test, run
```
 source <triton path>/python_backend/inferentia/qa/setup_test_enviroment_and_test.sh
```
where `<triton path>` is usually `/home/ubuntu`/.
This script will pull the [server repo](https://github.com/triton-inference-server/server)
that contains the tests for inferentia. It will then build the most recent
Triton Server and Triton SDK.

Note: If you would need to change some of the tests in the server repo,
you would need to run
```
 export TRITON_SERVER_REPO_TAG=<your branch name>
```
before running the script.

# Using Triton with Inferentia 2, or Trn1
## pytorch-neuronx and tensorflow-neuronx
1. Similar to the steps for inf1, change the argument to the pre-container and on-container setup scripts to include the `-inf2` or `-trn1`flags e.g.,
```
 chmod 777 /home/ubuntu/python_backend/inferentia/scripts/setup-pre-container.sh
 sudo /home/ubuntu/python_backend/inferentia/scripts/setup-pre-container.sh -inf2
```
2. On the container, followed by the `docker run` command, you can pass similar argument to the setup.sh script
For Pytorch:
```
source /home/ubuntu/python_backend/inferentia/scripts/setup.sh -inf2 -p
```
For Tensorflow:
```
source /home/ubuntu/python_backend/inferentia/scripts/setup.sh -inf2 -t
```
3. Following the above steps, when using the `gen_triton_model.py` script, you can pass similar argument `--inf2` to the setup.sh script e.g., for Pytorch
```
python3 inferentia/scripts/gen_triton_model.py --inf2 --model_type pytorch --triton_input INPUT__0,INT64,4x384 INPUT__1,INT64,4x384 INPUT__2,INT64,4x384 --triton_output OUTPUT__0,INT64,4x384 OUTPUT__1,INT64,4x384 --compiled_model bert_large_mlperf_neuron_hack_bs1_dynamic.pt --neuron_core_range 0:3 --triton_model_dir bert-large-mlperf-bs1x4
```
4. **Note**: When using the `--inf2` option, the `--compiled_model` path should be provided relative to the triton model directory. The `initialize()` function in model.py will derive the full path by concatenating the model path within the repository and the relative `--compiled_model` path.
## transformers-neuronx
To use inf2/trn1 instances with transformers-neuronx packages for serving models, generate a `pytorch` model as per above instructions. The transformers-neuronx currently supports the models listed [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/transformers-neuronx/readme.html#currently-supported-models).

As prescribed on the neuronx documentation page, while the neuronx load API differs per model, it follows the same pattern.

1. To serve transformers-neuronx models, first trace the model using `save_pretrained_split()` API on an inf2 instance (recommend inf2.24xl for Large Language Models). Following that, package the folder as the '--compiled_model' when using `gen_triton_model.py` file.
2. The following tree shows a sample model structure for OPT model:
```
opt/
├── 1
│   └── model.py
├── opt-125m-model
│   └── pytorch_model.bin
└── opt-125m-tp12
    ├── FullyUnrolled.1814.1
    │   ├── penguin-sg0000
    │   └── sg00
    ├── FullyUnrolled.1814.2
    │   ├── penguin-sg0000
    │   └── sg00
    ├── FullyUnrolled.1814.3
    │   ├── penguin-sg0000
    │   └── sg00
    ├── FullyUnrolled.1814.4
    │   ├── penguin-sg0000
    │   └── sg00
    └── FullyUnrolled.1814.5
        ├── penguin-sg0000
        └── sg00
  ├── config.pbtxt
```

3. Add the following imports (e.g., for OPT model). The import will differ as per the model you're trying to run.
```
from transformers_neuronx.opt.model import OPTForSampling
```

4. Add the following lines in `initialize()` function. Set the `batch_size`, `tp_degree`, `n_positions`, `amp` and `unroll` args as per your requirement. `tp_degree` should typically match the number of neuron cores available on inf2 instance.
```
batch_size = 1
tp_degree = 12
n_positions = 2048
amp = 'bf16'
unroll = None
self.model_neuron = OPTForSampling.from_pretrained(compiled_model, batch_size=batch_size, amp=amp, tp_degree=tp_degree, n_positions=n_positions, unroll=unroll)
self.model_neuron.to_neuron()

self.model_neuron.num_workers = num_threads
```
You may also chose to add the `batch_size` etc. arguments to config.pbtxt as parameters and read them in the `initialize()` function similar to `--compiled-model`.

5. Finally, in the `execute()` function, use the following API to run the inference:
```
batched_results = self.model_neuron.sample(batched_tensor, 2048)
```
Above, `2048` is a sufficiently-long output token. It may also be passed in as one of the inputs if you wanto specify it as part of the payload.

6. Proceed to load the model, and submit the inference payload similar to any other triton model.