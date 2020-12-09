<!--
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

# Model Repository

The Triton Inference Server serves models from one or more model
repositories that are specified when the server is stated. While
Triton is running, the models being served can be modified as
described in [Model Management](model_management.md).

## Repository Layout

These repository paths are specified when Triton is started using the
--model-repository option. The --model-repository option can be
specified multiple times to included models from multiple
repositories. The directories and files that compose a model
repository must follow a required layout. Assuming a repository path
is specified as follows.

```bash
$ tritonserver --model-repository=<model-repository-path>
```

The corresponding repository layout must be:

```
  <model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

Within the top-level model repository directory there must be zero or
more <model-name> sub-directories. Each of the <model-name>
sub-directories contains the repository information for the
corresponding model. The config.pbtxt file describes the [model
configuration](model_configuration.md) for the model. For some models,
config.pbtxt is required while for others it is optional. See
[Auto-Generated Model
Configuration](#auto-generated-model-configuration) for more
information.

Each <model-name> directory must have at least one numeric
sub-directory representing a version of the model.  For more
information about how the model versions are handled by Triton see
[Model Versions](#model-versions).  Each model is executed by a
specific
[backend](https://github.com/triton-inference-server/backend/blob/main/README.md).
Within each version sub-directory there must be the files required by
that backend. For example, models that use framework backends such as
PyTorch, ONNX and TensorFlow must provide the [framework-specific
model files](#model-files).

## Model Repository Locations

Triton can access models from one or more locally accessible file
paths, from Google Cloud Storage, from Amazon S3, and from Azure
Storage.

### Local File System

For a locally accessible file-system the absolute path must be
specified.

```bash
$ tritonserver --model-repository=/path/to/model/repository ...
```

### Google Cloud Storage

For a model repository residing in Google Cloud Storage, the
repository path must be prefixed with gs://.

```bash
$ tritonserver --model-repository=gs://bucket/path/to/model/repository ...
```

### S3

For a model repository residing in Amazon S3, the path must be
prefixed with s3://.

```bash
$ tritonserver --model-repository=s3://bucket/path/to/model/repository ...
```

For a local or private instance of S3, the prefix s3:// must be
followed by the host and port (separated by a semicolon) and
subsequently the bucket path.

```bash
$ tritonserver --model-repository=s3://host:port/bucket/path/to/model/repository ...
```

When using S3, the credentials and default region can be passed by
using either the [aws
config](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
command or via the respective [environment
variables](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).
If the environment variables are set they will take a higher priority
and will be used by Triton instead of the credentials set using the
aws config command.

### Azure Storage

For a model repository residing in Azure Storage, the repository path
must be prefixed with as://.

```bash
$ tritonserver --model-repository=as://account_name/container_name/path/to/model/repository ...
```

## Model Versions

Each model can have one or more versions available in the model
repository. Each version is stored in its own, numerically named,
subdirectory where the name of the subdirectory corresponds to the
version number of the model. The subdirectories that are not
numerically named, or have names that start with zero (0) will be
ignored. Each model configuration specifies a [version
policy](model_configuration.md#version-policy) that controls which of
the versions in the model repository are made available by Triton at
any given time.

## Model Files

The contents of each model version sub-directory is determined by the
type of the model and the requirements of the
[backend](https://github.com/triton-inference-server/backend/blob/main/README.md)
that supports the model.

### TensorRT Models

A TensorRT model definition is called a *Plan*. A TensorRT Plan is a
single file that by default must be named model.plan. This default
name can be overridden using the *default_model_filename* property in
the [model configuration](model_configuration.md).

A TensorRT Plan is specific to a GPU's [CUDA Compute
Capability](https://developer.nvidia.com/cuda-gpus).  As a result,
TensorRT models will need to set the *cc_model_filenames* property in
the [model configuration](model_configuration.md) to associate each
Plan file with the corresponding Compute Capability.

A minimal model repository for a TensorRT model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.plan
```

### ONNX Models

An ONNX model is a single file or a directory containing multiple
files. By default the file or directory must be named model.onnx.
This default name can be overridden using the *default_model_filename*
property in the [model configuration](model_configuration.md).

Triton supports all ONNX models that are supported by the version of
[ONNX Runtime](https://github.com/Microsoft/onnxruntime) being used by
Triton. Models will not be supported if they use a [stale ONNX opset
version](https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix)
or [contain operators with unsupported
types](https://github.com/microsoft/onnxruntime/issues/1122).

A minimal model repository for a ONNX model contained in a single file
is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx
```

An ONNX model composed from multiple files must be contained in a
directory.  By default this directory must be named model.onnx but can
be overridden using the *default_model_filename* property in the
[model configuration](model_configuration.md). The main model file
within this directory must be named model.onnx. A minimal model
repository for a ONNX model contained in a directory is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx/
           model.onnx
           <other model files>
```

### TorchScript Models

An TorchScript model is a single file that by default must be named
model.pt. This default name can be overridden using the
*default_model_filename* property in the [model
configuration](model_configuration.md). It is possible that some
models traced with different versions of PyTorch may not be supported
by Triton due to changes in the underlying opset.

A minimal model repository for a TorchScript model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pt
```

### TensorFlow Models

TensorFlow saves models in one of two formats: *GraphDef* or
*SavedModel*. Triton supports both formats.

A TensorFlow GraphDef is a single file that by default must be named
model.graphdef. A TensorFlow SavedModel is a directory containing
multiple files. By default the directory must be named
model.savedmodel. These default names can be overridden using the
*default_model_filename* property in the [model
configuration](model_configuration.md).

A minimal model repository for a TensorFlow
GraphDef model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef
```

A minimal model repository for a TensorFlow SavedModel model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>
```

### Python Models

The [Python
backend](https://github.com/triton-inference-server/python_backend)
allows you to run Python code as a model within Triton. By default the
Python script must be named model.py but this default name can be
overridden using the *default_model_filename* property in the [model
configuration](model_configuration.md).

A minimal model repository for a Python model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.py
```

### DALI Models

The [DALI backend](https://github.com/triton-inference-server/dali_backend)
allows you to run a [DALI pipeline](https://github.com/NVIDIA/DALI) as
a model within Triton. In order to use this backend, you need to generate
a file, by default named `model.dali`, and include it in your model repository.
Please refer to [DALI backend documentation
](https://github.com/triton-inference-server/dali_backend#how-to-use) for the
description, how to generate `model.dali`. The default model file name can be
overridden using the *default_model_filename* property in the
[model configuration](model_configuration.md).

A minimal model repository for a DALI model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.dali
```
