<!--
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

**Is this your first time setting up a model repository?** Check out
[these tutorials](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#setting-up-the-model-repository)
 to begin your Triton journey!

The Triton Inference Server serves models from one or more model
repositories that are specified when the server is started. While
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
Configuration](model_configuration.md#auto-generated-model-configuration)
for more information.

Each <model-name> directory must have at least one numeric
sub-directory representing a version of the model.  For more
information about how the model versions are handled by Triton see
[Model Versions](#model-versions).  Each model is executed by a
specific
[backend](https://github.com/triton-inference-server/backend/blob/main/README.md).
Within each version sub-directory there must be the files required by
that backend. For example, models that use framework backends such as
TensorRT, PyTorch, ONNX, OpenVINO and TensorFlow must provide the
[framework-specific model files](#model-files).

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

### Cloud Storage with Environment variables

#### Google Cloud Storage

For a model repository residing in Google Cloud Storage, the
repository path must be prefixed with gs://.

```bash
$ tritonserver --model-repository=gs://bucket/path/to/model/repository ...
```

When using Google Cloud Storage, credentials are fetched and attempted in the
following order:
1. [GOOGLE_APPLICATION_CREDENTIALS environment variable](https://cloud.google.com/docs/authentication/application-default-credentials#GAC)
   - The environment variable should be set and contains the location of a
credential JSON file.
   - Authorized user credential will be attempted first, and then service
account credential.
2. [The attached service account](https://cloud.google.com/docs/authentication/application-default-credentials#attached-sa)
   - A value for the
[Authorization HTTP header](https://googleapis.dev/cpp/google-cloud-storage/1.42.0/classgoogle_1_1cloud_1_1storage_1_1oauth2_1_1ComputeEngineCredentials.html#a8c3a5d405366523e2f4df06554f0a676)
should be obtainable.
3. Anonymous credential (also known as public bucket)
   - The bucket (and objects) should have granted `get` and `list` permission to
all users.
   - One way to grant such permission is by adding both
[storage.objectViewer](https://cloud.google.com/storage/docs/access-control/iam-roles#standard-roles)
and
[storage.legacyBucketReader](https://cloud.google.com/storage/docs/access-control/iam-roles#legacy-roles)
predefined roles for "allUsers" to the bucket, for example:
        ```
        $ gsutil iam ch allUsers:objectViewer "${BUCKET_URL}"
        $ gsutil iam ch allUsers:legacyBucketReader "${BUCKET_URL}"
        ```

By default, Triton makes a local copy of a remote model repository in
a temporary folder, which is deleted after Triton server is shut down.
If you would like to control where remote model repository is copied to,
you may set the `TRITON_GCS_MOUNT_DIRECTORY` environment variable to
a path pointing to the existing folder on your local machine.

```bash
export TRITON_GCS_MOUNT_DIRECTORY=/path/to/your/local/directory
```

**Make sure, that `TRITON_GCS_MOUNT_DIRECTORY` exists on your local machine
and it is empty.**

#### S3

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

By default, Triton uses HTTP to communicate with your instance of S3. If
your instance of S3 supports HTTPS and you wish for Triton to use the HTTPS
protocol to communicate with it, you can specify the same in the model
repository path by prefixing the host name with https://.

```bash
$ tritonserver --model-repository=s3://https://host:port/bucket/path/to/model/repository ...
```

When using S3, the credentials and default region can be passed by
using either the [aws
config](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
command or via the respective [environment
variables](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).
If the environment variables are set they will take a higher priority
and will be used by Triton instead of the credentials set using the
aws config command.

By default, Triton makes a local copy of a remote model repository
in a temporary folder, which is deleted after Triton server is shut down.
If you would like to control where remote model repository is copied to,
you may set the `TRITON_AWS_MOUNT_DIRECTORY` environment variable to
a path pointing to the existing folder on your local machine.

```bash
export TRITON_AWS_MOUNT_DIRECTORY=/path/to/your/local/directory
```

**Make sure, that `TRITON_AWS_MOUNT_DIRECTORY` exists on your local machine
and it is empty.**

#### Azure Storage

For a model repository residing in Azure Storage, the repository path
must be prefixed with as://.

```bash
$ tritonserver --model-repository=as://account_name/container_name/path/to/model/repository ...
```

When using Azure Storage, you must set the `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY`
environment variables to an account that has access to the Azure Storage repository.

If you don't know your `AZURE_STORAGE_KEY` and have your Azure CLI correctly configured,
here's an example of how to find a key corresponding to your `AZURE_STORAGE_ACCOUNT`:

```bash
$ export AZURE_STORAGE_ACCOUNT="account_name"
$ export AZURE_STORAGE_KEY=$(az storage account keys list -n $AZURE_STORAGE_ACCOUNT --query "[0].value")
```
By default, Triton makes a local copy of a remote model repository in
a temporary folder, which is deleted after Triton server is shut down.
If you would like to control where remote model repository is copied to,
you may set the `TRITON_AZURE_MOUNT_DIRECTORY` environment variable to a path
pointing to the existing folder on your local machine.

```bash
export TRITON_AZURE_MOUNT_DIRECTORY=/path/to/your/local/directory
```

**Make sure, that `TRITON_AZURE_MOUNT_DIRECTORY` exists on your local machine
and it is empty.**


### Cloud Storage with Credential file (Beta)

*This feature is currently in beta and may be subject to change.*

To group the credentials into a single file for Triton, you may set the
`TRITON_CLOUD_CREDENTIAL_PATH` environment variable to a path pointing to a
JSON file of the following format, residing in the local file system.

```
export TRITON_CLOUD_CREDENTIAL_PATH="cloud_credential.json"
```

"cloud_credential.json":
```
{
  "gs": {
    "": "PATH_TO_GOOGLE_APPLICATION_CREDENTIALS",
    "gs://gcs-bucket-002": "PATH_TO_GOOGLE_APPLICATION_CREDENTIALS_2"
  },
  "s3": {
    "": {
      "secret_key": "AWS_SECRET_ACCESS_KEY",
      "key_id": "AWS_ACCESS_KEY_ID",
      "region": "AWS_DEFAULT_REGION",
      "session_token": "",
      "profile": ""
    },
    "s3://s3-bucket-002": {
      "secret_key": "AWS_SECRET_ACCESS_KEY_2",
      "key_id": "AWS_ACCESS_KEY_ID_2",
      "region": "AWS_DEFAULT_REGION_2",
      "session_token": "AWS_SESSION_TOKEN_2",
      "profile": "AWS_PROFILE_2"
    }
  },
  "as": {
    "": {
      "account_str": "AZURE_STORAGE_ACCOUNT",
      "account_key": "AZURE_STORAGE_KEY"
    },
    "as://Account-002/Container": {
      "account_str": "",
      "account_key": ""
    }
  }
}
```

To match a credential, the longest matching credential name against the start
of a given path is used. For example: `gs://gcs-bucket-002/model_repository`
will match the "gs://gcs-bucket-002" GCS credential, and
`gs://any-other-gcs-bucket` will match the "" GCS credential.

This feature is intended for use-cases which multiple credentials are needed
for each cloud storage provider. Be sure to replace any credential paths/keys
with the actual paths/keys from the example above.

If the `TRITON_CLOUD_CREDENTIAL_PATH` environment variable is not set, the
[Cloud Storage with Environment variables](#cloud-storage-with-environment-variables)
will be used.

### Caching of Cloud Storage

Triton currently doesn't perform file caching for cloud storage.
However, this functionality can be implemented through
[repository agent API](https://github.com/triton-inference-server/server/blob/bbbcad7d87adc9596f99e3685da5d6b73380514f/docs/customization_guide/repository_agents.md) by injecting a proxy, which checks a specific local directory for caching
given the cloud storage (original path) of the model,
and then decides if cached files may be used.

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

### OpenVINO Models

An OpenVINO model is represented by two files, a *.xml and *.bin
file. By default the *.xml file must be named model.xml. This default
name can be overridden using the *default_model_filename* property in
the [model configuration](model_configuration.md).

A minimal model repository for an OpenVINO model is:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.xml
        model.bin
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
