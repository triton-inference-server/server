..
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

.. _section-model-repository:

Model Repository
================

The Triton Inference Server serves models from one or more model
repositories that are specified when the server is stated. While
Triton is running, the models being served can be modified as
described in :ref:`section-model-management`.

Model Repository Locations
--------------------------

Triton can access models from one or more locally accessible file
paths, from Google Cloud Storage, and from Amazon S3. These repository
paths are specified when Triton is started using the
-\\-model-repository option. The -\\-model-repository option can be
specified multiple times to included models from multiple
repositories.

Local File System
^^^^^^^^^^^^^^^^^

For a locally accessible file-system the absolute path must be
specified::

  $ tritonserver --model-repository=/path/to/model/repository ...

Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^

For a model repository residing in Google Cloud Storage, the
repository path must be prefixed with gs://::

  $ tritonserver --model-repository=gs://bucket/path/to/model/repository ...

S3
^^

For a model repository residing in Amazon S3, the path must be
prefixed with s3://::

  $ tritonserver --model-repository=s3://bucket/path/to/model/repository ...

For a local or private instance of S3, the prefix s3:// must be
followed by the host and port (separated by a semicolon) and
subsequently the bucket path::

  $ tritonserver --model-repository=s3://host:port/bucket/path/to/model/repository ...

When using S3, the credentials and default region can be passed by
using either the `aws config
<https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>_`
command or via the respective `environment variables
<https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>`_.
If the environment variables are set they will take a higher priority
and be used by Triton instead of the credentials set using the aws
config command.

Repository Layout
-----------------

The directories and files that compose a model repository must follow
a required layout. Assuming a repository path is specified as::

  $ tritonserver --model-repository=<model-repository-path>

The corresponding repository layout must be::

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

Within the top-level model repository directory there must be zero or
more <model-name> sub-directories. Each of the <model-name>
sub-directories contains the repository information for the
corresponding model. The config.pbtxt file describes the
:ref:`section-model-configuration` for the model. For some models,
config.pbtxt is required while for others it is optional. See
:ref:`section-generated-model-configuration` for more information.

A <model-name> directory can contain zero or more <output-labels-file>
that are used to provide labels for outputs that represent
classifications. The label file must be specified in the
:cpp:var:`label_filename
<nvidia::inferenceserver::ModelOutput::label_filename>` property of
the output it corresponds to in the :ref:`model configuration
<section-model-configuration>`.

Each <model-name> directory must have at least one numeric
sub-directory representing a version of the model.  For more
information about how the model versions are handled by Triton see
:ref:`section-model-versions`.  Within each version sub-directory
there are one or more model definition files that specify the actual
model, except for :ref:`ensemble models
<section-ensemble-models>`. The model definition can be either a
:ref:`framework-specific model file
<section-framework-model-definition>` or a shared library implementing
a :ref:`custom backend <section-custom-backends>`.

.. _section-modifying-the-model-repository:

Modifying the Model Repository
------------------------------

Triton has multiple execution modes that control how the models within
the model repository are managed. These modes are described in
:ref:`section-model-management`.

.. _section-model-versions:

Model Versions
--------------

Each model can have one or more versions available in the model
repository. Each version is stored in its own, numerically named,
subdirectory where the name of the subdirectory corresponds to the
version number of the model. The subdirectories that are not
numerically named, or that have zero prefix will be ignored. Each
model configuration specifies a :ref:`version policy
<section-version-policy>` that controls which of the versions in the
model repository are made available by Triton at any given time.

.. _section-framework-model-definition:

Framework Model Definition
--------------------------

Each model version sub-directory must contain at least one model
definition. By default, the name of this file or directory must be:

* **model.plan** for TensorRT models
* **model.graphdef** for TensorFlow GraphDef models
* **model.savedmodel** for TensorFlow SavedModel models
* **model.onnx** for ONNX Runtime ONNX models
* **model.pt** for PyTorch TorchScript models
* **model.netdef** and **init_model.netdef** for Caffe2 Netdef models

This default name can be overridden using the *default_model_filename*
property in the :ref:`model configuration
<section-model-configuration>`.

Optionally, a model can provide multiple model definition files, each
targeted at a GPU with a different `Compute Capability
<https://developer.nvidia.com/cuda-gpus>`_. Most commonly, this
feature is needed for TensorRT and TensorFlow/TensorRT integrated
models where the model definition is valid for only a single compute
capability. See the *cc_model_filenames* property in the :ref:`model
configuration <section-model-configuration>` for description of how to
specify different model definitions for different compute
capabilities.

.. _section-tensorrt-models:

TensorRT Models
^^^^^^^^^^^^^^^

A TensorRT model definition is called a *Plan*. A TensorRT Plan is a
single file that by default must be named model.plan. A TensorRT Plan
is specific to CUDA Compute Capability and so it is typically
necessary to use the :ref:`model configuration's
<section-model-configuration>` *cc_model_filenames* property as
described above.

A minimal model repository for a single TensorRT model would look
like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.plan

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  <model-repository-path>/
    <model-name>/
      1/
        model.plan

.. _section-tensorflow-models:

TensorFlow Models
^^^^^^^^^^^^^^^^^

TensorFlow saves trained models in one of two ways: *GraphDef* or
*SavedModel*. Triton supports both formats. Once you have a trained
model in TensorFlow, you can save it as a GraphDef directly or convert
it to a GraphDef by using a script like `freeze_graph.py
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py>`_,
or save it as a SavedModel using a `SavedModelBuilder
<https://www.tensorflow.org/serving/serving_basic>`_ or
`tf.saved_model.simple_save
<https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_. If
you use the Estimator API you can also use
`Estimator.export_savedmodel
<https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_savedmodel>`_.

A TensorFlow GraphDef is a single file that by default must be named
model.graphdef. A minimal model repository for a single TensorFlow
GraphDef model would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef

A TensorFlow SavedModel is a directory containing multiple files. By
default the directory must be named model.savedmodel. A minimal model
repository for a single TensorFlow SavedModel model would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  <model-repository-path>/
    <model-name>/
      1/
        model.savedmodel/
           <saved-model files>

.. _section-tensorrt-tensorflow-models:

TensorRT/TensorFlow Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow 1.7 and later integrates TensorRT to enable TensorFlow
models to benefit from the inference optimizations provided by
TensorRT. Triton supports models that have been optimized with
TensorRT and can serve those models just like any other TensorFlow
model. Triton’s TensorRT version (available in the
Release Notes) must match the TensorRT version that was used when the
model was created.

A TensorRT/TensorFlow integrated model is specific to CUDA Compute
Capability and so it is typically necessary to use the :ref:`model
configuration's <section-model-configuration>` *cc_model_filenames*
property as described above.

As an alternative to creating a TensorRT/TensorFlow model *offline* it
is possible to use model configuration settings to have the TensorRT
optimization performed dynamically, when the model is first loaded or
in response to inference requests. See
:ref:`section-optimization-policy-tensorrt` for more information.

.. _section-onnx-models:

ONNX Models
^^^^^^^^^^^

An ONNX model is a single file or a directory containing multiple
files. By default the file or directory must be named model.onnx.
Notice that some ONNX models may not be supported by Triton as they
are not supported by the underlying ONNX Runtime (due to either using
`stale ONNX opset version
<https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix>`_
or containing operators with `unsupported types
<https://github.com/microsoft/onnxruntime/issues/1122>`_).

By default the ONNX Runtime uses a default *execution provider* when
running models. For execution of models on CPU this default execution
provider does not utilize MKL-DNN. The model configuration
:ref:`section-optimization-policy` allows you to select the `OpenVino
<https://01.org/openvinotoolkit>`_ execution provider for CPU
execution of a model instead of the default execution provider. For
execution of models on GPU the default CUDA execution provider uses
CuDNN to accelerate inference. The model configuration
:ref:`section-optimization-policy` allows you to select the *tensorrt*
execution provider for GPU which causes the ONNX Runtime to use
TensorRT to accelerate all or part of the model. See
:ref:`section-optimization-policy-tensorrt` for more information on
the *tensorrt* execution provider.

A minimal model repository for a single ONNX model contained in a
single file would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  <model-repository-path>/
    <model-name>/
      1/
        model.onnx

An ONNX model composed from multiple files must be contained in a
directory.  By default this directory must be named model.onnx but can
be overridden using the *default_model_filename* property in the
:ref:`model configuration <section-model-configuration>`. The main
model file within this directory must be named model.onnx. A minimal
model repository for a single ONNX model contained in a directory
would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx/
           model.onnx
           <other model files>

.. _section-pytorch-models:

PyTorch Models
^^^^^^^^^^^^^^

An PyTorch model is a single file that by default must be named
model.pt.  It is possible that some models traced with different
versions of PyTorch may not be supported by Triton due to changes in
the underlying opset.  A minimal model repository for a single PyTorch
model would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pt

Caffe2 Models
^^^^^^^^^^^^^

A Caffe2 model definition is called a *NetDef*. A Caffe2 NetDef is a
single file that by default must be named model.netdef. A minimal
model repository for a single NetDef model would look like::

  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.netdef

.. _section-custom-backends:

Custom Backends
---------------

A model using a custom backend is represented in the model repository
in the same way as models using a deep-learning framework backend.
Each model version sub-directory must contain at least one shared
library that implements the custom model backend. By default, the name
of this shared library must be **libcustom.so** but the default name
can be overridden using the *default_model_filename* property in the
:ref:`model configuration <section-model-configuration>`.

Optionally, a model can provide multiple shared libraries, each
targeted at a GPU with a different `Compute Capability
<https://developer.nvidia.com/cuda-gpus>`_. See the
*cc_model_filenames* property in the :ref:`model configuration
<section-model-configuration>` for description of how to specify
different shared libraries for different compute capabilities.

Currently, only model repositories on the local filesystem support
custom backends. A custom backend contained in a model repository in
cloud storage (for example, a repository accessed with the gs://
prefix or s3:// prefix as described above) cannot be loaded.

Custom Backend API
^^^^^^^^^^^^^^^^^^

A custom backend must implement the C interface defined in `custom.h
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/backends/custom/custom.h>`_. The
interface is also documented in the API Reference.

Example Custom Backend
^^^^^^^^^^^^^^^^^^^^^^

Several example custom backends can be found in the `src/custom
directory
<https://github.com/NVIDIA/triton-inference-server/tree/master/src/custom>`_. For
more information on building your own custom backends as well as a
simple example you can build yourself, see
:ref:`section-building-a-custom-backend`.

.. _section-ensemble-backends:

Ensemble Backends
---------------

A model using an ensemble backend is represented in the model repository
in the same way as models using a deep-learning framework backend.
Currently, the ensemble backend does not require any version specific data,
so each model version subdirectory must exist but should be empty.

An example of an ensemble backend in a model repository can be found
in the
`docs/examples/ensemble_model_repository/preprocess_resnet50_ensemble
<https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples/ensemble_model_repository/preprocess_resnet50_ensemble>`_
directory.
