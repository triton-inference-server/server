..
  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

The TensorRT Inference Server accesses models from a locally
accessible file path or from Google Cloud Storage. This path is
specified when the server is started using the -\\-model-store option.

For a locally accessible file-system the absolute path must be
specified, for example, -\\-model-store=/path/to/model/repository. For
a model repository residing in Google Cloud Storage, the path must be
prefixed with gs://, for example,
-\\-model-store=gs://bucket/path/to/model/repository.

:ref:`section-example-model-repository` describes how to create an
example repository with a couple of image classification models.

An example of a typical model repository layout is shown below::

  <model-repository-path>/
    model_0/
      config.pbtxt
      output0_labels.txt
      1/
        model.plan
      2/
        model.plan
    model_1/
      config.pbtxt
      output0_labels.txt
      output1_labels.txt
      0/
        model.graphdef
      7/
        model.graphdef

Any number of models may be specified and the inference server will
attempt to load all models into the CPU and GPU when the server
starts. The :ref:`Status API <section-api-status>` can be used to
determine if any models failed to load successfully. The server's
console log will also show the reason for any failures during startup.

The name of the model directory (model_0 and model_1 in the above
example) must match the name of the model specified in the
:ref:`model configuration file <section-model-configuration>`,
config.pbtxt. The model name is used in the :ref:`client API
<section-client-api>` and :ref:`server API
<section-inference-server-api>` to identify the model. Each model
directory must have at least one numeric subdirectory. Each of these
subdirectories holds a version of the model with the version number
corresponding to the directory name.

For more information about how the model versions are handled by the
server see :ref:`section-model-versions`.  Within each version
subdirectory there are one or more model definition files that specify
the actual model, except for
:ref:`ensemble models <section-ensemble-models>`. The model
definition can be either a :ref:`framework-specific model file
<section-framework-model-definition>` or a shared library implementing
a :ref:`custom backend <section-custom-backends>`.

The \*_labels.txt files are optional and are used to provide labels for
outputs that represent classifications. The label file must be
specified in the :cpp:var:`label_filename
<nvidia::inferenceserver::ModelOutput::label_filename>` property of
the output it corresponds to in the :ref:`model configuration
<section-model-configuration>`.

.. _section-modifying-the-model-repository:

Modifying the Model Repository
------------------------------

By default, changes to the model repository will be detected and the
server will attempt to add, remove, and reload models as necessary
based on those changes. Changes to the model repository may not be
detected immediately because the server polls the repository
periodically. You can control the polling interval with the
-\\-repository-poll-secs options. The console log or the :ref:`Status
API <section-api-status>` can be used to determine when model
repository changes have taken effect. You can disable the server from
responding to repository changes by using the
-\\-allow-poll-model-repository=false option.

The TensorRT Inference Server responds to the following changes:

* Versions may be added and removed from models by adding and removing
  the corresponding version subdirectory. The inference server will
  allow in-flight requests to complete even if they are using a
  removed version of the model. New requests for a removed model
  version will fail. Depending on the model's :ref:`version policy
  <section-version-policy>`, changes to the available versions may
  change which model version is served by default.

* Existing models can be removed from the repository by removing the
  corresponding model directory.  The inference server will allow
  in-flight requests to any version of the removed model to
  complete. New requests for a removed model will fail.

* New models can be added to the repository by adding a new model
  directory.

* The :ref:`model configuration <section-model-configuration>`
  (config.pbtxt) can be changed and the server will unload and reload
  the model to pick up the new model configuration.

* Labels files providing labels for outputs that represent
  classifications can be added, removed, or modified and the inference
  server will unload and reload the model to pick up the new
  labels. If a label file is added or removed the corresponding edit
  to the :cpp:var:`label_filename
  <nvidia::inferenceserver::ModelOutput::label_filename>` property of
  the output it corresponds to in the :ref:`model configuration
  <section-model-configuration>` must be performed at the same time.

.. _section-model-versions:

Model Versions
--------------

Each model can have one or more versions available in the model
repository. Each version is stored in its own, numerically named,
subdirectory where the name of the subdirectory corresponds to the
version number of the model. Each model specifies a :ref:`version
policy <section-version-policy>` that controls which of the versions
in the model repository are made available by the server at any given
time.

.. _section-framework-model-definition:

Framework Model Definition
--------------------------

Each model version subdirectory must contain at least one model
definition. By default, the name of this file or directory must be:

* **model.plan** for TensorRT models
* **model.graphdef** for TensorFlow GraphDef models
* **model.savedmodel** for TensorFlow SavedModel models
* **model.netdef** and **init_model.netdef** for Caffe2 Netdef models
* **model.onnx** for ONNX Runtime ONNX models

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

  models/
    <model-name>/
      config.pbtxt
      1/
        model.plan

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  models/
    <model-name>/
      1/
        model.plan

.. _section-tensorflow-models:

TensorFlow Models
^^^^^^^^^^^^^^^^^

TensorFlow saves trained models in one of two ways: *GraphDef* or
*SavedModel*. The inference server supports both formats. Once you
have a trained model in TensorFlow, you can save it as a GraphDef
directly or convert it to a GraphDef by using a script like
`freeze_graph.py
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

  models/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef

A TensorFlow SavedModel is a directory containing multiple files. By
default the directory must be named model.savedmodel. A minimal model
repository for a single TensorFlow SavedModel model would look like::

  models/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  models/
    <model-name>/
      1/
        model.savedmodel/
           <saved-model files>

Caffe2 Models
^^^^^^^^^^^^^

A Caffe2 model definition is called a *NetDef*. A Caffe2 NetDef is a
single file that by default must be named model.netdef. A minimal
model repository for a single NetDef model would look like::

  models/
    <model-name>/
      config.pbtxt
      1/
        model.netdef

TensorRT/TensorFlow Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow 1.7 and later integrates TensorRT to enable TensorFlow
models to benefit from the inference optimizations provided by
TensorRT. The inference server supports models that have been
optimized with TensorRT and can serve those models just like any other
TensorFlow model. The inference server’s TensorRT version (available
in the Release Notes) must match the TensorRT version that was used
when the model was created.

A TensorRT/TensorFlow integrated model is specific to CUDA Compute
Capability and so it is typically necessary to use the :ref:`model
configuration's <section-model-configuration>` *cc_model_filenames*
property as described above.

.. _section-onnx-models:

ONNX Models
^^^^^^^^^^^

An ONNX model is a single file that by default must be named model.onnx.
Notice that some ONNX models may not be supported by the inference server
as they are not supported by the underlying ONNX Runtime (due to either
using `stale ONNX opset version
<https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix>`_
or containing operators with `unsupported types
<https://github.com/microsoft/onnxruntime/issues/1122>`_).
A minimal model repository for a single ONNX model would look like::

  models/
    <model-name>/
      config.pbtxt
      1/
        model.onnx

As described in :ref:`section-generated-model-configuration` the
config.pbtxt is optional for some models. In cases where it is not
required the minimal model repository would look like::

  models/
    <model-name>/
      1/
        model.onnx

.. _section-custom-backends:

Custom Backends
---------------

A model using a custom backend is represented in the model repository
in the same way as models using a deep-learning framework backend.
Each model version subdirectory must contain at least one shared
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

Custom Backend API
^^^^^^^^^^^^^^^^^^

A custom backend must implement the C interface defined in `custom.h
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/backends/custom/custom.h>`_. The
interface is also documented in the API Reference.

Example Custom Backend
^^^^^^^^^^^^^^^^^^^^^^

An example of a custom backend can be found in the `addsub backend
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/custom/addsub/addsub.cc>`_. You
can see the custom backend being used as part of CI testing in
`L0_infer
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/qa/L0_infer>`_.

.. _section-ensemble-backends:

Ensemble Backends
---------------

A model using an ensemble backend is represented in the model repository
in the same way as models using a deep-learning framework backend.
Currently, the ensemble backend does not require any version specific data,
so each model version subdirectory must exist but should be empty.

An example of an ensemble backend in a model repository can be found in the
`docs/examples/ensemble_model_repository/preprocess_resnet50_ensemble
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/docs/examples/ensemble_model_repository/preprocess_resnet50_ensemble>`_
directory.
