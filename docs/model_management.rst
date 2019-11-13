..
  # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

.. _section-model-management:

Model Management
================

The inference server operates in one of three model control modes:
NONE, POLL, or EXPLICIT.

Model Control Mode NONE
-----------------------

The server attempts to load all models in the model repository at
startup. Models that the server is not able to load will be marked as
UNAVAILABLE in the server status and will not be available for
inferencing.

Changes to the model repository while the server is running will be
ignored. Model control requests using the :ref:`Model Control API
<section-api-model-control>` will have no affect and will receive an
error response.

This model control mode is selected by specifing
-\\-model-control-mode=none when starting the inference
server.

Model Control Mode POLL
-----------------------

The server attempts to load all models in the model repository at
startup. Models that the server is not able to load will be marked as
UNAVAILABLE in the server status and will not be available for
inferencing.

Changes to the model repository will be detected and the server will
attempt to load and unload models as necessary based on those
changes. Changes to the model repository may not be detected
immediately because the server polls the repository periodically. You
can control the polling interval with the -\\-repository-poll-secs
option. The console log or the :ref:`Status API <section-api-status>`
can be used to determine when model repository changes have taken
effect.

Model control requests using the :ref:`Model Control API
<section-api-model-control>` will have no affect and will receive an
error response.

This model control mode is the default but can be explicitly enabled
by specifing -\\-model-control-mode=poll and by setting
-\\-repository-poll-secs to a non-zero value when starting the
inference server.

In POLL mode the inference server responds to the following model
repository changes:

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

Model Control Mode EXPLICIT
---------------------------

At startup, the server loads only those models specified explicitly
with the -\\-load-model command-line option. If -\\-load-model is not
specified then no models are loaded at startup. After startup, all
model load and unload actions must be initiated explicitly by using
the :ref:`Model Control API <section-api-model-control>`. The response
status of the model control request indicates success or failure of
the load or unload action.

This model control mode is enabled by specifing
-\\-model-control-mode=explicit.

**The EXPLICIT model control mode is experimental**. The inference
server will attempt to load and unload models using the APIs provided
by the framework backends, but it is likely that at least some of the
backends will have difficulty managing repeated load/unload
cycles.
