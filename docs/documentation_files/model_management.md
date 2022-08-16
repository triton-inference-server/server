<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Model Management

Triton provides model management APIs are part of the [HTTP/REST and
GRPC protocols, and as part of the C
API](inference_protocols.md). Triton operates in one of three model
control modes: NONE, EXPLICIT or POLL. The model control mode
determines how changes to the model repository are handled by Triton
and which of these protocols and APIs are available.

## Model Control Mode NONE

Triton attempts to load all models in the model repository at
startup. Models that Triton is not able to load will be marked as
UNAVAILABLE and will not be available for inferencing.

Changes to the model repository while the server is running will be
ignored. Model load and unload requests using the [model control
protocol](protocol/extension_model_repository.md) will have no affect
and will return an error response.

This model control mode is selected by specifying
`--model-control-mode=none` when starting Triton. This is the default
model control mode. Changing the model repository while Triton is
running must be done carefully, as explained in [Modifying the Model
Repository](#modifying-the-model-repository).

## Model Control Mode EXPLICIT

At startup, Triton loads only those models specified explicitly with the
`--load-model` command-line option. To load ALL models at startup, specify 
`--load-model=*` as the ONLY `--load-model` argument. Specifying 
`--load-model=*` in conjunction with another `--load-model` argument will
result in error. If `--load-model` is not specified then no models are loaded
at startup. Models that Triton is not able to load will be marked as
UNAVAILABLE and will not be available for inferencing.

After startup, all model load and unload actions must be initiated
explicitly by using the [model control
protocol](protocol/extension_model_repository.md). The response
status of the model control request indicates success or failure of
the load or unload action. When attempting to reload an already loaded
model, if the reload fails for any reason the already loaded model
will be unchanged and will remain loaded. If the reload succeeds, the
newly loaded model will replace the already loaded model without any
loss in availability for the model.

This model control mode is enabled by specifying
`--model-control-mode=explicit`. Changing the model repository while
Triton is running must be done carefully, as explained in [Modifying
the Model Repository](#modifying-the-model-repository).

## Model Control Mode POLL

Triton attempts to load all models in the model repository at
startup. Models that Triton is not able to load will be marked as
UNAVAILABLE and will not be available for inferencing.

Changes to the model repository will be detected and Triton will
attempt to load and unload models as necessary based on those changes.
When attempting to reload an already loaded model, if the reload fails
for any reason the already loaded model will be unchanged and will
remain loaded. If the reload succeeds, the newly loaded model will
replace the already loaded model without any loss of availability for
the model.

Changes to the model repository may not be detected immediately
because Triton polls the repository periodically. You can control the
polling interval with the `--repository-poll-secs` option. The console
log or the [model ready
protocol](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md)
or the index operation of the [model control
protocol](protocol/extension_model_repository.md) can be used to
determine when model repository changes have taken effect.

**WARNING: There is no synchronization between when Triton polls the
model repository and when you make any changes to the repository. As a
result Triton could observe partial and incomplete changes that lead
to unexpected behavior. For this reason POLL mode is not recommended
for use in production environments.**

Model load and unload requests using the [model control
protocol](protocol/extension_model_repository.md) will have no affect
and will return an error response.

This model control mode is enabled by specifying
`--model-control-mode=poll` and by setting `--repository-poll-secs` to a
non-zero value when starting Triton. Changing the model repository
while Triton is running must be done carefully, as explained in
[Modifying the Model Repository](#modifying-the-model-repository).

In POLL mode Triton responds to the following model repository
changes:

* Versions may be added and removed from models by adding and removing
  the corresponding version subdirectory. Triton will allow in-flight
  requests to complete even if they are using a removed version of the
  model. New requests for a removed model version will fail. Depending
  on the model's [version
  policy](model_configuration.md#version-policy), changes to the
  available versions may change which model version is served by
  default.

* Existing models can be removed from the repository by removing the
  corresponding model directory.  Triton will allow in-flight requests
  to any version of the removed model to complete. New requests for a
  removed model will fail.

* New models can be added to the repository by adding a new model
  directory.

* The [model configuration file](model_configuration.md)
  (config.pbtxt) can be changed and Triton will unload and reload the
  model to pick up the new model configuration.

* Label(s) files providing labels for outputs that represent
  classifications can be added, removed, or modified and Triton will
  unload and reload the model to pick up the new labels. If a label
  file is added or removed the corresponding edit to the
  *label_filename* property of the output it corresponds to in the
  [model configuration](model_configuration.md) must be performed at
  the same time.

## Modifying the Model Repository

Each model in a model repository [resides in its own
sub-directory](model_repository.md#repository-layout). The activity
allowed on the contents of a model's sub-directory varies depending on
how Triton is using that model. The state of a model can be determined
by using the [model
metadata](inference_protocols.md#inference-protocols-and-apis) or
[repository index](protocol/extension_model_repository.md#index) APIs.

* If the model is actively loading or unloading, no files or
directories within that sub-directory must be added, removed or
modified.

* If the model has never been loaded or has been completely unloaded,
  then the entire model sub-directory can be removed or any of its
  contents can be added, removed or modified.

* If the model has been completely loaded then any files or
directories within that sub-directory can be added, removed or
modified; except for shared libraries implementing the model's
backend. Triton uses the backend shared libraries while the model is
loading so removing or modifying them will likely cause Triton to
crash. To update a model's backend you must first unload the model
completely, modify the backend shared libraries, and then reload the
model. On some OSes it may also be possible to simply move the
existing shared-libraries to another location outside of the model
repository, copy in the new shared libraries, and then reload the
model.

## Concurrently Loading Models

To reduce service downtime, Triton loads new models in the background while
continuing to serve inferences on existing models. Based on use case and
performance requirements, the optimal amount of resources dedicated to loading
models may differ. Triton exposes a `--model-load-thread-count` option to
configure the number of threads dedicated to loading models, which defaults to
twice the number of CPU cores (`2*num_cpus`) visible to the server. 

To set this parameter with the C API, refer to 
`TRITONSERVER_ServerOptionsSetModelLoadThreadCount` in 
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

