<!--
# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Schedulers

Triton supports batch inferencing by allowing individual inference
requests to specify a batch of inputs. The inferencing for a batch of
inputs is performed at the same time which is especially important for
GPUs since it can greatly increase inferencing throughput. In many use
cases the individual inference requests are not batched, therefore,
they do not benefit from the throughput benefits of batching.

The inference server contains multiple scheduling and batching
algorithms that support many different model types and use-cases. More
information about model types and schedulers can be found in [Models
And Schedulers](architecture.md#models-and-schedulers).

## Default Scheduler

The default scheduler is used for a model if none of the
*scheduling_choice* properties are specified in the model
configuration. The default scheduler simply distributes inference
requests to all [model instances](model_configuration.md#instance-groups) configured for the
model.

## Ensemble Scheduler

The ensemble scheduler must be used for [ensemble
 models](architecture.md#ensemble-models) and cannot be used for any
 other type of model.

The ensemble scheduler is enabled and configured independently for
each model using the *ModelEnsembleScheduling* property in the model
configuration. The settings describe the models that are included in
the ensemble and the flow of tensor values between the models. See
[Ensemble Models](architecture.md#ensemble-models) for more
information and examples.