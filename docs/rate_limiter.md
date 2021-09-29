<!--
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Rate Limiter

Rate limiter manages the rate at which requests are scheduled on
model instances by Triton. The rate limiter operates across all
models loaded in Triton to allow *cross-model prioritization*.

In absence of rate limiting (--rate-limit=off), Triton schedules
execution of request (or set of requests when using batching) as
soon as a model instance is available. This behavior is typically
best suited for performance. However, there can be cases 
where running all the models simultaneously place excessive load 
on the server. For instance, model execution on some 
frameworks dynamically allocate memory. Running all such models
simultaneously may lead to system going out-of-memory.

Rate limiter allows to post-pone the inference execution on some
model instances such that not all of them runs simultaneously. 
The model priorities are used to decide which model instance
to schedule next. 

## Using Rate Limiter

To enable rate limiting users must set `--rate-limit` server
cli option. For more information, consult usage of the option
emitted by `tritonserver --help`.

The rate limiter is controlled by rate limiter configuration
described in [rate limiter config](model_configuration.md#rate-limiter-config).
The rate limiter configuration includes 
[resources](model_configuration.md#resources) and
[priority](model_configuration.md#priority) for the instances
defined by the instance group.

### Resources

Resources are just names with a corresponding count. By default
instance uses no rate-limiter resources. By listing a 
resource/count the instance indicates that it requests that
many resources to be available on the instance device before
it can be allowed to execute. When under execution the specified
many resources are allocated to the instance only to be released
when the execution is over. The available number of resources
is, by default, the max across all instances that list that resource
(and so must be adjusted each time an instance is added/removed).
For example, assume three loaded model instances A, B and C each 
specifying the following resource counts for a single device:

```
A: [R1: 4, R2: 4]
B: [R2: 5, R3: 10, R4: 5]
C: [R1: 1, R3: 7, R4: 2]
```

By default, the server will select following available resource
counts in the system:

```
R1: 4
R2: 5
R3: 10
R4: 7
```

These values ensure that all model instances can be successfully
scheduled. The default for a resource can be overridden by giving
it explicitly on command-line using `--rate-limit-resource` option.
`tritonserver --help` will provide with more detailed usage
instructions.

See [resources](model_configuration.md#resources) for more details on
how to specify them in model configuration.

### Priority

In a resource constrained system, there will be a contention for
the resources among model instances to execute their inference
requests. Priority setting helps determining which model instance
to select for next execution. See [priority](model_configuration.md#priority)
for more information.
