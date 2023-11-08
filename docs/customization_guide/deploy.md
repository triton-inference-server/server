<!--
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security Considerations in Deploying a Triton Inference Server

The Triton Inference Server project is designed for flexibility and
allows developers to create and deploy inferencing solutions in a
variety of ways. Developers can deploy Triton as an http server, a
grpc server, a server supporting both, or embed a Triton server into
their own application. Developers can deploy Triton locally or in the
cloud, within a kubernetes cluster, behind an API gateway or as a
standalone process.  This guide is intended to provide some key points
and best practices that users deploying Triton based solutions should
consider.

> [!IMPORTANT]
> Ultimately the security of a solution based on Triton
> is the responsibility of the developer building and deploying that
> solution. When deploying in production settings please have security
> experts review any potential risks and threats.

## Running with Least Privilege

The security principle of least privilege advocates that a process be
granted the minimum permissions required to do its job.

For an inference solution based on Triton Inference Server there are a
number of ways to reduce security risks by limiting the permissions
and capabilities of the server to the minimum required for correct
operation.

### 1. Follow Best Practices for Launching Docker Containers

When Triton is deployed as a containerized service, standard docker
security practices apply. This includes limiting the resources that a
container has access to as well as limiting network access to the
container.

https://docs.docker.com/engine/security/

### 2. Run as a Non-Root User

Triton's pre-built containers contain a non-root user that can be used
to launch the tritonserver application with limited permissions. This
user, `triton-server` is created with `user id 1000`. When launching
the container using docker the user can be set with the `--user`
command line option.

##### Example Launch Command

```
docker run --rm --user triton-server -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:YY.MM-py3 tritonserver --model-repository=/models
```

### 3. Restrict and Disable Access to Protocols and APIs

The pre-built Triton inference server application enables a full set
of features including health checks, server metadata, inference apis,
shared memory apis, model and model repository configuration,
statistics, tracing and logging. Care should be taken to only expose
those capabilities that are required for your solution.

#### Disabling Features at Compile Time

When building a custom inference server application features can be
selectively enabled or disabled using the `build.py` script. As an
example a developer can use the flags `--endpoint http` and
`--endpoint grpc` to compile support for `http`, `grpc` or
both. Support for individual backends can be enabled as well. For more
details please see [documentation](build.md) on building a custom
inference server application.

#### Disabling Features at Application Launch

The `tritonserver` application provides a number of command line
options to enable and disable features when launched. For a full list
of options please see `tritonserver --help`. The following options are highlighted.

##### `--exit-on-error <boolean>, default True`

Exits the inference server if any error occurs during
initialization. Recommended to set to `True` to catch any
unanticipated errors.

##### `--disable-auto-complete-config, default enabled`

Disables backends from autocompleting model configuration. If not
required for your solution recommended to disable to ensure model
configurations are defined statically.

##### `--strict-readiness <boolean>, default True`

If set to true `/v2/health/ready` will only report ready when all
selected models are loaded. Recommended to set to `True` to provide a
signal to other services and orchestration frameworks when full
initialization is complete and server is healthy.

##### `--model-control-mode <string>, default "none"`

> [!WARNING]
> Allowing dynamic updates to the model repository
> can lead to arbitrary execution attacks. Model repository
> access control is critical in production deployments.

Specify the mode for model management.

 * `none`


https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html#limit-endpoint-access-beta




https://www.digitalguardian.com/blog/what-principle-least-privilege-polp-best-practice-information-security-and-compliance#:~:text=The%20principle%20of%20least%20privilege%20works%20by%20allowing%20only%20enough,account%2C%20device%2C%20or%20application.



Least Privilege: Allow running code only the permissions needed to
complete the required tasks and no more. The intent of the least
privileged principle is to reduce the "Exploitability" by minimizing
the privilege proliferation.

##

##

##



https://www.nginx.com/blog/architecting-zero-trust-security-for-kubernetes-apps-with-nginx/


https://istio.io/latest/docs/concepts/security/

https://konghq.com/blog/enterprise/envoy-service-mesh

https://www.solo.io/topics/envoy-proxy/

