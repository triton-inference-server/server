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

# Secure Deployment Considerations

The Triton Inference Server project is designed for flexibility and
allows developers to create and deploy inferencing solutions in a
variety of ways. Developers can deploy Triton as an http server, a
grpc server, a server supporting both, or embed a Triton server into
their own application. Developers can deploy Triton locally or in the
cloud, within a Kubernetes cluster behind an API gateway or as a
standalone process.  This guide is intended to provide some key points
and best practices that users deploying Triton based solutions should
consider.

| [Deploying Behind a Secure Gateway or Proxy](#deploying-behind-a-secure-proxy-or-gateway) | [Running with Least Privilege](#running-with-least-privilege) |

> [!IMPORTANT]
> Ultimately the security of a solution based on Triton
> is the responsibility of the developer building and deploying that
> solution. When deploying in production settings please have security
> experts review any potential risks and threats.

> [!WARNING]
> Dynamic updates to model repositories are disabled by
> default. Enabling dynamic updates to model repositories either
> through model loading APIs or through directory polling can lead to
> arbitrary code execution. Model repository access control is
> critical in production deployments. If dynamic updates are required,
> ensure only trusted entities have access to model loading APIs and
> model repository directories.

## Deploying Behind a Secure Proxy or Gateway

The Triton Inference Server is designed primarily as a microservice to
be deployed as part of a larger solution within an application
framework or service mesh.

In such deployments it is typical to utilize dedicated gateway or
proxy servers to handle authorization, access control, resource
management, encryption, load balancing, redundancy and many other
security and availability features.

The full design of such systems is outside the scope of this
deployment guide but in such scenarios dedicated ingress controllers
handle access from outside the trusted network while Triton Inference
Server handles only trusted, validated requests.

In such scenarios Triton Inference Server is not exposed directly to
an untrusted network.

### References on Secure Deployments

In the following references, Triton Inference Server would be deployed
as an "Application" or "Service" within the trusted internal network.

* [https://www.nginx.com/blog/architecting-zero-trust-security-for-kubernetes-apps-with-nginx/]
* [https://istio.io/latest/docs/concepts/security/]
* [https://konghq.com/blog/enterprise/envoy-service-mesh]
* [https://www.solo.io/topics/envoy-proxy/]

## Running with Least Privilege

  The security principle of least privilege advocates that a process be
  granted the minimum permissions required to do its job.

  For an inference solution based on Triton Inference Server there are a
  number of ways to reduce security risks by limiting the permissions
  and capabilities of the server to the minimum required for correct
  operation.

### 1. Follow Best Practices for Securing Kubernetes Deployments

 When deploying Triton within a Kubernetes pod ensure that it is
 running with a service account with the fewest possible
 permissions. Ensure that you have configured [role based access
 control](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
 to limit access to resources and capabilities as required by your
 application.

### 2. Follow Best Practices for Launching Standalone Docker Containers

  When Triton is deployed as a containerized service, standard docker
  security practices apply. This includes limiting the resources that a
  container has access to as well as limiting network access to the
  container. https://docs.docker.com/engine/security/

### 3. Run as a Non-Root User

   Triton's pre-built containers contain a non-root user that can be used
   to launch the tritonserver application with limited permissions. This
   user, `triton-server` is created with `user id 1000`. When launching
   the container using docker the user can be set with the `--user`
   command line option.

##### Example Launch Command

   ```
   docker run --rm --user triton-server -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:YY.MM-py3 tritonserver --model-repository=/models
   ```

### 4. Restrict or Disable Access to Protocols and APIs

The pre-built Triton Inference Serrver application enables a full set
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

#### Disabling / Restricting Features at Run Time

The `tritonserver` application provides a number of command line
options to enable and disable features when launched. For a full list
of options please see `tritonserver --help`. The following subset are
described here with basic recommendations.

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

Specifies the mode for model management.

> [!WARNING]
> Allowing dynamic updates to the model repository can lead
> to arbitrary code execution. Model repository access control is
> critical in production deployments. Unless required for operation, it's recommended
> to disable dynamic updates. If required, please ensure only trusted entities
> can add or remove models from a model repository.

Options:

 * `none`- Models are loaded at start up and can not be modified.
 * `poll`- Server process will poll the model repository for changes.
 * `explicit` - Models can be loaded and unloaded via the model control APIs.

Recommended to set to `none` unless dynamic updates are required. If
dynamic updates are required care must be taken to control access to
the model repository files and load and unload APIs.

##### `--allow-http <boolean>, default True`

Enable HTTP request handling. Recommended to set to `False` if not required.

##### `--allow-grpc <boolean>, default True`

Enable gRPC request handling. Recommended to set to `False` if not required.

##### `--grpc-use-ssl <boolean> default False`

Use SSL authentication for gRPC requests. Recommended to set to `True` if service is not protected by a gateway or proxy.

##### `--grpc-use-ssl-mutual <boolean> default False`

Use mutual SSL authentication for gRPC requests. Recommended to set to `True` if service is not protected by a gateway or proxy.

##### `--grpc-restricted-protocol <<string>:<string>=<string>>`

Restrict access to specific gRPC protocol categories to users with
specific key, value pair shared secret. See
[limit-endpoint-access](inference_protocols.md#limit-endpoint-access-beta)
for more information.

> [!Note]
> Restricting access can be used to limit exposure to model
> control APIs to trusted users.

##### `--http-restricted-api <<string>:<string>=<string>>`

Restrict access to specific HTTP API categories to users with
specific key, value pair shared secret. See
[limit-endpoint-access](inference_protocols.md#limit-endpoint-access-beta)
for more information.

> [!Note]
> Restricting access can be used to limit exposure to model
> control APIs to trusted users.

##### `--allow-sagemaker <boolean> default False`

Enable Sagemaker request handling. Recommended to set to `False` unless required.

##### `--allow-vertex-ai <boolean> default depends on environment variable`

Enable Vertex AI request handling. Default is `True` if
`AIP_MODE=PREDICTION`, `False` otherwise. Recommended to set to
`False` unless required.

##### `--allow-metrics <boolean> default True`

Allow server to publish prometheus style metrics. Recommended to set
to `False` if not required to avoid capturing or exposing any sensitive information.

#### `--trace-config level=<string> default "off"`

Tracing mode. Trace mode supports `triton` and `opentelemetry`. Unless required `--trace-config level=off` should be set to avoid capturing or exposing any sensitive information.


##### `backend-directory <string> default /opt/tritonserver/backends`

Directory where backend shared libraries are found.

> [!Warning]
> Access to add or remove files from the backend directory
> must be access controlled. Adding untrusted files
> can lead to arbitrarty code execution.

##### `repoagent-directory <string> default /opt/tritonserver/repoagents`
Directory where repository agent shared libraries are found.

> [!Warning]
> Access to add or remove files from the repoagent directory
> must be access controlled. Adding untrusted files
> can lead to arbitrarty code execution.

##### `cache-directory <string> default /opt/tritonserver/caches`

Directory where cache shared libraries are found.

> [!Warning]
> Access to add or remove files from the cache directory
> must be access controlled. Adding untrusted files
> can lead to arbitrarty code execution.





