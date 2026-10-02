<!--
# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Report a Security Vulnerability

To report a potential security vulnerability in any NVIDIA product, please use either:
* This web form: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html), or
* Send email to: [NVIDIA PSIRT](mailto:psirt@nvidia.com)

**OEM Partners should contact their NVIDIA Customer Program Manager**

If reporting a potential vulnerability via email, please encrypt it using NVIDIA’s public PGP key ([see PGP Key page](https://www.nvidia.com/en-us/security/pgp-key/)) and include the following information:
1. Product/Driver name and version/branch that contains the vulnerability
2. Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code
5. Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.

## Reporting Channels

**Please do not open a public GitHub issue, discussion or pull request to report a security vulnerability.** Use one of the following private channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred): https://www.nvidia.com/en-us/security/
2. **Email**: [psirt@nvidia.com](mailto:psirt@nvidia.com), encrypted with NVIDIA's [public PGP key](https://www.nvidia.com/en-us/security/pgp-key/)
3. **GitHub Private Vulnerability Reporting**: use the "Report a vulnerability" button on the Security tab of this repository, where it is enabled

NVIDIA PSIRT acknowledges reports, assesses severity and coordinates remediation and disclosure with the reporter.

## Security Architecture and Context

Triton Inference Server (`server`) is a **Service**: a long-running inference server that loads models from a model repository and serves inference requests over network endpoints. Its primary security responsibility is to parse untrusted network input safely, enforce the configured access restrictions, and keep requests for different models and clients isolated within one process.

Key interfaces and trust boundaries:

- **HTTP/REST endpoint** (`src/http_server.cc`): KServe v2 and extension APIs for inference, model metadata, model repository control, shared memory, statistics, tracing and logging. Also optional SageMaker and Vertex AI adapters.
- **gRPC endpoint** (`src/grpc/`): the same APIs over gRPC, including streaming inference. Optional TLS and mutual TLS are configured through the `--grpc-use-ssl*` options.
- **Metrics endpoint**: optional Prometheus metrics, enabled with `--allow-metrics`.
- **Command line and configuration** (`src/command_line_parser.cc`, `src/main.cc`): options set by the deployer control listeners, enabled APIs, model control mode, tracing output and TLS material.
- **Model repository and backends**: model files, model configuration and backend libraries are loaded from local or cloud storage into the server process.
- **Shared memory** (`src/shared_memory_manager.cc`): clients on the same host can register system or CUDA shared memory regions for tensor exchange.

The HTTP and gRPC listeners default to binding all interfaces (`0.0.0.0`). Built-in access control is limited to optional per-API-category restrictions (`--http-restricted-api`, `--grpc-restricted-protocol`), which require a configured request header key and value. The categories are health, metadata, inference, shared-memory, model-config, model-repository, statistics, trace and logging.

## Threat Model

1. **Unauthenticated access to management APIs.** If the endpoints are reachable by untrusted clients and the restrictions above are not configured, any client can call the model repository, shared memory, trace, logging and statistics APIs. In particular, the model repository load and unload APIs, available when `--model-control-mode=explicit`, can change which models and backends run in the server.
2. **Malicious or tampered models and backends.** Model files, configuration and backend libraries are loaded from the model repository. A model from an untrusted source, or a repository that an attacker can write to, can execute arbitrary code in the server process, since backends are loaded as shared libraries and the Python backend runs model code.
3. **Malformed or oversized requests.** The HTTP parser (`src/http_server.cc`) and the gRPC handlers (`src/grpc/infer_handler.cc`, `src/grpc/stream_infer_handler.cc`) process attacker-controlled headers, JSON, binary tensor data and compressed bodies. Malformed input can lead to memory exhaustion, crashes or denial of service. Request size is bounded by `--http-max-input-size` and the gRPC message limits, which must be sized for the deployment.
4. **Shared memory abuse.** A client that can register shared memory regions (`src/shared_memory_manager.cc`) supplies a region key, offset and size that the server maps and reads or writes. Incorrect bounds, access to regions belonging to other processes, or exposure of CUDA IPC handles can leak or corrupt data across trust boundaries.
5. **Information disclosure through observability.** Metrics, statistics, model metadata and verbose logs can reveal model names, configuration, request rates and error details. Trace output (`src/tracer.cc`, `--trace-config`) can include input and output tensors at the `TENSORS` level and is written to a path chosen by the deployer, so the file's location and permissions determine who can read it.
6. **Weak transport security.** Endpoints that are not protected by TLS expose request and response data, including tensors and any restricted-API header values, to anyone on the network path. gRPC TLS and mutual TLS are optional and are off by default.
7. **Build and supply chain.** `build.py`, the Dockerfiles and the dependencies fetched during the build determine what ends up in the released server and container images. Unpinned or unverified dependencies can introduce compromised code.

## Critical Security Assumptions

- Triton is deployed on a **trusted network** or behind a gateway that provides authentication, authorization, rate limiting and TLS termination. The server itself does not provide user identity or per-user authorization.
- Operators who expose Triton beyond a trusted network configure `--http-address`, `--grpc-address` and the restricted-API options so that management APIs are not reachable by untrusted clients.
- The **model repository is trusted and access-controlled.** Anyone who can write models or backends to the repository, or call the model repository API, is able to run code in the server. Models from untrusted sources must be reviewed or run in an isolated deployment.
- Shared memory is used only between **mutually trusted processes on the same host**, and the host operating system and GPU driver enforce memory isolation correctly.
- TLS certificates and keys provided through the `--grpc-use-ssl*` options, and any restricted-API header values, are stored and rotated securely by the operator.
- Trace and log output locations are writable only by the server and readable only by authorized users.
- Backends and third-party libraries (including the framework runtimes each backend embeds) are kept up to date and are not affected by known vulnerabilities in the deployed version.
