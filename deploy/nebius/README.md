<!--
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
# Triton on a Nebius Serverless HTTP Endpoint

This example packages DenseNet-121 with Triton and serves it through a
[Nebius Serverless Endpoint](https://docs.nebius.com/serverless/overview).
It uses one GPU, one HTTP port and managed HTTPS bearer-token authentication.
No Kubernetes cluster or changes to Triton are required.

> **Validation status:** model preparation and the local CPU/HTTP-contract tests
> have been exercised. The GPU container and Nebius deployment have not yet been
> validated. Complete the [validation checklist](#validation-checklist) before
> relying on this example. In particular, confirm the selected image's driver
> compatibility with the GPU platform.

An Endpoint runs until stopped or deleted. This recipe does not configure
autoscaling, scale-to-zero, high availability or preemptible recovery. It uses
the Triton V2 inference protocol, not the Nebius Token Factory/AI Studio model
APIs. Serverless Jobs are finite background workloads without an inference URL;
they are not needed for this example.

## Prerequisites and version selection

- Python 3.10 or later, Docker with Buildx, `jq`, and a Bash-compatible shell.
- A registry you can push to and from which Nebius can pull images.
- [Nebius CLI](https://docs.nebius.com/cli/install) configured for your project;
  the commands below were checked against **0.12.265** help.
- Project permissions and quotas for Serverless Endpoints, GPU resources, disks
  and networking; a subnet and an available single-GPU platform/preset pair.
- A Linux amd64 GPU environment to validate the image before cloud deployment.

[pins.json](pins.json) fixes the Linux amd64 manifest of
`nvcr.io/nvidia/tritonserver:26.08-py3`, the DenseNet ONNX file, Triton's model
configuration/labels, and their source revisions and SHA-256 hashes. This is the
full ONNX-capable server image, not an SDK, minimal or LLM-specific image.

The [26.08 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-08.html)
list ONNX Runtime 1.28.0 and GPUs with compute capability 7.5 or later. At the time
this example was prepared, that page had conflicting CUDA versions and an
incomplete minimum-driver field. Treat this pin as a candidate until GPU
initialization and inference pass on your selected platform. An L40S is a
reasonable single-GPU candidate, but no region's capacity or preset is assumed.
If another release is required, update the pin and Dockerfile together and
repeat the tests; do not use `latest` or assume a TensorRT-LLM image's driver
matrix applies to the full server image.

## Prepare the model and image

From the root of the Triton checkout:

```bash
cd deploy/nebius
python3 prepare_models.py
```

Preparation downloads approximately 32 MB plus configuration and notices,
verifies each pinned size/hash, and adds a single GPU instance and a policy
selecting only model version 1. It publishes `build/` only after success and
refuses to replace an existing directory. It downloads actual model bytes,
not the Git LFS pointer. To repeat preparation, move the existing build aside
or choose `--output` with a new directory. The Dockerfile expects `build/`.

```text
build/
  SHA256SUMS
  model_repository/densenet_onnx/
    config.pbtxt
    densenet_labels.txt
    1/model.onnx
  notices/
```

The generated checksum manifest covers the final configuration as well as
weights and labels. The image checks it during build and startup, and makes
the files non-writable for UID/GID 1000. The source DenseNet README declares
MIT, while the ONNX model repository's root license is Apache-2.0; preparation
preserves both source notices, together with Triton's license. Keep these
notices and review model/container terms before redistributing an image.
This example does not publish an image on your behalf.

Build and push to your registry, then resolve the resulting image digest:

```bash
export IMAGE_REPOSITORY='YOUR_REGISTRY/YOUR_NAMESPACE/triton-densenet'
export IMAGE_TAG="${IMAGE_REPOSITORY}:nebius-example-1"
docker buildx build --platform linux/amd64 --tag "$IMAGE_TAG" --push .
docker buildx imagetools inspect "$IMAGE_TAG"
```

Set `IMAGE` to the returned digest reference, for example
`YOUR_REGISTRY/YOUR_NAMESPACE/triton-densenet@sha256:...`. Use the **derived**
image digest, not the base image digest from `pins.json`: the base does not
contain the prepared model. Keep the image available while an Endpoint may
restart. A private registry also requires the credential secret described below.

## Create an authenticated Endpoint

Choose your own project and resources. The CLI profile supplies control-plane
IAM credentials; these are separate from the Endpoint's inference token.
Create a [SecretStash secret version](https://docs.nebius.com/mysterybox/overview)
with an `AUTH_TOKEN` payload containing a random token (for example generated
with `openssl rand -hex 32`). Retain the token securely for the client. Set
`AUTH_TOKEN_SECRET` to its `SECRET_ID@VERSION_ID` selector. Do not embed tokens
in this checkout, image or command-line arguments.

For a private registry, create a separate secret version with
`REGISTRY_USERNAME` and `REGISTRY_PASSWORD` payload keys and add
`--registry-secret "$REGISTRY_SECRET"` to the create command.

```bash
export NEBIUS_PROFILE='YOUR_CONFIGURED_PROFILE'
export PROJECT_ID='YOUR_PROJECT_ID'
export SUBNET_ID='YOUR_SUBNET_ID'
export GPU_PLATFORM='YOUR_SINGLE_GPU_PLATFORM'
export GPU_PRESET='YOUR_COMPATIBLE_SINGLE_GPU_PRESET'
export AUTH_TOKEN_SECRET='YOUR_SECRET_ID@YOUR_VERSION_ID'
export IMAGE='YOUR_REGISTRY/YOUR_NAMESPACE/triton-densenet@sha256:YOUR_DIGEST'

umask 077
RUN_DIR="receipts/$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
mkdir -p "$RUN_DIR"
export ENDPOINT_NAME="triton-$(basename "$RUN_DIR")"
printf '%s\n' "$PROJECT_ID" > "$RUN_DIR/project-id"
printf '%s\n' "$IMAGE" > "$RUN_DIR/image"
cp pins.json "$RUN_DIR/pins.json"

nebius --profile "$NEBIUS_PROFILE" ai endpoint create \
  --parent-id "$PROJECT_ID" --subnet-id "$SUBNET_ID" \
  --name "$ENDPOINT_NAME" --image "$IMAGE" \
  --platform "$GPU_PLATFORM" --preset "$GPU_PRESET" \
  --container-port 8000/http \
  --auth token --token-secret "$AUTH_TOKEN_SECRET" \
  --disk-size 250Gi --shm-size 1Gi \
  --async --retries 1 --format json > "$RUN_DIR/create-operation.json"
```

Stop here if creation returns an error or the response is missing. A lost
response does not prove that no Endpoint exists. Reconcile using your project,
name, image and creation time in the console or `ai endpoint list`; names are
not necessarily unique. Do not automatically run create again. Keep the receipt
directory private: raw service responses may contain sensitive information.

On a successful asynchronous response, save the operation and resource IDs:

```bash
OPERATION_ID=$(jq -er '.id | select(type == "string" and length > 0)' \
  "$RUN_DIR/create-operation.json")
ENDPOINT_ID=$(jq -er '.resource_id | select(type == "string" and length > 0)' \
  "$RUN_DIR/create-operation.json")
printf '%s\n' "${ENDPOINT_ID:?Missing Endpoint ID}" > "$RUN_DIR/endpoint-id"
nebius --profile "$NEBIUS_PROFILE" ai endpoint operation wait \
  "${OPERATION_ID:?Missing operation ID}" --timeout 35m --format json \
  > "$RUN_DIR/create-result.json"
```

Confirm successful operation completion. If waiting times out, inspect the
**same operation** with `ai endpoint operation get "$OPERATION_ID"` and the
Endpoint's state/details; do not create another resource. Choose your own
bounded waiting/spending limit. The service documents a 30-minute capacity
provisioning timeout; image pull and model readiness are separate stages.

No public VM IP is requested: the managed HTTPS URL works without `--public`.
Ensure the subnet can reach the registry. Only port 8000 is registered; the
entrypoint disables Triton's gRPC and native metrics listeners. Although Nebius
CLI help describes gRPC over managed HTTPS, this recipe does not validate Triton
gRPC clients or expose raw ports 8001/8002.

## Wait for the model and send inference

Get the Endpoint, confirm its state is `RUNNING`, and extract the single HTTPS
URL returned for this example:

```bash
nebius --profile "$NEBIUS_PROFILE" ai endpoint get "${ENDPOINT_ID:?}" \
  --format json > "$RUN_DIR/endpoint.json"
jq -e '.status.state == "RUNNING"' "$RUN_DIR/endpoint.json"
export ENDPOINT_URL=$(jq -er \
  '[.status.public_endpoints[] | select(startswith("https://"))] |
   if length == 1 then .[0] else error("Expected one HTTPS URL") end' \
  "$RUN_DIR/endpoint.json")
```

If not yet running, inspect `status.state_details` and logs before checking again:

```bash
nebius --profile "$NEBIUS_PROFILE" ai endpoint logs "$ENDPOINT_ID"
```

Supply the same `AUTH_TOKEN` to the client through the environment, preferably
from your secret manager. For a local interactive test in Bash:

```bash
read -r -s -p 'Endpoint token: ' ENDPOINT_AUTH_TOKEN
printf '\n'
export ENDPOINT_AUTH_TOKEN
python3 smoke_test.py --wait-seconds 600 --request-timeout 15
unset ENDPOINT_AUTH_TOKEN
```

The smoke test waits for **both** `/v2/health/ready` and
`/v2/models/densenet_onnx/versions/1/ready`, then verifies that missing and wrong
tokens return 401/403 at health and inference routes. It sends a zero-filled
FP32 tensor to `/v2/models/densenet_onnx/versions/1/infer` with
`Authorization: Bearer ...` and checks model/version, output metadata and 1,000
finite values. Credentials and response bodies are not printed; redirects are
not followed with the token.

The JSON input shape is `[3,224,224]`: the model configuration reshapes it to
ONNX's `[1,3,224,224]`. Likewise the output is exposed as `[1000]`. This checks
execution and the HTTP contract, not image-classification accuracy. These
outputs are logits, not probabilities. The script limits response size to 1 MiB
and does not retry inference. It is an example client, not a load generator.

`RUNNING` or `/v2/health/live` alone does not prove model readiness. No custom
platform readiness-probe field is configured; do not assume Docker health
checks gate Nebius ingress. Strict Triton readiness and client polling provide
the checks here, while early callers may still receive startup errors.

## Persistence, stop/start and cleanup

The image is the durable model source. Local caches/container disk are
disposable, and inference results are returned to the client. No bucket or
shared filesystem is required. For larger models, see
[Triton model repositories](../../docs/user_guide/model_repository.md) and
[Nebius volume configuration](https://docs.nebius.com/serverless/endpoints/manage).
An external repository needs its own credentials, immutable revisions,
checksums and live validation; this recipe does not poll a mutable bucket.

Stop/start preserves the Endpoint definition, not warm GPU memory or a specific
VM. Stop synchronously and wait for the operation to finish before starting:

```bash
nebius --profile "$NEBIUS_PROFILE" ai endpoint stop "${ENDPOINT_ID:?}" --timeout 10m
nebius --profile "$NEBIUS_PROFILE" ai endpoint get "$ENDPOINT_ID" \
  --format jsonpath='{.status.state}'
```

Confirm `STOPPED`. If stop times out, inspect `ai endpoint operation list
--resource-id "$ENDPOINT_ID" --all` and wait for the relevant operation before
starting again; state alone is not confirmation of operation completion.

```bash
nebius --profile "$NEBIUS_PROFILE" ai endpoint start "$ENDPOINT_ID" \
  --async --retries 1 --format json > "$RUN_DIR/start-operation.json"
OPERATION_ID=$(jq -er '.id' "$RUN_DIR/start-operation.json")
nebius --profile "$NEBIUS_PROFILE" ai endpoint operation wait \
  "${OPERATION_ID:?}" --timeout 35m
```

Retrieve the current URL and repeat readiness/auth/inference checks after a
successful start. Replacements can require another image pull and model load.
The entrypoint uses `exec` and Triton's 30-second exit allowance, but platform
termination grace and in-flight request draining still require validation.

When finished, delete by the recorded ID and confirm that a subsequent get
returns **NotFound**, not a connection/authentication error:

```bash
nebius --profile "$NEBIUS_PROFILE" ai endpoint delete "${ENDPOINT_ID:?}" \
  --async --retries 1 --format json > "$RUN_DIR/delete-operation.json"
OPERATION_ID=$(jq -er '.id' "$RUN_DIR/delete-operation.json")
nebius --profile "$NEBIUS_PROFILE" ai endpoint operation wait \
  "${OPERATION_ID:?}" --timeout 10m
nebius --profile "$NEBIUS_PROFILE" ai endpoint get "$ENDPOINT_ID"
```

If deletion is incomplete, retain the receipt and reconcile it through
Serverless. Do not directly modify the managed VM. Closing a terminal or
interrupting a request does not stop the Endpoint. Separately mounted storage
has its own lifecycle/billing; remove only dedicated test objects, images and
secrets after confirming no remaining Endpoint uses them. There is no cleanup
watchdog in this example.

## Validation checklist

Run the offline tests without cloud access or a GPU:

```bash
python3 -m unittest discover -s tests -v
```

On a Linux GPU host, build locally with `--load` instead of `--push`, run the
container with `--gpus 1 --shm-size 1g -p 127.0.0.1:8000:8000`, and verify GPU
initialization and inference. A bare local Triton container has no Nebius
gateway; the full smoke test must reject its unauthenticated surface. To check
only the local model during this stage:

```python
from smoke_test import Client

client = Client("http://127.0.0.1:8000", "local-test", allow_local_http=True)
client.wait_ready(60, 15)
values = client.infer(15)
assert len(values) == 1000
```

For numerical validation, generate a 1,000-value JSON reference array by running
`build/model_repository/densenet_onnx/1/model.onnx` with ONNX Runtime's CPU
provider and a zero FP32 input of shape `[1,3,224,224]`. Pass it to
`smoke_test.py --reference reference.json` against the authenticated Endpoint.
The comparison uses `rtol=1e-3, atol=1e-4`; investigate differences rather than
loosening tolerances merely to pass.

Before marking this example cloud-validated, record:

- Exact derived digest, model/config hashes, CLI, GPU, driver and backend versions.
- Two cold starts, correct-token success, missing/wrong-token rejection, numerical
  reference agreement, and malformed input/unknown version rejection.
- Failed startup with a missing/corrupt model, finite concurrency 1/2/4,
  stop/start, an in-flight request during stop, and complete deletion reconciliation.
- No public VM IP; working registry pull and secret delivery; observed ingress
  size/time limits, not assumptions about raw Triton ports.

The local HTTP fixtures exercise the example's client, not Nebius's gateway.
Neither those tests nor CPU ONNX inference establish GPU compatibility.
Use [Serverless lifecycle/status details](https://docs.nebius.com/serverless/lifecycle)
and logs to distinguish capacity, image-pull, model-load and authentication failures.
GPU OOM requires an explicit concurrency/preset decision. Keep model-control
mode `none`; authenticated callers are not given model-load/unload access.

For contribution checks, run the repository's pre-commit hooks and documentation
checks and follow [CONTRIBUTING.md](../../CONTRIBUTING.md), including the CLA and
required upstream CI/test evidence. This example makes no gRPC, binary-tensor,
streaming-model, multi-GPU, autoscaling or production SLA claim.
