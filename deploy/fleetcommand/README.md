<!--
# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Fleet Command Deploy: NVIDIA Triton Inference Server

A helm chart for installing a single cluster of NVIDIA Triton Inference Server
on Fleet Command is provided. By default the cluster contains a single instance
of the Triton but the *replicaCount* configuration parameter can be set to
create a cluster of any size, as described below.

This guide assumes you already have a functional Fleet Command location
deployed.  Please refer to the [Fleet Command
Documentation](https://docs.nvidia.com/fleet-command/prod_fleet-command/prod_fleet-command/overview.html)

The steps below describe how to set-up a model repository, use helm to launch
the Triton, and then send inference requests to the running Triton Inference
Server. You can optionally scrape metrics with Prometheus and access a Grafana
endpoint to see real-time metrics reported by Triton.

## Model Repository

If you already have a model repository you may use that with this helm chart. If
you do not have a model repository, you can checkout a local copy of the Triton
Inference Server source repository to create an example model repository::

```
$ git clone https://github.com/triton-inference-server/server.git
```

Triton needs a repository of models that it will make available for inferencing.
For this example you will place the model repository in an S3 Storage bucket
(either in AWS or other S3 API compatible on-premises object storage).

```
$ aws s3 mb s3://triton-inference-server-repository
```

Following the [QuickStart](../../docs/quickstart.md) download the example model
repository to your system and copy it into the AWS S3 bucket.

```
$ aws s3 cp -r docs/examples/model_repository s3://triton-inference-server-repository/model_repository
```

### AWS Model Repository

To load the model from the AWS S3, you need to convert the following AWS
credentials in the base64 format and add it to the Application Configuration
section when creating the Fleet Command Deployment.

```
echo -n 'REGION' | base64
echo -n 'SECRECT_KEY_ID' | base64
echo -n 'SECRET_ACCESS_KEY' | base64
# Optional for using session token
echo -n 'AWS_SESSION_TOKEN' | base64
```

## Deploy the Triton Inference Server

Deploy the Triton Inference Server to your Location in Fleet Command by creating
a Deployment.  You can specify configuration parameters to override the default
[values.yaml](values.yaml) in the Application Configuration section.  

*Note:* You _must_ provide a `--model-repository` parameter with a path to your
prepared model repository in your S3 bucket.  Otherwise, the Triton will not
start.

An example Application Configuration for Triton on Fleet Command:
```yaml
image:
  serverArgs:
    - --model-repository=s3://triton-inference-server-repository

secret:
  region: <region in base 64 >
  id: <access id in base 64 >
  key: <access key in base 64>
  token: <session token in base 64 (optional)>
```

See [Fleet Command documentation](https://docs.nvidia.com/fleet-command/prod_fleet-command/prod_fleet-command/ug-deploying-to-the-edge.html)
for more info.

### Prometheus ServiceMonitor Support

If you have `prometheus-operator` deployed, you can enable the ServiceMonitor
for the Triton Inference Server by setting `serviceMonitor.enabled: true` in
Application Configuration.  This will also deploy a Grafana dashboard for Triton
as a ConfigMap.  

Otherwise, metrics can be scraped by pointing an external Prometheus
instance at the `metricsNodePort` in the values.

## Using Triton Inference Server

Now that the Triton Inference Server is running you can send HTTP or GRPC
requests to it to perform inferencing. By default, the service is exposed with a
NodePort service type, where the same port is opened on all systems in a
Location.

Triton exposes an HTTP endpoint on port 30343, and GRPC endpoint on port 30344
and a Prometheus metrics endpoint on port 30345. These ports can be overridden
in the application configuration when deploying.  You can use curl to get the
meta-data of Triton from the HTTP endpoint.  For example, if a system in your
location has the IP `34.83.9.133`:

```
$ curl 34.83.9.133:30343/v2
```

Follow the [QuickStart](../../docs/quickstart.md) to get the example image
classification client that can be used to perform inferencing using image
classification models being served by the Triton. For example,

```
$ image_client -u 34.83.9.133:30343 -m densenet_onnx -s INCEPTION -c 3 mug.jpg
Request 0, batch size 1
Image '/workspace/images/mug.jpg':
    15.349568 (504) = COFFEE MUG
    13.227468 (968) = CUP
    10.424893 (505) = COFFEEPOT
```
