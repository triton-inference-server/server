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

# Fleet Command Deploy: Triton Inference Server Cluster

A helm chart for installing a single cluster of Triton Inference Server on Fleet
Command is provided. By default the cluster contains a single instance of the
inference server but the *replicaCount* configuration parameter can be set to
create a cluster of any size, as described below.

This guide assumes you already have a functional Fleet Command location
deployed.  Please refer to the [Fleet Command
Documentation](https://docs.nvidia.com/fleet-command/prod_fleet-command/prod_fleet-command/overview.html)

The steps below describe how to set-up a model repository, use helm to launch
the inference server, and then send inference requests to the running server.
You can access a Grafana endpoint to see real-time metrics reported by the
inference server.

## Model Repository

If you already have a model repository you may use that with this helm
chart. If you do not have a model repository, you can checkout a local
copy of the inference server source repository to create an example
model repository::

```
$ git clone https://github.com/triton-inference-server/server.git
```

Triton Server needs a repository of models that it will make available
for inferencing. For this example you will place the model repository
in an S3 Storage bucket (either in AWS or other S3 API compatible on-premises object storage).

```
$ aws mb s3://triton-inference-server-repository
```

Following the [QuickStart](../../docs/quickstart.md) download the
example model repository to your system and copy it into the AWS S3
bucket.

```
$ aws cp -r docs/examples/model_repository s3://triton-inference-server-repository/model_repository
```

### AWS Model Repository
To load the model from the AWS S3, you need to convert the following AWS
credentials in the base64 format and add it to the Application Configuration
section when creating the Fleet Command Deployment.

```
echo -n 'REGION' | base64
```
```
echo -n 'SECRECT_KEY_ID' | base64
```
```
echo -n 'SECRET_ACCESS_KEY' | base64
```

## Deploy the Inference Server

Deploy the inference server to your Location in Fleet Command by creating a
Deployment.  You can specify configuration parameters to override the default
[values.yaml](values.yaml) in the Application Configuration section.  

*Note:* You _must_ provide a `--model-repository` parameter with a path to your
prepared model repository in your S3 bucket.  Otherwise, the Triton Inference
Server will not start.

See [Fleet Command documentation](https://docs.nvidia.com/fleet-command/prod_fleet-command/prod_fleet-command/ug-deploying-to-the-edge.html)
for more info.

## Using Triton Inference Server

Now that the inference server is running you can send HTTP or GRPC requests to
it to perform inferencing. By default, the inferencing service is exposed with a
NodePort service type, where the same port is opened on all systems in a
Location.

The inference server exposes an HTTP endpoint on port 30343, and GRPC endpoint
on port 30344 and a Prometheus metrics endpoint on port 30345. These ports can
be overridden in the application configuration when deploying.  You can use curl
to get the meta-data of the inference server from the HTTP endpoint.  For
example, if a system in your location has the IP `34.83.9.133`:

```
$ curl 34.83.9.133:30343/v2
```

Follow the [QuickStart](../../docs/quickstart.md) to get the example
image classification client that can be used to perform inferencing
using image classification models being served by the inference
server. For example,

```
$ image_client -u 34.83.9.133:30343 -m inception_graphdef -s INCEPTION -c3 mug.jpg
Request 0, batch size 1
Image 'images/mug.jpg':
    504 (COFFEE MUG) = 0.723992
    968 (CUP) = 0.270953
    967 (ESPRESSO) = 0.00115997
```
