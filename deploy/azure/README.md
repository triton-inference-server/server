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

# Kubernetes Deploy: Triton Inference Server Cluster

A helm chart for installing a single cluster of Triton Inference
Server is provided. By default the cluster contains a single instance
of the inference server but the *replicaCount* configuration parameter
can be set to create a cluster of any size, as described below.

This guide assumes you already have a functional Kubernetes cluster
and helm installed (see below for instructions on installing
helm). Note the following requirements:

* The helm chart deploys Prometheus and Grafana to collect and display Triton metrics. To use this helm chart you must install Prpmetheus and Grafana in your cluster as described below and your cluster must contain sufficient CPU resourses to support these services.

* If you want Triton Server to use GPUs for inferencing, your cluster
must be configured to contain the desired number of GPU nodes (Azure NC series VMs recommended)
with support for the NVIDIA driver and CUDA version required by the version
of the inference server you are using.

The steps below describe how to set-up a model repository, use helm to
launch the inference server, and then send inference requests to the
running server. You can access a Grafana endpoint to see real-time
metrics reported by the inference server.

## Installing Helm

If you do not already have Helm installed in your Kubernetes cluster,
executing the following steps from the [official helm install
guide](https://helm.sh/docs/intro/install/) will
give you a quick setup.

```
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```

## Model Repository

If you already have a model repository you may use that with this helm
chart. If you do not have a model repository, you can checkout a local
copy of the inference server source repository to create an example
model repository:

```
$ git clone https://github.com/triton-inference-server/server.git
```

Triton Server needs a repository of models that it will make available
for inferencing. For this example you will place the model repository
in an Azure Storage Container.

```
az storage container create --name triton-inference-server-repository
```

Following the [QuickStart](../../docs/quickstart.md) download the
example model repository to your system and copy it into the Azure Storage Container

```
$ cd ./docs/examples
$ for file in `find model_repository -type f`; do
  az storage blob upload --container-name triton-inference-server-repository --account-name
done
$ aws cp -r docs/examples/model_repository s3://triton-inference-server-repository/model_repository
```

### AWS Model Repository
To load the model from the AWS S3, you need to convert the following AWS credentials in the base64 format and add it to the values.yaml

```
echo -n 'REGION' | base64
```
```
echo -n 'SECRECT_KEY_ID' | base64
```
```
echo -n 'SECRET_ACCESS_KEY' | base64
```

## Deploy Prometheus and Grafana

The inference server metrics are collected by Prometheus and viewable
by Grafana. The inference server helm chart assumes that Prometheus
and Grafana are available so this step must be followed even if you
don't want to use Grafana.

Use the prometheus-operator to install these components. The
*serviceMonitorSelectorNilUsesHelmValues* flag is needed so that
Prometheus can find the inference server metrics in the *example*
release deployed below.

```
$ helm install --name example-metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false stable/prometheus-operator
```

Then port-forward to the Grafana service so you can access it from
your local browser.

```
$ kubectl port-forward service/example-metrics-grafana 8080:80
```

Now you should be able to navigate in your browser to localhost:8080
and see the Grafana login page. Use username=admin and
password=prom-operator to login.

An example Grafana dashboard is available in dashboard.json. Use the
import function in Grafana to import and view this dashboard.

## Deploy the Inference Server

Deploy the inference server using the default configuration with the
following commands.

```
$ cd <directory containing Chart.yaml>
$ helm install --name example .
```

Use kubectl to see status and wait until the inference server pods are
running.

```
$ kubectl get pods
NAME                                               READY   STATUS    RESTARTS   AGE
example-triton-inference-server-5f74b55885-n6lt7   1/1     Running   0          2m21s
```

There are several ways of overriding the default configuration as
described in this [helm
documentation](https://helm.sh/docs/using_helm/#customizing-the-chart-before-installing).

You can edit the values.yaml file directly or you can use the *--set*
option to override a single parameter with the CLI. For example, to
deploy a cluster of four inference servers use *--set* to set the
replicaCount parameter.

```
$ helm install --name example --set replicaCount=4 .
```

You can also write your own "config.yaml" file with the values you
want to override and pass it to helm.

```
$ cat << EOF > config.yaml
namespace: MyCustomNamespace
image:
  imageName: nvcr.io/nvidia/tritonserver:custom-tag
  modelRepositoryPath: gs://my_model_repository
EOF
$ helm install --name example -f config.yaml .
```

## Using Triton Inference Server

Now that the inference server is running you can send HTTP or GRPC
requests to it to perform inferencing. By default, the inferencing
service is exposed with a LoadBalancer service type. Use the following
to find the external IP for the inference server. In this case it is
34.83.9.133.

```
$ kubectl get services
NAME                             TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)                                        AGE
...
example-triton-inference-server  LoadBalancer   10.18.13.28    34.83.9.133   8000:30249/TCP,8001:30068/TCP,8002:32723/TCP   47m
```

The inference server exposes an HTTP endpoint on port 8000, and GRPC
endpoint on port 8001 and a Prometheus metrics endpoint on
port 8002. You can use curl to get the meta-data of the inference server
from the HTTP endpoint.

```
$ curl 34.83.9.133:8000/v2
```

Follow the [QuickStart](../../docs/quickstart.md) to get the example
image classification client that can be used to perform inferencing
using image classification models being served by the inference
server. For example,

```
$ image_client -u 34.83.9.133:8000 -m inception_graphdef -s INCEPTION -c3 mug.jpg
Request 0, batch size 1
Image 'images/mug.jpg':
    504 (COFFEE MUG) = 0.723992
    968 (CUP) = 0.270953
    967 (ESPRESSO) = 0.00115997
```

## Cleanup

Once you've finished using the inference server you should use helm to
delete the deployment.

```
$ helm list
NAME            REVISION  UPDATED                   STATUS    CHART                          APP VERSION   NAMESPACE
example         1         Wed Feb 27 22:16:55 2019  DEPLOYED  triton-inference-server-1.0.0  1.0           default
example-metrics	1       	Tue Jan 21 12:24:07 2020	DEPLOYED	prometheus-operator-6.18.0   	 0.32.0     	 default

$ helm delete --purge example
$ helm delete --purge example-metrics
```

For the Prometheus and Grafana services you should [explicitly delete
CRDs](https://github.com/helm/charts/tree/master/stable/prometheus-operator#uninstalling-the-chart):

```
$ kubectl delete crd alertmanagers.monitoring.coreos.com servicemonitors.monitoring.coreos.com podmonitors.monitoring.coreos.com prometheuses.monitoring.coreos.com prometheusrules.monitoring.coreos.com
```

You may also want to delete the AWS bucket you created to hold the
model repository.

```
$ aws s3 rm -r gs://triton-inference-server-repository
```
