<!--
# Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Kubernetes Deploy: NVIDIA Triton Inference Server Cluster

This repository includes a Helm chart and instructions for installing NVIDIA Triton
Inference Server in an on-premises or AWS EC2 Kubernetes cluster. You can also use this
repository to enable load balancing and autoscaling for your Triton cluster.

This guide assumes you already have a functional Kubernetes cluster with support for GPUs.
See the [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/install-k8s.html)
for instructions on how to install Kubernetes and enable GPU access in your Kubernetes cluster.
You must also have Helm installed (see [Installing Helm](#installing-helm) for instructions). Note the following requirements:

* To deploy Prometheus and Grafana to collect and display Triton metrics, your cluster must contain sufficient CPU resources to support these services.

* To use GPUs for inferencing, your cluster must be configured to contain the desired number of GPU nodes, with
support for the NVIDIA driver and CUDA version required by the version
of the inference server you are using.

* To enable autoscaling, your cluster's kube-apiserver must have the [aggregation layer
enabled](https://kubernetes.io/docs/tasks/extend-kubernetes/configure-aggregation-layer/).
This will allow the horizontal pod autoscaler to read custom metrics from the prometheus adapter.

This Helm chart is available from [Triton Inference Server
GitHub.](https://github.com/triton-inference-server/server)

For more information on Helm and Helm charts, visit the [Helm documentation](https://helm.sh/docs/).

## Quickstart

First, clone this repository to a local machine. Then, execute the following commands:

Install helm

```
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```

Deploy Prometheus and Grafana

```
$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
$ helm repo update
$ helm install example-metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false prometheus-community/kube-prometheus-stack
```

Deploy Triton with default settings

```
helm install example ./deploy/k8s-onprem
```


<!-- The steps below describe how to set-up a model repository, use Helm to
launch the inference server, and then send inference requests to the
running server. You can access a Grafana endpoint to see real-time
metrics reported by the inference server. -->


## Installing Helm

### Helm v3

If you do not already have Helm installed in your Kubernetes cluster,
executing the following steps from the [official Helm install
guide](https://helm.sh/docs/intro/install/) will
give you a quick setup.

If you are currently using Helm v2 and would like to migrate to Helm v3,
see the [official migration guide](https://helm.sh/docs/topics/v2_v3_migration/).

## Model Repository
If you already have a model repository, you may use that with this Helm
chart. If you do not have a model repository, you can check out a local
copy of the server source repository to create an example
model repository:

```
$ git clone https://github.com/triton-inference-server/server.git
```

Triton Server needs a repository of models that it will make available
for inferencing. For this example, we are using an existing NFS server and
placing our model files there. See the
[Model Repository documentation](../../docs/user_guide/model_repository.md) for other
supported locations.

Following the [QuickStart](../../docs/getting_started/quickstart.md), download the
example model repository to your system and copy it onto your NFS server.
Then, add the url or IP address of your NFS server and the server path of your
model repository to `values.yaml`.


## Deploy Prometheus and Grafana

The inference server metrics are collected by Prometheus and viewable
through Grafana. The inference server Helm chart assumes that Prometheus
and Grafana are available so this step must be followed even if you
do not want to use Grafana.

Use the [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) Helm chart to install these components. The
*serviceMonitorSelectorNilUsesHelmValues* flag is needed so that
Prometheus can find the inference server metrics in the *example*
release deployed in a later section.

```
$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
$ helm repo update
$ helm install example-metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false prometheus-community/kube-prometheus-stack
```

Then port-forward to the Grafana service so you can access it from
your local browser.

```
$ kubectl port-forward service/example-metrics-grafana 8080:80
```

Now you should be able to navigate in your browser to localhost:8080
and see the Grafana login page. Use username=admin and
password=prom-operator to log in.

An example Grafana dashboard is available in dashboard.json. Use the
import function in Grafana to import and view this dashboard.

## Enable Autoscaling
To enable autoscaling, ensure that autoscaling tag in `values.yaml`is set to `true`.
This will do two things:

1. Deploy a Horizontal Pod Autoscaler that will scale replicas of the triton-inference-server
based on the information included in `values.yaml`.

2. Install the [prometheus-adapter](https://github.com/prometheus-community/helm-charts/tree/main/charts/prometheus-adapter) helm chart, allowing the Horizontal Pod Autoscaler to scale
based on custom metrics from prometheus.

The included configuration will scale Triton pods based on the average queue time,
as described in [this blog post](https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/#:~:text=Query%20NVIDIA%20Triton%20metrics%20using%20Prometheus). To customize this,
you may replace or add to the list of custom rules in `values.yaml`. If you change
the custom metric, be sure to change the values in autoscaling.metrics.

If autoscaling is disabled, the number of Triton server pods is set to the minReplicas
variable in `values.yaml`.

## Enable Load Balancing
To enable load balancing, ensure that the loadBalancing tag in `values.yaml`
is set to `true`. This will do two things:

1. Deploy a Traefik reverse proxy through the [Traefik Helm Chart](https://github.com/traefik/traefik-helm-chart).

2. Configure two Traefik [IngressRoutes](https://doc.traefik.io/traefik/providers/kubernetes-crd/),
one for http and one for grpc. This will allow the Traefik service to expose two
ports that will be forwarded to and balanced across the Triton pods.

To choose the port numbers exposed, or to disable either http or grpc, edit the
configured variables in `values.yaml`.

## Deploy the Inference Server

Deploy the inference server, autoscaler, and load balancer using the default
configuration with the following commands.

Here, and in the following commands we use the name `example` for our chart.
This name will be added to the beginning of all resources created during the helm
installation.

```
$ cd <directory containing Chart.yaml>
$ helm install example .
```

Use kubectl to see status and wait until the inference server pods are
running.

```
$ kubectl get pods
NAME                                               READY   STATUS    RESTARTS   AGE
example-triton-inference-server-5f74b55885-n6lt7   1/1     Running   0          2m21s
```

There are several ways of overriding the default configuration as
described in this [Helm
documentation](https://helm.sh/docs/using_helm/#customizing-the-chart-before-installing).

You can edit the values.yaml file directly or you can use the *--set*
option to override a single parameter with the CLI. For example, to
deploy a cluster with a minimum of two inference servers use *--set* to
set the autoscaler.minReplicas parameter.

```
$ helm install example --set autoscaler.minReplicas=2 .
```

You can also write your own "config.yaml" file with the values you
want to override and pass it to Helm. If you specify a "config.yaml" file, the
values set will override those in values.yaml.

```
$ cat << EOF > config.yaml
namespace: MyCustomNamespace
image:
  imageName: nvcr.io/nvidia/tritonserver:custom-tag
  modelRepositoryPath: gs://my_model_repository
EOF
$ helm install example -f config.yaml .
```

## Probe Configuration

In `templates/deployment.yaml` is configurations for `livenessProbe`, `readinessProbe` and `startupProbe` for the Triton server container.
By default, Triton loads all the models before starting the HTTP server to respond to the probes. The process can take several minutes, depending on the models sizes.
If it is not completed in `startupProbe.failureThreshold * startupProbe.periodSeconds` seconds then Kubernetes considers this as a pod failure and restarts it,
ending up with an infinite loop of restarting pods, so make sure to sufficiently set these values for your use case.
The liveliness and readiness probes are being sent only after the first success of a startup probe.

For more details, see the [Kubernetes probe documentation](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/) and the [feature page of the startup probe](https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/950-liveness-probe-holdoff/README.md).

## Using Triton Inference Server

Now that the inference server is running you can send HTTP or GRPC
requests to it to perform inferencing. By default, this chart deploys [Traefik](https://traefik.io/)
and uses [IngressRoutes](https://doc.traefik.io/traefik/providers/kubernetes-crd/)
to balance requests across all available nodes.

To send requests through the Traefik proxy, use the Cluster IP of the
traefik service deployed by the Helm chart. In this case, it is 10.111.128.124.

```
$ kubectl get services
NAME                              TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                    AGE
...
example-traefik                   LoadBalancer   10.111.128.124   <pending>     8001:31752/TCP,8000:31941/TCP,80:30692/TCP,443:30303/TCP   74m
example-triton-inference-server   ClusterIP      None             <none>        8000/TCP,8001/TCP,8002/TCP                                 74m
```

Use the following command to refer to the Cluster IP:
```
cluster_ip=`kubectl get svc -l app.kubernetes.io/name=traefik -o=jsonpath='{.items[0].spec.clusterIP}'`
```


The Traefik reverse-proxy exposes an HTTP endpoint on port 8000, and GRPC
endpoint on port 8001 and a Prometheus metrics endpoint on
port 8002. You can use curl to get the meta-data of the inference server
from the HTTP endpoint.

```
$ curl $cluster_ip:8000/v2
```

Follow the [QuickStart](../../docs/getting_started/quickstart.md) to get the example
image classification client that can be used to perform inferencing
using image classification models on the inference
server. For example,

```
$ image_client -u $cluster_ip:8000 -m inception_graphdef -s INCEPTION -c3 mug.jpg
Request 0, batch size 1
Image 'images/mug.jpg':
    504 (COFFEE MUG) = 0.723992
    968 (CUP) = 0.270953
    967 (ESPRESSO) = 0.00115997
```

## Testing Load Balancing and Autoscaling
After you have confirmed that your Triton cluster is operational and can perform inference,
you can test the load balancing and autoscaling features by sending a heavy load of requests.
One option for doing this is using the
[perf_analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
application.

You can apply a progressively increasing load with a command like:
```
perf_analyzer -m simple -u $cluster_ip:8000 --concurrency-range 1:10
```

From your Grafana dashboard, you should be able to see the number of pods increase
as the load increases, with requests being routed evenly to the new pods.

## Cleanup

After you have finished using the inference server, you should use Helm to
delete the deployment.

```
$ helm list
NAME            REVISION  UPDATED                   STATUS    CHART                          APP VERSION   NAMESPACE
example         1         Wed Feb 27 22:16:55 2019  DEPLOYED  triton-inference-server-1.0.0  1.0           default
example-metrics	1       	Tue Jan 21 12:24:07 2020	DEPLOYED	prometheus-operator-6.18.0   	 0.32.0     	 default

$ helm uninstall example
$ helm uninstall example-metrics
```

For the Prometheus and Grafana services, you should [explicitly delete
CRDs](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack#uninstall-helm-chart):

```
$ kubectl delete crd alertmanagerconfigs.monitoring.coreos.com alertmanagers.monitoring.coreos.com podmonitors.monitoring.coreos.com probes.monitoring.coreos.com prometheuses.monitoring.coreos.com prometheusrules.monitoring.coreos.com servicemonitors.monitoring.coreos.com thanosrulers.monitoring.coreos.com
```
