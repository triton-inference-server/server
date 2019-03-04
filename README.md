# TensorRT Inference Server Helm Chart

Simple helm chart for installing the NVIDIA Inference Server.

# Quickstart

This guide assumes you already have a functional Kubernetes cluster.
A later section in this guide goes over installing helm.
Setting up the model repository is also detailed later in this guide.

To deploy the inference server:
```shell
$ git clone https://gitlab-master.nvidia.com/rgaubert/nvidia-inference-server.git
$ helm install nvidia-inference-server
```

In the future the user experience might be (the first two steps might even be unecessary):
```shell
$ helm repo add gpu-helm-charts https://nvidia.github.io/tensorrt-inference-server/helm-charts
$ helm repo update
$ helm install nvidia-inference-server
```

# Model Repository

TensorRT Inference Server needs a repository of models that it will make
available for inferencing.
You can find an example repository in the [open-source repo](
https://github.com/NVIDIA/tensorrt-inference-server) and
instructions on how to create your own model repository in the User
Guide referenced from the README.

For this example you will place the model repository in a Google Cloud
Storage bucket.

```shell
$ gsutil mb gs://trtis-kubeflow
```

Following the User Guide download the example model repository to your
system and copy it into the GCS bucket.

```shell
$ gsutil cp -r docs/examples/model_repository gs://trtis-kubeflow
```

Make sure the bucket is setup with public ACLs so the inference server can access the model store.

# Overriding the default configuration

You can always edit this repository's "values.yaml" if you have the full chart source.

Many ways of overriding the charts are possible, these are described here: 
[https://helm.sh/docs/using_helm/#customizing-the-chart-before-installing](https://helm.sh/docs/using_helm/#customizing-the-chart-before-installing)

You can use the `--set` option to override a single parameter with the CLI. e.g:
```shell
helm install nvidia-inference-server --set image.imageName="nvcr.io/nvidia/tensorrtserver:custom-tag"
```

You can use a file by writing your own "values.yaml" file with the values you want to override and pass it to helm:
```shell
$ cat << EOF > config.yaml
namespace: MyCustomNamespace

image:
  imageName: nvcr.io/nvidia/tensorrtserver:custom-tag
  modelRepositoryPath: gs://trtis-kubeflow/model_repository
EOF

$ helm install -f config.yaml nvidia-inference-server
```

## Using the TensorRT Inference Server

Now that TRTIS is running you can send HTTP or GRPC requests to it to
perform inferencing. By default the inferencing service is exposed
with a LoadBalancer service type. Use the following to find the
external IP for TRTIS. In this case it is 35.232.176.113.

```shell
$ kubectl get services
NAME         TYPE           CLUSTER-IP    EXTERNAL-IP      PORT(S)                                        AGE
inference-se LoadBalancer   10.7.241.36   35.232.176.113   8000:31220/TCP,8001:32107/TCP,8002:31682/TCP   1m
kubernetes   ClusterIP      10.7.240.1    <none>           443/TCP                                        1h
```

TRTIS exposes an HTTP endpoint on port 8000, and GRPC endpoint on port
8001 and a Prometheus metrics endpoint on port 8002. You can use curl
to get the status of the inference server from the HTTP endpoint.

```shell
$ curl 35.232.176.113:8000/api/status
```

Follow the
[instructions](https://github.com/NVIDIA/tensorrt-inference-server) to
build TRTIS example image and performance clients. You can then use
these examples to send requests to the server. For example, for an
image classification model use the image\_client example to perform
classification of an image.

```shell
$ image_client -u 35.232.176.113:8000 -m resnet50_netdef -s INCEPTION -c3 mug.jpg
Output probabilities:
batch 0: 504 (COFFEE MUG) = 0.777365267277
batch 0: 968 (CUP) = 0.213909029961
batch 0: 967 (ESPRESSO) = 0.00294389552437
```


# Cleanup

If you want to cleanup the inference server:
```
$ helm list
NAME            REVISION        UPDATED                         STATUS          CHART                           APP VERSION     NAMESPACE
wobbly-coral    1               Wed Feb 27 22:16:55 2019        DEPLOYED        nvidia-inference-server-1.0.0   1.0             default

$ helm delete wobbly-coral
```

# Install helm
You'll find the official helm install guide at this url: [https://github.com/helm/helm/blob/master/docs/install.md](https://github.com/helm/helm/blob/master/docs/install.md)

The following steps will give you a quick and dirty setup:
```
$ curl https://raw.githubusercontent.com/helm/helm/master/scripts/get | bash
$ kubectl create serviceaccount -n kube-system tiller
serviceaccount/tiller created
$ kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
$ helm init --service-account tiller --wait
```
