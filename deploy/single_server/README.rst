..
  # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

|License|

TensorRT Inference Server Helm Chart
====================================

Simple helm chart for installing a single instance of the NVIDIA
TensorRT Inference Server. This guide assumes you already have a
functional Kubernetes cluster and helm installed (see below for
instructions on installing helm). Your cluster must be configured with
support for the NVIDIA driver and CUDA version required by the version
of the inference server you are using. For example, for inference
server release 19.03 you need to have CUDA 10.1.

The steps below describe how to set-up a model repository, use helm to
launch the inference server, and then send inference requests to the
running server.

Get The Inference Server Source
-------------------------------

The need to have a local copy of the inference server source
repository to access the helm chart and the example model repository::

  $ git clone https://github.com/NVIDIA/tensorrt-inference-server.git

Model Repository
----------------

TensorRT Inference Server needs a repository of models that it will
make available for inferencing. For this example you will place the
model repository in a Google Cloud Storage bucket::

  $ gsutil mb gs://tensorrt-inference-server-repository

Following the `instructions
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/run.html#example-model-repository>`_
download the example model repository to your system and copy it into
the GCS bucket::

  $ gsutil cp -r docs/examples/model_repository gs://tensorrt-inference-server-repository/model_repository

Make sure the bucket permissions are set so that the inference server
can access the model repository.

Running The Inference Server
----------------------------

Once you have helm installed (see below if you need help installing
helm) and your model repository ready, you can deploy the inference
server using the default configuration with::

  $ helm install deploy/single_server

You can use kubectl to wait until the inference server pod is running::

  $ kubectl get pods
  NAME                                                              READY   STATUS              RESTARTS   AGE
  wobbley-coral-tensorrt-inference-server-5f74b55885-n6lt7   1/1     Running   0          2m21s

There are several ways of overriding the default configuration as
described in this `helm documentation
<https://helm.sh/docs/using_helm/#customizing-the-chart-before-installing>`_.

For example, you can edit the values.yaml file directly or you can use
the `--set` option to override a single parameter with the CLI, for
example::

  helm install tensorrt-inference-server --set image.imageName="nvcr.io/nvidia/tensorrtserver:custom-tag"

You can also use a file by writing your own "values.yaml" file with
the values you want to override and pass it to helm::

  $ cat << EOF > config.yaml
  namespace: MyCustomNamespace
  image:
    imageName: nvcr.io/nvidia/tensorrtserver:custom-tag
    modelRepositoryPath: gs://my_model_repository
  EOF

  $ helm install -f config.yaml tensorrt-inference-server

Using the TensorRT Inference Server
-----------------------------------

Now that the inference server is running you can send HTTP or GRPC
requests to it to perform inferencing. By default, the inferencing
service is exposed with a LoadBalancer service type. Use the following
to find the external IP for the inference server. In this case it is
35.232.176.113::

  $ kubectl get services
  NAME         TYPE           CLUSTER-IP    EXTERNAL-IP      PORT(S)                                        AGE
  inference-se LoadBalancer   10.7.241.36   35.232.176.113   8000:31220/TCP,8001:32107/TCP,8002:31682/TCP   1m
  kubernetes   ClusterIP      10.7.240.1    <none>           443/TCP                                        1h

The inference server exposes an HTTP endpoint on port 8000, and GRPC
endpoint on port 8001 and a Prometheus metrics endpoint on
port 8002. You can use curl to get the status of the inference server
from the HTTP endpoint::

  $ curl 35.232.176.113:8000/api/status

Follow the `instructions
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/client.html#getting-the-client-examples>`_
to get the example image classification client that can be used to
perform inferencing using image classification models being served by
the inference server. For example::

  $ image_client -u 35.232.176.113:8000 -m resnet50_netdef -s INCEPTION -c3 mug.jpg
  Output probabilities:
  batch 0: 504 (COFFEE MUG) = 0.777365267277
  batch 0: 968 (CUP) = 0.213909029961
  batch 0: 967 (ESPRESSO) = 0.00294389552437

Cleanup
-------

Once you've finished using the inference server you should use helm to delete the deployment::

  $ helm list
  NAME            REVISION        UPDATED                         STATUS          CHART                           APP VERSION     NAMESPACE
  wobbly-coral    1               Wed Feb 27 22:16:55 2019        DEPLOYED        tensorrt-inference-server-1.0.0   1.0             default

  $ helm delete wobbly-coral

You may also want to delete the GCS bucket you created to hold the model repository::

  $ gsutil rm -r gs://tensorrt-inference-server-repository

Installing Helm
---------------

The following steps from the `official helm install guide
<https://github.com/helm/helm/blob/master/docs/install.md>`_ will give
you a quick setup::

  $ curl https://raw.githubusercontent.com/helm/helm/master/scripts/get | bash
  $ kubectl create serviceaccount -n kube-system tiller
  serviceaccount/tiller created
  $ kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
  $ helm init --service-account tiller --wait

.. |License| image:: https://img.shields.io/badge/License-BSD3-lightgrey.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
