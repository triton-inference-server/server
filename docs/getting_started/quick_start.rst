..
.. Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions
.. are met:
..  * Redistributions of source code must retain the above copyright
..    notice, this list of conditions and the following disclaimer.
..  * Redistributions in binary form must reproduce the above copyright
..    notice, this list of conditions and the following disclaimer in the
..    documentation and/or other materials provided with the distribution.
..  * Neither the name of NVIDIA CORPORATION nor the names of its
..    contributors may be used to endorse or promote products derived
..    from this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
.. EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
.. PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
.. OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
.. (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
.. OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

.. raw:: html


Quickstart
==========

**New to Triton Inference Server and want do just deploy your model
quickly?** Make use of `these
tutorials <../tutorials/README.html#quick-deploy>`__ to begin your Triton
journey!

The Triton Inference Server is available as `buildable source
code <../customization_guide/build.html>`__, but the easiest way to
install and run Triton is to use the pre-built Docker image available
from the `NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`__.

Launching and maintaining Triton Inference Server revolves around the
use of building model repositories. This tutorial will cover:

-  Creating a Model Repository
-  Launching Triton
-  Send an Inference Request

Create A Model Repository
-------------------------

The `model repository <../user_guide/model_repository.html>`__ is the
directory where you place the models that you want Triton to serve. An
example model repository is included in the
`docs/examples/model_repository <https://github.com/triton-inference-server/server/blob/main/docs/examples/model_repository>`__.
Before using the repository, you must fetch any missing model definition
files from their public model zoos via the provided script.

::

   $ cd docs/examples
   $ ./fetch_models.sh

Launch Triton
-------------

Triton is optimized to provide the best inferencing performance by using
GPUs, but it can also work on CPU-only systems. In both cases you can
use the same Triton Docker image.

Run on System with GPUs
~~~~~~~~~~~~~~~~~~~~~~~

Use the following command to run Triton with the example model
repository you just created. The `NVIDIA Container
Toolkit <https://github.com/NVIDIA/nvidia-docker>`__ must be installed
for Docker to recognize the GPU(s). The –gpus=1 flag indicates that 1
system GPU should be made available to Triton for inferencing.

::

   $ docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models

Where <xx.yy> is the version of Triton that you want to use (and pulled
above). After you start Triton you will see output on the console
showing the server starting up and loading the model. When you see
output like the following, Triton is ready to accept inference requests.

::

   +----------------------+---------+--------+
   | Model                | Version | Status |
   +----------------------+---------+--------+
   | <model_name>         | <v>     | READY  |
   | ..                   | .       | ..     |
   | ..                   | .       | ..     |
   +----------------------+---------+--------+
   ...
   ...
   ...
   I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
   I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
   I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002

All the models should show “READY” status to indicate that they loaded
correctly. If a model fails to load the status will report the failure
and a reason for the failure. If your model is not displayed in the
table check the path to the model repository and your CUDA drivers.

Run on CPU-Only System
~~~~~~~~~~~~~~~~~~~~~~

On a system without GPUs, Triton should be run without using the –gpus
flag to Docker, but is otherwise identical to what is described above.

::

   $ docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models

Because the –gpus flag is not used, a GPU is not available and Triton
will therefore be unable to load any model configuration that requires a
GPU.

Verify Triton Is Running Correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Triton’s *ready* endpoint to verify that the server and the models
are ready for inference. From the host system use curl to access the
HTTP endpoint that indicates server status.

::

   $ curl -v localhost:8000/v2/health/ready
   ...
   < HTTP/1.1 200 OK
   < Content-Length: 0
   < Content-Type: text/plain

The HTTP request returns status 200 if Triton is ready and non-200 if it
is not ready.

Send an Inference Request
-------------------------

Use docker pull to get the client libraries and examples image from NGC.

::

   $ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

Where <xx.yy> is the version that you want to pull. Run the client
image.

::

   $ docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

From within the nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk image, run
the example image-client application to perform image classification
using the example densenet_onnx model.

To send a request for the densenet_onnx model use an image from the
/workspace/images directory. In this case we ask for the top 3
classifications.

::

   $ /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
   Request 0, batch size 1
   Image '/workspace/images/mug.jpg':
       15.346230 (504) = COFFEE MUG
       13.224326 (968) = CUP
       10.422965 (505) = COFFEEPOT