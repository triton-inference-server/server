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


Triton Inference Server In-Process Python API [BETA]
====================================================

Starting with release 24.01 Triton Inference Server will include a
Python package enabling developers to embed Triton Inference Server
instances in their Python applications. The in-process Python API is
designed to match the functionality of the in-process C API while
providing a higher level abstraction. At its core the API relies on a
1:1 python binding of the C API and provides all the flexibility and
power of the C API with a simpler to use interface.

   [!Note] As the API is in BETA please expect some changes as we test
   out different features and get feedback. All feedback is weclome and
   we look forward to hearing from you!

| `Requirements <#requirements>`__ \| `Installation <#installation>`__
  \| `Hello World <#hello-world>`__ \| `Stable
  Diffusion <#stable-diffusion>`__ \| `Ray Serve
  Deployment <../tutorials/Triton_Inference_Server_Python_API/examples/rayserve>`__ \|
Requirements
------------

The following instructions require a linux system with Docker installed.
For CUDA support, make sure your CUDA driver meets the requirements in
“NVIDIA Driver” section of Deep Learning Framework support matrix:
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

Installation
------------

The tutorial and Python API package are designed to be installed and run
within the ``nvcr.io/nvidia/tritonserver:24.01-py3`` docker image.

A set of convenience scripts are provided to create a docker image based
on the ``nvcr.io/nvidia/tritonserver:24.01-py3`` image with the Python
API installed plus additional dependencies required for the examples.

Triton Inference Server 24.01 + Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone Repository
^^^^^^^^^^^^^^^^

.. code:: bash
   git clone https://github.com/triton-inference-server/tutorials.git
   cd tutorials/Triton_Inference_Server_Python_API
Build ``triton-python-api:r24.01`` Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash
   ./build.sh
Supported Backends
^^^^^^^^^^^^^^^^^^

The built image includes all the backends shipped by default in the
tritonserver ``nvcr.io/nvidia/tritonserver:24.01-py3`` container.

::

   dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt

Included Models
^^^^^^^^^^^^^^^

The ``default`` build includes an ``identity`` model that can be used
for exercising basic operations including sending input tensors of
different data types. The ``identity`` model copies provided inputs of
``shape [-1, -1]`` to outputs of shape ``[-1, -1]``. Inputs are named
``data_type_input`` and outputs are named ``data_type_output``
(e.g. ``string_input``, ``string_output``, ``fp16_input``,
``fp16_output``).

Hello World
-----------

Start ``triton-python-api:r24.01`` Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command starts a container and volume mounts the current
directory as ``workspace``.

.. code:: bash
   ./run.sh
Enter Python Shell
~~~~~~~~~~~~~~~~~~

.. code:: bash
   python3
Create and Start a Server Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python
   import tritonserver
   server = tritonserver.Server(model_repository="/workspace/identity-models")
   server.start()
List Models
~~~~~~~~~~~

::

   server.models()

Example Output
^^^^^^^^^^^^^^

``server.models()`` returns a dictionary of the available models with
their current state.

.. code:: python
   {('identity', 1): {'name': 'identity', 'version': 1, 'state': 'READY'}}
Send an Inference Request
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python
   model = server.model("identity")
   responses = model.infer(inputs={"string_input":[["hello world!"]]})
Iterate through Responses
~~~~~~~~~~~~~~~~~~~~~~~~~

``model.infer()`` returns an iterator that can be used to process the
results of an inference request.

.. code:: python
   for response in responses:
       print(response.outputs["string_output"].to_string_array())
.. _example-output-1:

Example Output
^^^^^^^^^^^^^^

.. code:: python
   [['hello world!']]
Stable Diffusion
----------------

This example is based on the
`Popular_Models_Guide/StableDiffusion <../tutorials/Popular_Models_Guide/StableDiffusion/README.html>`__
tutorial.

Build ``triton-python-api:r24.01-diffusion`` Image and Stable Diffusion Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please note the following command will take many minutes depending on
your hardware configuration and network connection.

.. code:: bash
      ./build.sh --framework diffusion --build-models
.. _supported-backends-1:

Supported Backends
^^^^^^^^^^^^^^^^^^

The built image includes all the backends shipped by default in the
tritonserver ``nvcr.io/nvidia/tritonserver:24.01-py3`` container.

::

   dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt

.. _included-models-1:

Included Models
^^^^^^^^^^^^^^^

The ``diffusion`` build includes a ``stable_diffustion`` pipeline that
takes a text prompt and returns a generated image. For more details on
the models and pipeline please see the
`Popular_Models_Guide/StableDiffusion <../tutorials/Popular_Models_Guide/StableDiffusion/README.html>`__
tutorial.

Start Container
~~~~~~~~~~~~~~~

The following command starts a container and volume mounts the current
directory as ``workspace``.

.. code:: bash
   ./run.sh --framework diffusion
.. _enter-python-shell-1:

Enter Python Shell
~~~~~~~~~~~~~~~~~~

.. code:: bash
   python3
.. _create-and-start-a-server-instance-1:

Create and Start a Server Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python
   import tritonserver
   import numpy
   from PIL import Image
   server = tritonserver.Server(model_repository="/workspace/diffusion-models")
   server.start()
.. _list-models-1:

List Models
~~~~~~~~~~~

::

   server.models()

.. _example-output-2:

Example Output
^^^^^^^^^^^^^^

.. code:: python
   {('stable_diffusion', 1): {'name': 'stable_diffusion', 'version': 1, 'state': 'READY'}, ('text_encoder', 1): {'name': 'text_encoder', 'version': 1, 'state': 'READY'}, ('vae', 1): {'name': 'vae', 'version': 1, 'state': 'READY'}}
.. _send-an-inference-request-1:

Send an Inference Request
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python
   model = server.model("stable_diffusion")
   responses = model.infer(inputs={"prompt":[["butterfly in new york, realistic, 4k, photograph"]]})
Iterate through Responses and save image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python
   for response in responses:
       generated_image = numpy.from_dlpack(response.outputs["generated_image"])
       generated_image = generated_image.squeeze().astype(numpy.uint8)
       image_ = Image.fromarray(generated_image)
       image_.save("sample_generated_image.jpg")
.. _example-output-3:

Example Output
^^^^^^^^^^^^^^

.. figure:: ../tutorials/Triton_Inference_Server_Python_API/docs/sample_generated_image.jpg
   :alt: sample_generated_image

   sample_generated_image