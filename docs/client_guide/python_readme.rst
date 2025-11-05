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

Triton Inference Server In-Process Python API
=============================================

Starting with release 24.01, Triton Inference Server includes a Python package
that allows developers to embed Triton Inference Server instances in their
Python applications. The in-process Python API matches the functionality of
the in-process C API while providing a higher-level abstraction.

.. note::
   As the API is in BETA, please expect some changes. All feedback is welcome.

Contents
--------

- `Requirements <#requirements>`__
- `Installation <#installation>`__
- `Hello World <#hello-world>`__
- `Stable Diffusion <#stable-diffusion>`__
- `Ray Serve Deployment <../tutorials/Triton_Inference_Server_Python_API/examples/rayserve>`__

Requirements
------------

- Linux system with Docker installed.
- CUDA driver meeting the requirements in the
  `NVIDIA Deep Learning Framework support matrix <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`__.

Installation
------------

The tutorial and Python API package are designed to run within the
``nvcr.io/nvidia/tritonserver:24.01-py3`` Docker image.

Convenience scripts are provided to create a Docker image with the Python API
and example dependencies.

Triton Inference Server 24.01 + Python API
------------------------------------------

Clone Repository
^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/triton-inference-server/tutorials.git
   cd tutorials/Triton_Inference_Server_Python_API

Build ``triton-python-api:r24.01`` Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./build.sh

Supported Backends
^^^^^^^^^^^^^^^^^^

The built image includes all backends shipped by default in the
``nvcr.io/nvidia/tritonserver:24.01-py3`` container:

::

   dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorrt

Included Models
^^^^^^^^^^^^^^

The ``default`` build includes an ``identity`` model for testing input/output operations:

- Inputs: ``data_type_input`` (string, fp16, etc.)  
- Outputs: ``data_type_output``

Hello World
-----------

Start Container
^^^^^^^^^^^^^^

.. code-block:: bash

   ./run.sh

Enter Python Shell
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3

In-Process Server Example
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tritonserver
   server = tritonserver.Server(model_repository="/workspace/identity-models")
   server.start()

List Models
^^^^^^^^^^^

::

   server.models()

Example Output
^^^^^^^^^^^^^^

.. code-block:: python

   {('identity', 1): {'name': 'identity', 'version': 1, 'state': 'READY'}}

Send an Inference Request
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = server.model("identity")
   responses = model.infer(inputs={"string_input":[["hello world!"]]})

Iterate Responses
^^^^^^^^^^^^^^^^^

.. code-block:: python

   for response in responses:
       print(response.outputs["string_output"].to_string_array())

gRPC Python Client
-----------------

.. code-block:: python

   import tritonclient.grpc as grpcclient
   import numpy as np

   # Connect to server
   url = "localhost:8001"
   client = grpcclient.InferenceServerClient(url)

   # Input data
   input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

   # Send request
   response = client.infer(model_name="identity",
                           inputs={"input": input_data})

   # Output
   print(response.as_numpy("output"))

Stable Diffusion Example
-----------------------

Build diffusion image
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./build.sh --framework diffusion --build-models

Start Container
^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run.sh --framework diffusion

Python In-Process Server
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tritonserver
   import numpy
   from PIL import Image

   server = tritonserver.Server(model_repository="/workspace/diffusion-models")
   server.start()

List Models
^^^^^^^^^^^

::

   server.models()

Send Request and Save Image
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = server.model("stable_diffusion")
   responses = model.infer(inputs={"prompt":[["butterfly in new york, realistic, 4k, photograph"]]})

   for response in responses:
       generated_image = numpy.from_dlpack(response.outputs["generated_image"])
       generated_image = generated_image.squeeze().astype(numpy.uint8)
       image_ = Image.fromarray(generated_image)
       image_.save("sample_generated_image.jpg")

Example Output
^^^^^^^^^^^^^^

.. figure:: ../tutorials/Triton_Inference_Server_Python_API/docs/sample_generated_image.jpg
   :alt: sample_generated_image

   sample_generated_image
