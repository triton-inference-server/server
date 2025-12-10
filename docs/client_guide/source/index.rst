Triton Inference Server Python API
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
Requirements
Installation
Hello World
In-Process Server Example
Send Inference Request
Stable Diffusion Example

Introduction
------------

The Triton Inference Server provides an optimized cloud and edge inferencing solution.
This documentation focuses on the Python API.

Requirements
------------

- Linux system with Docker installed.
- CUDA driver compatible with Triton.
- Python 3.13+ (for Python API)

Installation
------------

.. code-block:: bash

   git clone https://github.com/triton-inference-server/tutorials.git
   cd tutorials/Triton_Inference_Server_Python_API
   ./build.sh

Hello World
-----------

Start the container:

.. code-block:: bash

   ./run.sh

Enter Python shell:

.. code-block:: bash

   python3

In-Process Server Example
-------------------------

.. code-block:: python

   import tritonserver
   server = tritonserver.Server(model_repository="/workspace/identity-models")
   server.start()

Send Inference Request
----------------------

.. code-block:: python

   model = server.model("identity")
   responses = model.infer(inputs={"string_input":[["hello world!"]]})

   for response in responses:
       print(response.outputs["string_output"].to_string_array())

Stable Diffusion Example
-------------------------


Build diffusion image:

.. code-block:: bash

   ./build.sh --framework diffusion --build-models

Start container:

.. code-block:: bash

   ./run.sh --framework diffusion

Python In-Process Server:

.. code-block:: python

   import tritonserver
   import numpy
   from PIL import Image

   server = tritonserver.Server(model_repository="/workspace/diffusion-models")
   server.start()

Send request and save image:

.. code-block:: python

   model = server.model("stable_diffusion")
   responses = model.infer(inputs={"prompt":[["butterfly in new york, realistic, 4k, photograph"]]})

   for response in responses:
       generated_image = numpy.from_dlpack(response.outputs["generated_image"])
       generated_image = generated_image.squeeze().astype(numpy.uint8)
       image_ = Image.fromarray(generated_image)
       image_.save("sample_generated_image.jpg")
