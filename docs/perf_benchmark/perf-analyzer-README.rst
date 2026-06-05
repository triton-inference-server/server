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


Triton Performance Analyzer
===========================

Triton Performance Analyzer is CLI tool which can help you optimize the
inference performance of models running on Triton Inference Server by
measuring changes in performance as you experiment with different
optimization strategies.

Features
========

Inference Load Modes
~~~~~~~~~~~~~~~~~~~~

-  `Concurrency Mode <docs/inference_load_modes.md#concurrency-mode>`__
   simlulates load by maintaining a specific concurrency of outgoing
   requests to the server

-  `Request Rate
   Mode <docs/inference_load_modes.md#request-rate-mode>`__ simulates
   load by sending consecutive requests at a specific rate to the server

-  `Custom Interval
   Mode <docs/inference_load_modes.md#custom-interval-mode>`__ simulates
   load by sending consecutive requests at specific intervals to the
   server

Performance Measurement Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Time Windows Mode <docs/measurements_metrics.md#time-windows>`__
   measures model performance repeatedly over a specific time interval
   until performance has stabilized

-  `Count Windows Mode <docs/measurements_metrics.md#count-windows>`__
   measures model performance repeatedly over a specific number of
   requests until performance has stabilized

Other Features
~~~~~~~~~~~~~~

-  `Sequence Models <../user_guide/architecture.md#stateful-models>`__,
   `Ensemble Models <../user_guide/architecture.md#ensemble-models>`__,
   and `Decoupled Models <../user_guide/decoupled_models.md>`__ can be
   profiled in addition to standard/stateless/coupled models

-  `Input Data <docs/input_data.md>`__ to model inferences can be
   auto-generated or specified as well as verifying output

-  `TensorFlow
   Serving <docs/benchmarking.md#benchmarking-tensorflow-serving>`__ and
   `TorchServe <docs/benchmarking.md#benchmarking-torchserve>`__ can be
   used as the inference server in addition to the default Triton server

Quick Start
===========

The steps below will guide you on how to start using Perf Analyzer.

Step 1: Start Triton Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export RELEASE=<yy.mm> # e.g. to use the release from the end of February of 2023, do `export RELEASE=23.02`

   docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3

   docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3

Step 2: Download ``simple`` Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # inside triton container
   git clone --depth 1 https://github.com/triton-inference-server/server

   mkdir model_repository ; cp -r server/docs/examples/model_repository/simple model_repository

Step 3: Start Triton Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # inside triton container
   tritonserver --model-repository $(pwd)/model_repository &> server.log &

   # confirm server is ready, look for 'HTTP/1.1 200 OK'
   curl -v localhost:8000/v2/health/ready

   # detach (CTRL-p CTRL-q)

Step 4: Start Triton SDK Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

   docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

Step 5: Run Perf Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # inside sdk container
   perf_analyzer -m simple

See the full `quick start guide <docs/quick_start.md>`__ for additional
tips on how to analyze output.

Documentation
=============

-  `Installation <docs/install.md>`__
-  `Perf Analyzer CLI <docs/cli.md>`__
-  `Inference Load Modes <docs/inference_load_modes.md>`__
-  `Input Data <docs/input_data.md>`__
-  `Measurements & Metrics <docs/measurements_metrics.md>`__
-  `Benchmarking <docs/benchmarking.md>`__

Contributing
============

Contributions to Triton Perf Analyzer are more than welcome. To
contribute please review the `contribution
guidelines <https://github.com/triton-inference-server/server/blob/main/CONTRIBUTING.md>`__,
then fork and create a pull request.

Reporting problems, asking questions
====================================

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve) document.
Ensure posted examples are:

-  minimal - use as little code as possible that still produces the same
   problem

-  complete - provide all parts needed to reproduce the problem. Check
   if you can strip external dependency and still show the problem. The
   less time we spend on reproducing problems the more time we have to
   fix it

-  verifiable - test the code you’re about to provide to make sure it
   reproduces the problem. Remove all other problems that are not
   related to your request/question.
