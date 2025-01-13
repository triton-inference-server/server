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


|License|

Triton Model Analyzer
=====================

   [!Warning]

   .. rubric:: LATEST RELEASE
      :name: latest-release

   You are currently on the ``main`` branch which tracks
   under-development progress towards the next release. The latest
   release of the Triton Model Analyzer is 1.42.0 and is available on
   branch
   `r24.07 <https://github.com/triton-inference-server/model_analyzer/tree/r24.07>`__.

Triton Model Analyzer is a CLI tool which can help you find a more
optimal configuration, on a given piece of hardware, for single,
multiple, ensemble, or BLS models running on a `Triton Inference
Server <https://github.com/triton-inference-server/server/>`__. Model
Analyzer will also generate reports to help you better understand the
trade-offs of the different configurations along with their compute and
memory requirements.

Features
========

Search Modes
~~~~~~~~~~~~

-  `Optuna Search <docs/config_search.md#optuna-search-mode>`__ **-ALPHA
   RELEASE-** allows you to search for every parameter that can be
   specified in the model configuration, using a hyperparameter
   optimization framework. Please see the
   `Optuna <https://optuna.org/>`__ website if you are interested in
   specific details on how the algorithm functions.

-  `Quick Search <docs/config_search.md#quick-search-mode>`__ will
   **sparsely** search the `Max Batch
   Size <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size>`__,
   `Dynamic
   Batching <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher>`__,
   and `Instance
   Group <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups>`__
   spaces by utilizing a heuristic hill-climbing algorithm to help you
   quickly find a more optimal configuration

-  `Automatic Brute
   Search <docs/config_search.md#automatic-brute-search>`__ will
   **exhaustively** search the `Max Batch
   Size <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size>`__,
   `Dynamic
   Batching <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher>`__,
   and `Instance
   Group <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups>`__
   parameters of your model configuration

-  `Manual Brute Search <docs/config_search.md#manual-brute-search>`__
   allows you to create manual sweeps for every parameter that can be
   specified in the model configuration

Model Types
~~~~~~~~~~~

-  `Ensemble <docs/model_types.md#ensemble>`__: Model Analyzer can help
   you find the optimal settings when profiling an ensemble model

-  `BLS <docs/model_types.md#bls>`__: Model Analyzer can help you find
   the optimal settings when profiling a BLS model

-  `Multi-Model <docs/model_types.md#multi-model>`__: Model Analyzer can
   help you find the optimal settings when profiling multiple concurrent
   models

-  `LLM <docs/model_types.md#llm>`__: Model Analyzer can help you find
   the optimal settings when profiling Large Language Models

Other Features
~~~~~~~~~~~~~~

-  `Detailed and summary reports <docs/report.md>`__: Model Analyzer is
   able to generate summarized and detailed reports that can help you
   better understand the trade-offs between different model
   configurations that can be used for your model.

-  `QoS Constraints <docs/config.md#constraint>`__: Constraints can help
   you filter out the Model Analyzer results based on your QoS
   requirements. For example, you can specify a latency budget to filter
   out model configurations that do not satisfy the specified latency
   threshold.

Examples and Tutorials
======================

**Single Model**
~~~~~~~~~~~~~~~~

See the `Single Model Quick Start <docs/quick_start.md>`__ for a guide
on how to use Model Analyzer to profile, analyze and report on a simple
PyTorch model.

**Multi Model**
~~~~~~~~~~~~~~~

See the `Multi-model Quick Start <docs/mm_quick_start.md>`__ for a guide
on how to use Model Analyzer to profile, analyze and report on two
models running concurrently on the same GPU.

**Ensemble Model**
~~~~~~~~~~~~~~~~~~

See the `Ensemble Model Quick Start <docs/ensemble_quick_start.md>`__
for a guide on how to use Model Analyzer to profile, analyze and report
on a simple Ensemble model.

**BLS Model**
~~~~~~~~~~~~~

See the `BLS Model Quick Start <docs/bls_quick_start.md>`__ for a guide
on how to use Model Analyzer to profile, analyze and report on a simple
BLS model.

Documentation
=============

-  `Installation <docs/install.md>`__
-  `Model Analyzer CLI <docs/cli.md>`__
-  `Launch Modes <docs/launch_modes.md>`__
-  `Configuring Model Analyzer <docs/config.md>`__
-  `Model Analyzer Metrics <docs/metrics.md>`__
-  `Model Config Search <docs/config_search.md>`__
-  `Model Types <docs/model_types.md>`__
-  `Checkpointing <docs/checkpoints.md>`__
-  `Model Analyzer Reports <docs/report.md>`__
-  `Deployment with Kubernetes <docs/kubernetes_deploy.md>`__

Terminology
===========

Below are definitions of some commonly used terms in Model Analyzer:

-  **Model Type** - Category of model being profiled. Examples of this
   include single, multi, ensemble, BLS, etc..
-  **Search Mode** - How Model Analyzer explores the possible
   configuration space when profiling. This is either exhaustive (brute)
   or heuristic (quick/optuna).
-  **Model Config Search** - The cross product of model type and search
   mode.
-  **Launch Mode** - How the Triton Server is deployed and used by Model
   Analyzer.

Reporting problems, asking questions
====================================

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve) document.
Ensure posted examples are:

-  minimal – use as little code as possible that still produces the same
   problem

-  complete – provide all parts needed to reproduce the problem. Check
   if you can strip external dependency and still show the problem. The
   less time we spend on reproducing problems the more time we have to
   fix it

-  verifiable – test the code you’re about to provide to make sure it
   reproduces the problem. Remove all other problems that are not
   related to your request/question.

.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg
   :target: https://opensource.org/licenses/Apache-2.0
