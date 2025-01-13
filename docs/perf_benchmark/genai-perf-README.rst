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


GenAI-Perf
==========

GenAI-Perf is a command line tool for measuring the throughput and
latency of generative AI models as served through an inference server.
For large language models (LLMs), GenAI-Perf provides metrics such as
`output token throughput <#output_token_throughput_metric>`__, `time to
first token <#time_to_first_token_metric>`__, `inter token
latency <#inter_token_latency_metric>`__, and `request
throughput <#request_throughput_metric>`__. For a full list of metrics
please see the `Metrics section <#metrics>`__.

Users specify a model name, an inference server URL, the type of inputs
to use (synthetic or from dataset), and the type of load to generate
(number of concurrent requests, request rate).

GenAI-Perf generates the specified load, measures the performance of the
inference server and reports the metrics in a simple table as console
output. The tool also logs all results in a csv and json file that can
be used to derive additional metrics and visualizations. The inference
server must already be running when GenAI-Perf is run.

You can use GenAI-Perf to run performance benchmarks on - `Large
Language Models <docs/tutorial.md>`__ - `Vision Language
Models <docs/multi_modal.md>`__ - `Embedding
Models <docs/embeddings.md>`__ - `Ranking Models <docs/rankings.md>`__ -
`Multiple LoRA Adapters <docs/lora.md>`__

   [!Note] GenAI-Perf is currently in early release and under rapid
   development. While we will try to remain consistent, command line
   options and functionality are subject to change as the tool matures.

.. raw:: html

   <!--
   ======================
   INSTALLATION
   ======================
   -->

Installation
------------

The easiest way to install GenAI-Perf is through `Triton Server SDK
container <https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver>`__.
Install the latest release using the following command:

.. code:: bash

   export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

   docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

   # Check out genai_perf command inside the container:
   genai-perf --help

.. raw:: html

   <details>

Alternatively, to install from source:

Since GenAI-Perf depends on Perf Analyzer, you’ll need to install the
Perf Analyzer binary:

Install Perf Analyzer (Ubuntu, Python 3.8+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE**: you must already have CUDA 12 installed (checkout the `CUDA
installation
guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__).

.. code:: bash

   pip install tritonclient

   apt update && apt install -y --no-install-recommends libb64-0d libcurl4

You can also build Perf Analyzer `from
source <../docs/install.md#build-from-source>`__ as well.

Install GenAI-Perf from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/triton-inference-server/perf_analyzer.git && cd perf_analyzer

   pip install -e genai-perf

.. raw:: html

   </details>

.. raw:: html

   <!--
   ======================
   QUICK START
   ======================
   -->

Quick Start
-----------

In this quick start, we will use GenAI-Perf to run performance
benchmarking on the GPT-2 model running on Triton Inference Server with
a TensorRT-LLM engine.

Serve GPT-2 TensorRT-LLM model using Triton CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can follow the `quickstart
guide <https://github.com/triton-inference-server/triton_cli?tab=readme-ov-file#serving-a-trt-llm-model>`__
on Triton CLI github repo to run GPT-2 model locally. The full
instructions are copied below for convenience:

.. code:: bash

   # This container comes with all of the dependencies for building TRT-LLM engines
   # and serving the engine with Triton Inference Server.
   docker run -ti \
       --gpus all \
       --network=host \
       --shm-size=1g --ulimit memlock=-1 \
       -v /tmp:/tmp \
       -v ${HOME}/models:/root/models \
       -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
       nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3

   # Install the Triton CLI
   pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.8

   # Build TRT LLM engine and generate a Triton model repository pointing at it
   triton remove -m all
   triton import -m gpt2 --backend tensorrtllm

   # Start Triton pointing at the default model repository
   triton start

Running GenAI-Perf
~~~~~~~~~~~~~~~~~~

Now we can run GenAI-Perf from Triton Inference Server SDK container:

.. code:: bash

   export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

   docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

   # Run GenAI-Perf in the container:
   genai-perf profile \
     -m gpt2 \
     --service-kind triton \
     --backend tensorrtllm \
     --num-prompts 100 \
     --random-seed 123 \
     --synthetic-input-tokens-mean 200 \
     --synthetic-input-tokens-stddev 0 \
     --streaming \
     --output-tokens-mean 100 \
     --output-tokens-stddev 0 \
     --output-tokens-mean-deterministic \
     --tokenizer hf-internal-testing/llama-tokenizer \
     --concurrency 1 \
     --measurement-interval 4000 \
     --profile-export-file my_profile_export.json \
     --url localhost:8001

Example output:

::

                                      LLM Metrics
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
   ┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
   │ Time to first token (ms) │  11.70 │   9.88 │  17.21 │  14.35 │  12.01 │  11.87 │
   │ Inter token latency (ms) │   1.46 │   1.08 │   1.89 │   1.87 │   1.62 │   1.52 │
   │     Request latency (ms) │ 161.24 │ 153.45 │ 200.74 │ 200.66 │ 179.43 │ 162.23 │
   │   Output sequence length │ 103.39 │  95.00 │ 134.00 │ 120.08 │ 107.30 │ 105.00 │
   │    Input sequence length │ 200.01 │ 200.00 │ 201.00 │ 200.13 │ 200.00 │ 200.00 │
   └──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
   Output token throughput (per sec): 635.61
   Request throughput (per sec): 6.15

See `Tutorial <docs/tutorial.md>`__ for additional examples.

.. raw:: html

   <!--
   ======================
   VISUALIZATION
   ======================
   -->

Visualization
-------------

GenAI-Perf can also generate various plots that visualize the
performance of the current profile run. This is disabled by default but
users can easily enable it by passing the ``--generate-plots`` option
when running the benchmark:

.. code:: bash

   genai-perf profile \
     -m gpt2 \
     --service-kind triton \
     --backend tensorrtllm \
     --streaming \
     --concurrency 1 \
     --generate-plots

This will generate a `set of default
plots <docs/compare.md#example-plots>`__ such as: - Time to first token
(TTFT) analysis - Request latency analysis - TTFT vs Input sequence
lengths - Inter token latencies vs Token positions - Input sequence
lengths vs Output sequence lengths

Using ``compare`` Subcommand to Visualize Multiple Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``compare`` subcommand in GenAI-Perf facilitates users in comparing
multiple profile runs and visualizing the differences through plots.

Usage
^^^^^

Assuming the user possesses two profile export JSON files, namely
``profile1.json`` and ``profile2.json``, they can execute the
``compare`` subcommand using the ``--files`` option:

.. code:: bash

   genai-perf compare --files profile1.json profile2.json

Executing the above command will perform the following actions under the
``compare`` directory: 1. Generate a YAML configuration file
(e.g. ``config.yaml``) containing the metadata for each plot generated
during the comparison process. 2. Automatically generate the `default
set of plots <docs/compare.md#example-plots>`__ (e.g. TTFT vs. Input
Sequence Lengths) that compare the two profile runs.

::

   compare
   ├── config.yaml
   ├── distribution_of_input_sequence_lengths_to_output_sequence_lengths.jpeg
   ├── request_latency.jpeg
   ├── time_to_first_token.jpeg
   ├── time_to_first_token_vs_input_sequence_lengths.jpeg
   ├── token-to-token_latency_vs_output_token_position.jpeg
   └── ...

Customization
^^^^^^^^^^^^^

Users have the flexibility to iteratively modify the generated YAML
configuration file to suit their specific requirements. They can make
alterations to the plots according to their preferences and execute the
command with the ``--config`` option followed by the path to the
modified configuration file:

.. code:: bash

   genai-perf compare --config compare/config.yaml

This command will regenerate the plots based on the updated
configuration settings, enabling users to refine the visual
representation of the comparison results as per their needs.

See `Compare documentation <docs/compare.md>`__ for more details.

.. raw:: html

   <!--
   ======================
   MODEL INPUTS
   ======================
   -->

Model Inputs
------------

GenAI-Perf supports model input prompts from either synthetically
generated inputs, or from the HuggingFace
`OpenOrca <https://huggingface.co/datasets/Open-Orca/OpenOrca>`__ or
`CNN_DailyMail <https://huggingface.co/datasets/cnn_dailymail>`__
datasets. This is specified using the ``--input-dataset`` CLI option.

When the dataset is synthetic, you can specify the following options: \*
``--num-prompts <int>``: The number of unique prompts to generate as
stimulus, >= 1. \* ``--synthetic-input-tokens-mean <int>``: The mean of
number of tokens in the generated prompts when using synthetic data, >=
1. \* ``--synthetic-input-tokens-stddev <int>``: The standard deviation
of number of tokens in the generated prompts when using synthetic data,
>= 0. \* ``--random-seed <int>``: The seed used to generate random
values, >= 0.

When the dataset is coming from HuggingFace, you can specify the
following options: \* ``--input-dataset {openorca,cnn_dailymail}``:
HuggingFace dataset to use for benchmarking. \* ``--num-prompts <int>``:
The number of unique prompts to generate as stimulus, >= 1.

When the dataset is coming from a file, you can specify the following
options: \* ``--input-file <path>``: The input file containing the
prompts to use for benchmarking as JSON objects.

For any dataset, you can specify the following options: \*
``--output-tokens-mean <int>``: The mean number of tokens in each
output. Ensure the ``--tokenizer`` value is set correctly, >= 1. \*
``--output-tokens-stddev <int>``: The standard deviation of the number
of tokens in each output. This is only used when output-tokens-mean is
provided, >= 1. \* ``--output-tokens-mean-deterministic``: When using
``--output-tokens-mean``, this flag can be set to improve precision by
setting the minimum number of tokens equal to the requested number of
tokens. This is currently supported with the Triton service-kind. Note
that there is still some variability in the requested number of output
tokens, but GenAi-Perf attempts its best effort with your model to get
the right number of output tokens.

You can optionally set additional model inputs with the following
option: \* ``--extra-inputs <input_name>:<value>``: An additional input
for use with the model with a singular value, such as ``stream:true`` or
``max_tokens:5``. This flag can be repeated to supply multiple extra
inputs.

For `Large Language Models <docs/tutorial.md>`__, there is no batch size
(i.e. batch size is always ``1``). Each request includes the inputs for
one individual inference. Other modes such as the
`embeddings <docs/embeddings.md>`__ and `rankings <docs/rankings.md>`__
endpoints support client-side batching, where ``--batch-size N`` means
that each request sent will include the inputs for ``N`` separate
inferences, allowing them to be processed together.

.. raw:: html

   <!--
   ======================
   METRICS
   ======================
   -->

Metrics
-------

GenAI-Perf collects a diverse set of metrics that captures the
performance of the inference server.

+-----------------------+-----------------------+-----------------------+
| Metric                | Description           | Aggregations          |
+=======================+=======================+=======================+
| Time to First Token   | Time between when a   | Avg, min, max, p99,   |
|                       | request is sent and   | p90, p75              |
|                       | when its first        |                       |
|                       | response is received, |                       |
|                       | one value per request |                       |
|                       | in benchmark          |                       |
+-----------------------+-----------------------+-----------------------+
| Inter Token Latency   | Time between          | Avg, min, max, p99,   |
|                       | intermediate          | p90, p75              |
|                       | responses for a       |                       |
|                       | single request        |                       |
|                       | divided by the number |                       |
|                       | of generated tokens   |                       |
|                       | of the latter         |                       |
|                       | response, one value   |                       |
|                       | per response per      |                       |
|                       | request in benchmark  |                       |
+-----------------------+-----------------------+-----------------------+
| Request Latency       | Time between when a   | Avg, min, max, p99,   |
|                       | request is sent and   | p90, p75              |
|                       | when its final        |                       |
|                       | response is received, |                       |
|                       | one value per request |                       |
|                       | in benchmark          |                       |
+-----------------------+-----------------------+-----------------------+
| Output Sequence       | Total number of       | Avg, min, max, p99,   |
| Length                | output tokens of a    | p90, p75              |
|                       | request, one value    |                       |
|                       | per request in        |                       |
|                       | benchmark             |                       |
+-----------------------+-----------------------+-----------------------+
| Input Sequence Length | Total number of input | Avg, min, max, p99,   |
|                       | tokens of a request,  | p90, p75              |
|                       | one value per request |                       |
|                       | in benchmark          |                       |
+-----------------------+-----------------------+-----------------------+
| Output Token          | Total number of       | None–one value per    |
| Throughput            | output tokens from    | benchmark             |
|                       | benchmark divided by  |                       |
|                       | benchmark duration    |                       |
+-----------------------+-----------------------+-----------------------+
| Request Throughput    | Number of final       | None–one value per    |
|                       | responses from        | benchmark             |
|                       | benchmark divided by  |                       |
|                       | benchmark duration    |                       |
+-----------------------+-----------------------+-----------------------+

.. raw:: html

   <!--
   ======================
   COMMAND LINE OPTIONS
   ======================
   -->

Command Line Options
--------------------

``-h``
''''''

``--help``
''''''''''

Show the help message and exit.

Endpoint Options:
~~~~~~~~~~~~~~~~~

``-m <list>``
'''''''''''''

``--model <list>``
''''''''''''''''''

The names of the models to benchmark. A single model is recommended,
unless you are `profiling multiple LoRA adapters <docs/lora.md>`__.
(default: ``None``)

``--model-selection-strategy {round_robin, random}``
''''''''''''''''''''''''''''''''''''''''''''''''''''

When multiple models are specified, this is how a specific model is
assigned to a prompt. Round robin means that each model receives a
request in order. Random means that assignment is uniformly random
(default: ``round_robin``)

``--backend {tensorrtllm,vllm}``
''''''''''''''''''''''''''''''''

When using the “triton” service-kind, this is the backend of the model.
For the TRT-LLM backend, you currently must set
``exclude_input_in_output`` to true in the model config to not echo the
input tokens in the output. (default: tensorrtllm)

``--endpoint <str>``
''''''''''''''''''''

Set a custom endpoint that differs from the OpenAI defaults. (default:
``None``)

``--endpoint-type {chat,completions,embeddings,rankings}``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The endpoint-type to send requests to on the server. This is only used
with the ``openai`` service-kind. (default: ``None``)

``--service-kind {triton,openai}``
''''''''''''''''''''''''''''''''''

The kind of service perf_analyzer will generate load for. In order to
use ``openai``, you must specify an api via ``--endpoint-type``.
(default: ``triton``)

``--streaming``
'''''''''''''''

An option to enable the use of the streaming API. (default: ``False``)

``-u <url>``
''''''''''''

``--url <url>``
'''''''''''''''

URL of the endpoint to target for benchmarking. (default: ``None``)

Input Options
~~~~~~~~~~~~~

``-b <int>``
''''''''''''

``--batch-size <int>``
''''''''''''''''''''''

The batch size of the requests GenAI-Perf should send. This is currently
only supported with the `embeddings <docs/embeddings.md>`__,
image_retrieval, and `rankings <docs/rankings.md>`__ endpoint types.
(default: ``1``)

``--extra-inputs <str>``
''''''''''''''''''''''''

Provide additional inputs to include with every request. You can repeat
this flag for multiple inputs. Inputs should be in an input_name:value
format. Alternatively, a string representing a json formatted dict can
be provided. (default: ``None``)

``--input-dataset {openorca,cnn_dailymail}``
''''''''''''''''''''''''''''''''''''''''''''

The HuggingFace dataset to use for prompts. (default: ``openorca``)

``--input-file <path>``
'''''''''''''''''''''''

The input file containing the prompts to use for profiling. Each line
should be a JSON object with a ‘text_input’ field in JSONL format.
Example: {"text_input": "Your prompt here"}"

``--num-prompts <int>``
'''''''''''''''''''''''

The number of unique prompts to generate as stimulus. (default: ``100``)

``--output-tokens-mean <int>``
''''''''''''''''''''''''''''''

The mean number of tokens in each output. Ensure the ``--tokenizer``
value is set correctly. (default: ``-1``)

``--output-tokens-mean-deterministic``
''''''''''''''''''''''''''''''''''''''

When using ``--output-tokens-mean``, this flag can be set to improve
precision by setting the minimum number of tokens equal to the requested
number of tokens. This is currently supported with the Triton
service-kind. Note that there is still some variability in the requested
number of output tokens, but GenAi-Perf attempts its best effort with
your model to get the right number of output tokens. (default:
``False``)

``--output-tokens-stddev <int>``
''''''''''''''''''''''''''''''''

The standard deviation of the number of tokens in each output. This is
only used when ``--output-tokens-mean`` is provided. (default: ``0``)

``--random-seed <int>``
'''''''''''''''''''''''

The seed used to generate random values. (default: ``0``)

``--synthetic-input-tokens-mean <int>``
'''''''''''''''''''''''''''''''''''''''

The mean of number of tokens in the generated prompts when using
synthetic data. (default: ``550``)

``--synthetic-input-tokens-stddev <int>``
'''''''''''''''''''''''''''''''''''''''''

The standard deviation of number of tokens in the generated prompts when
using synthetic data. (default: ``0``)

Profiling Options
~~~~~~~~~~~~~~~~~

``--concurrency <int>``
'''''''''''''''''''''''

The concurrency value to benchmark. (default: ``None``)

``--measurement-interval <int>``
''''''''''''''''''''''''''''''''

``-p <int>``
''''''''''''

The time interval used for each measurement in milliseconds. Perf
Analyzer will sample a time interval specified and take measurement over
the requests completed within that time interval. (default: ``10000``)

``--request-rate <float>``
''''''''''''''''''''''''''

Sets the request rate for the load generated by PA. (default: ``None``)

``-s <float>``
''''''''''''''

``--stability-percentage <float>``
''''''''''''''''''''''''''''''''''

The allowed variation in latency measurements when determining if a
result is stable. The measurement is considered as stable if the ratio
of max / min from the recent 3 measurements is within (stability
percentage) in terms of both infer per second and latency. (default:
``999``)

Output Options
~~~~~~~~~~~~~~

``--artifact-dir``
''''''''''''''''''

The directory to store all the (output) artifacts generated by
GenAI-Perf and Perf Analyzer. (default: ``artifacts``)

``--generate-plots``
''''''''''''''''''''

An option to enable the generation of plots. (default: False)

``--profile-export-file <path>``
''''''''''''''''''''''''''''''''

The path where the perf_analyzer profile export will be generated. By
default, the profile export will be to ``profile_export.json``. The
genai-perf files will be exported to
``<profile_export_file>_genai_perf.json`` and
``<profile_export_file>_genai_perf.csv``. For example, if the profile
export file is ``profile_export.json``, the genai-perf file will be
exported to ``profile_export_genai_perf.csv``. (default:
``profile_export.json``)

Other Options
~~~~~~~~~~~~~

``--tokenizer <str>``
'''''''''''''''''''''

The HuggingFace tokenizer to use to interpret token metrics from prompts
and responses. (default: ``hf-internal-testing/llama-tokenizer``)

``-v``
''''''

``--verbose``
'''''''''''''

An option to enable verbose mode. (default: ``False``)

``--version``
'''''''''''''

An option to print the version and exit.

.. raw:: html

   <!--
   ======================
   Known Issues
   ======================
   -->

Known Issues
------------

-  GenAI-Perf can be slow to finish if a high request-rate is provided
-  Token counts may not be exact
