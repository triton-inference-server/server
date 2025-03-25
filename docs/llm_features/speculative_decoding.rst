..
.. Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


About Speculative Decoding
=========================
Speculative Decoding (also referred to as Speculative Sampling) is a set of techniques designed
to allow generation of more than one token per forward pass iteration. This can lead to a reduction
in the average per-token latency in situations where the GPU is underutilized due to small batch sizes.

Speculative decoding involves predicting a sequence of future tokens, referred to as draft tokens,
using a method that is substantially more efficient than repeatedly executing the target Large Language
Model (LLM). These draft tokens are then collectively validated by processing them through the target LLM
in a single forward pass. The underlying assumptions are twofold:

1. processing multiple draft tokens concurrently will be as rapid as processing a single token
2. multiple draft tokens will be validated successfully over the course of the full generation

If the first assumption holds true, the latency of speculative decoding will no worse than the standard
approach. If the second holds, output token generation advances by statistically more than one token per
forward pass. The combination of both these allows speculative decoding to result in reduced latency.

Performance Improvements
========================
It's important to note that the effectiveness of speculative decoding techniques is highly dependent
on the specific task at hand. For instance, forecasting subsequent tokens in a code-completion scenario
may prove simpler than generating a summary for an article. `Spec-Bench <https://sites.google.com/view/spec-bench>`__
shows the performance of different speculative decoding approaches on different tasks.