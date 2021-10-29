<!--
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Triton Response Cache (beta)

**This feature is currently in beta and may be subject to change.**

In this document an *inference request* is the model name, model version, and
input tensors (name, shape, datatype and tensor data) that make up a request
submitted to Triton. An inference result is the output tensors (name, shape,
datatype and tensor data) produced by an inference execution. The response cache
is used by Triton to hold inference results generated for previous executed
inference requests. Triton will maintain the response cache so that inference
requests that hit in the cache will not need to execute a model to produce
results and will instead extract their results from the cache. For some use
cases this can significantly reduce the inference request latency.

The response cache is enabled by setting a non-zero size when Triton is launched
using the `--response-cache-byte-size` flag. The flag defaults to 0 (zero). When
non-zero, Triton allocates the requested size in CPU memory and **shares the
cache across all inference requests and across all models**. For a given model
to use response caching, the model must enable response caching in the model
configuration. **By default, no model uses response caching even if the response
cache is enabled with the `--response-cache-byte-size` flag.** For more
information on enabling the response cache for each model, see the [model
configuration
docs](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#response-cache).

Triton accesses the response cache with a hash of the inference request that
includes the model name, model version and model inputs. If the hash is found in
the cache, the corresponding inference result is extracted from the cache and
used for the request. When this happens there is no need for Triton to execute
the model to produce the inference result. If the hash is not found in the
cache, Triton executes the model to produce the inference result, and then
records that result in the cache so that subsequent inference requests can
(re)use those results. 

The response cache is a fixed-size resource, as a result it must be managed by a
replacement policy when the number of cacheable responses exceeds the capacity
of the cache. Currently, the cache only implements a least-recently-used
([LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)))
replacement policy which will automatically evict one or more LRU entries to
make room for new entries.

## Known Limitations

- Only input tensors located in CPU memory will be hashable for accessing the
  cache. If an inference request contains input tensors not in CPU memory, the
  request will not be hashed and therefore the response will not be cached.
- Only responses with all output tensors located in CPU memory will be eligible
  for caching. If any output tensor in a response is not located in CPU memory,
  the response will not be cached.
- The cache is accessed using only the inference request hash. As a result, if
  two different inference requests generate the same hash (a hash collision),
  then Triton may incorrectly use the cached result for an inference request.
  The hash is a 64-bit value so the likelihood of collision is small.
- Only successful inference requests will have their responses cached. If a
  request fails or returns an error during inference, its response will not be
  cached.
- Only requests going through the Default Scheduler or Dynamic Batch Scheduler
  are eligible for caching. The Sequence Batcher does not currently support
  response caching.
