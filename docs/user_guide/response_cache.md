<!--
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Response Cache

## Overview

In this document an *inference request* is the model name, model version, and
input tensors (name, shape, datatype and tensor data) that make up a request
submitted to Triton. An inference result is the output tensors (name, shape,
datatype and tensor data) produced by an inference execution. The response cache
is used by Triton to hold inference results generated for previous executed
inference requests. Triton will maintain the response cache so that inference
requests that hit in the cache will not need to execute a model to produce
results and will instead extract their results from the cache. For some use
cases this can significantly reduce the inference request latency.

Triton accesses the response cache with a hash of the inference request that
includes the model name, model version and model inputs. If the hash is found in
the cache, the corresponding inference result is extracted from the cache and
used for the request. When this happens there is no need for Triton to execute
the model to produce the inference result. If the hash is not found in the
cache, Triton executes the model to produce the inference result, and then
records that result in the cache so that subsequent inference requests can
(re)use those results.

## Usage

In order for caching to be used on a given model, it must be enabled
on both the server-side, and in the model's
[model config](model_configuration.md#response-cache). See the following
sections below for more details.

### Enable Caching on Server-side

The response cache is enabled on the server-side by specifying a
`<cache_implementation>` and corresponding configuration when starting
the Triton server.

Through the CLI, this translates to setting
`tritonserver --cache-config <cache_implementation>,<key>=<value> ...`. For example:
```
tritonserver --cache-config local,size=1048576
```

For in-process C API applications, this translates to calling
`TRITONSERVER_SetCacheConfig(const char* cache_implementation, const char* config_json)`.

This allows users to enable/disable caching globally on server startup.

### Enable Caching for a Model

**By default, no model uses response caching even if the response cache
is enabled globally with the `--cache-config` flag.**

For a given model to use response caching, the model must also have
response caching enabled in its model configuration:
```
# config.pbtxt

response_cache {
  enable: true
}
```

This allows users to enable/disable caching for specific models.

For more information on enabling the response cache for each model, see the
[model configuration docs](model_configuration.md#response-cache).

### Cache Implementations

Starting in the 23.03 release, Triton has a set of
[TRITONCACHE APIs](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritoncache.h)
that are used to communicate with a cache implementation of the user's choice.

A cache implementation is a shared library that implements the required
TRITONCACHE APIs and is dynamically loaded on server startup, if enabled.

Triton's most recent
[tritonserver release containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
come with the following cache implementations out of the box:
- [local](https://github.com/triton-inference-server/local_cache): `/opt/tritonserver/caches/local/libtritoncache_local.so`
- [redis](https://github.com/triton-inference-server/redis_cache): `/opt/tritonserver/caches/redis/libtritoncache_redis.so`

With these TRITONCACHE APIs, `tritonserver` exposes a new `--cache-config`
CLI flag that gives the user flexible customization of which cache implementation
to use, and how to configure it. Similar to the `--backend-config` flag,
the expected format is `--cache-config <cache_name>,<key>=<value>` and may
be specified multiple times to specify multiple keys if the cache implementation
requires it.

#### Local Cache

The `local` cache implementation is equivalent to the response cache used
internally before the 23.03 release. For more implementation specific details,
see the
[local cache implementation](https://github.com/triton-inference-server/local_cache).

When `--cache-config local,size=SIZE` is specified with a non-zero `SIZE`,
Triton allocates the requested size in CPU memory and **shares the
cache across all inference requests and across all models**.

#### Redis Cache

The `redis` cache implementation exposes the ability for Triton to communicate
with a Redis server for caching. The `redis_cache` implementation is essentially
a Redis client that acts as an intermediary between Triton and Redis.

To list a few benefits of the `redis` cache compared to the `local` cache in
the context of Triton:
- The Redis server can be hosted remotely as long as it is accessible by Triton,
  so it is not tied directly to the Triton process lifetime.
  - This means Triton can be restarted and still have access to previously cached entries.
  - This also means that Triton doesn't have to compete with the cache for memory/resource usage.
- Multiple Triton instances can share a cache by configuring each Triton instance
  to communicate with the same Redis server.
- The Redis server can be updated/restarted independently of Triton, and
  Triton will fallback to operating as it would with no cache access during
  any Redis server downtime, and log appropriate errors.

In general, the Redis server can be configured/deployed as needed for your use
case, and Triton's `redis` cache will simply act as a client of your Redis
deployment. The [Redis docs](https://redis.io/docs/) should be consulted for
questions and details about configuring the Redis server.

For Triton-specific `redis` cache implementation details/configuration, see the
[redis cache implementation](https://github.com/triton-inference-server/redis_cache).

#### Custom Cache

With the TRITONCACHE API interface, it is now possible for
users to implement their own cache to suit any use-case specific needs.
To see the required interface that must be implemented by a cache
developer, see the
[TRITONCACHE API header](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritoncache.h).
The `local` or `redis` cache implementations may be used as reference.

Upon successfully developing and building a custom cache, the resulting shared
library (ex: `libtritoncache_<name>.so`) must be placed in the cache directory
similar to where the `local` and `redis` cache implementations live. By default,
this directory is `/opt/tritonserver/caches`, but a custom directory may be
specified with `--cache-dir` as needed.

To put this example together, if the custom cache were named "custom"
(this name is arbitrary), by default Triton would expect to find the
cache implementation at `/opt/tritonserver/caches/custom/libtritoncache_custom.so`.

## Deprecation Notes

> **Note**
> Prior to 23.03, enabling the `local` cache used to be done through setting a non-zero size
> (in bytes) when Triton was launched using the `--response-cache-byte-size` flag.
>
> Starting in 23.03, the `--response-cache-byte-size` flag is now deprecated and
> `--cache-config` should be used instead. For backwards compatibility,
> `--response-cache-byte-size` will continue to function under the hood by being
> converted to the corresponding `--cache-config` argument, but it will default
> to using the `local` cache implementation. It is not possible to choose other
> cache implementations using the `--response-cache-byte-size` flag.
>
> For example, `--response-cache-byte-size 1048576`
> would be equivalent to `--cache-config local,size=1048576`. However, the
> `--cache-config` flag is much more flexible and should be used instead.

> **Warning**
>
> The `local` cache implementation may fail to initialize for very small values
> of `--cache-config local,size=<small_value>` or `--response-cache-byte-size`
> (ex: less than 1024 bytes) due to internal memory management requirements.
> If you encounter an initialization error for a relatively small cache size,
> try increasing it.
>
> Similarly, the size is upper bounded by the available RAM on the system.
> If you encounter an initial allocation error for a very large cache size
> setting, try decreasing it.

## Performance

The response cache is intended to be used for use cases where a significant
number of duplicate requests (cache hits) are expected and therefore would
benefit from caching. The term "significant" here is subjective to the use
case, but a simple interpretation would be to consider the proportion of
expected cache hits/misses, as well as the average time spend computing
a response.

For cases where cache hits are common and computation is expensive,
the cache can significantly improve overall performance.

For cases where most requests are unique (cache misses) or the compute is
fast/cheap (the model is not compute-bound), the cache can negatively impact
the overall performance due to the overhead of managing and communicating with
the cache.

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
- The response cache does not currently support
  [decoupled models](decoupled_models.md).
- Top-level requests to ensemble models do not currently support response
  caching. However, composing models within an ensemble may have their
  responses cached if supported and enabled by that composing model.

