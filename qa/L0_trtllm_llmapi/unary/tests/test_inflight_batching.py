# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Concurrent requests to one model instance must be generated together.

The LLM API model does not use Triton's request batching (max_batch_size 0,
JIRA-4496); concurrency comes from the in-flight batching of the TensorRT-LLM
executor behind its asynchronous execute(). If execute() serialized requests,
each would be scheduled only after the previous one finished.
"""

import concurrent.futures
import uuid

import llmapi_test_util as util

CONCURRENCY = 16
MAX_TOKENS = 64


def _timestamps(index):
    # Distinct prompts, so that no request is served from another's KV cache.
    # They must not start with the UUID: TensorRT-LLM reuses partially
    # matching KV-cache blocks by default, and test_perf_metrics_outputs, which
    # runs in the same server, needs its UUID-led prompt to reuse nothing.
    result = util.grpc_infer(
        f"Request {index} ({uuid.uuid4()}): tell a long story about a lighthouse.",
        sampling_param_max_tokens=MAX_TOKENS,
        sampling_param_temperature=0.0,
        sampling_param_return_perf_metrics=True,
    )
    values = []
    for name in ("first_scheduled_time_ns", "last_token_time_ns"):
        # Returned as strings, like the other perf metrics (see test_unary.py).
        value = result.as_numpy(name).flatten()[0]
        values.append(int(value.decode() if isinstance(value, bytes) else value))
    return tuple(values)


def test_concurrent_requests_are_generated_together():
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        spans = list(pool.map(_timestamps, range(CONCURRENCY)))
    latest_start = max(start for start, _ in spans)
    earliest_end = min(end for _, end in spans)
    # Every request was scheduled before any of them finished: at that moment
    # all of them were in the executor's batch.
    assert latest_start < earliest_end, (
        "requests were not generated concurrently: the last one was scheduled "
        f"{(latest_start - earliest_end) / 1e6:.1f} ms after the first finished"
    )
