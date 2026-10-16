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

import numpy as np
import triton_python_backend_utils as pb_utils

# The TensorRT-LLM C++ backend used to publish these gauges, and the ORCA
# endpoint-load-metrics header is computed from them (src/orca_http.cc):
#   kv_cache_utilization = used / max
#   max_token_capacity   = max * tokens_per
# The LLM API backend does not publish them, so this model stands in as the
# source, with fixed values, to keep the server-side logic tested.
KV_CACHE_BLOCK_METRICS = {"tokens_per": 32, "used": 25, "max": 100}


class TritonPythonModel:
    def initialize(self, args):
        # Keep the family: its metrics are unregistered when it is collected.
        self.family = pb_utils.MetricFamily(
            name="nv_trt_llm_kv_cache_block_metrics",
            description="KV cache block counts, by kv_cache_block_type",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.metrics = []
        for block_type, value in KV_CACHE_BLOCK_METRICS.items():
            metric = self.family.Metric(labels={"kv_cache_block_type": block_type})
            metric.set(float(value))
            self.metrics.append(metric)

    def execute(self, requests):
        responses = []
        for request in requests:
            text_output = pb_utils.Tensor(
                "text_output", np.array([b"mock output"], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[text_output]))
        return responses
