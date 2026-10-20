#!/usr/bin/env python3
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

"""Send identity requests and check lock-free latency histograms.

The server must already be running with summary_latencies=simplified and the
identity_latency model loaded.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

DURATION_HISTOGRAMS = [
    "nv_inference_request_histogram_us",
    "nv_inference_queue_histogram_us",
    "nv_inference_compute_input_histogram_us",
    "nv_inference_compute_infer_histogram_us",
    "nv_inference_compute_output_histogram_us",
]
SUMMARY_NAMES = [
    "nv_inference_request_summary",
    "nv_inference_queue_summary",
    "nv_inference_compute_input_summary",
    "nv_inference_compute_infer_summary",
    "nv_inference_compute_output_summary",
]


def sample(metrics, name, model):
    match = re.search(
        rf"^{re.escape(name)}\{{[^}}]*model=\"{re.escape(model)}\"[^}}]*\}} (\d+(?:\.\d+)?)\s*$",
        metrics,
        re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing sample {name} for model {model}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8002/metrics")
    parser.add_argument("--model", default="identity")
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()

    body = json.dumps(
        {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "FP32",
                    "shape": [1, 16],
                    "data": [0.0] * 16,
                }
            ]
        }
    ).encode("utf-8")
    url = f"{args.http}/v2/models/{args.model}/infer"
    for i in range(args.requests):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise AssertionError(f"request {i} status {response.status}")
                response.read()
        except urllib.error.URLError as exc:
            raise AssertionError(f"request {i} failed: {exc}") from exc

    with urllib.request.urlopen(args.metrics, timeout=10) as response:
        metrics = response.read().decode("utf-8")

    for name in SUMMARY_NAMES:
        if name in metrics:
            raise AssertionError(f"summary metric still published: {name}")
    if 'quantile="' in metrics:
        raise AssertionError("summary quantile label is still published")

    request_count = sample(
        metrics, "nv_inference_request_histogram_us_count", args.model
    )
    inf_match = re.search(
        rf"nv_inference_request_histogram_us_bucket\{{[^}}]*model=\"{re.escape(args.model)}\"[^}}]*le=\"\+Inf\"[^}}]*\}} (\d+)",
        metrics,
    )
    if inf_match is None:
        raise AssertionError("missing +Inf request histogram bucket")
    request_inf = float(inf_match.group(1))
    if request_count < args.requests:
        raise AssertionError(
            f"request histogram count {request_count} < requests {args.requests}"
        )
    if request_inf != request_count:
        raise AssertionError(
            f"+Inf bucket {request_inf} != count {request_count}"
        )

    for name in DURATION_HISTOGRAMS:
        count = sample(metrics, name + "_count", args.model)
        if count < args.requests:
            raise AssertionError(f"{name}_count {count} < {args.requests}")
        if name + "_sum" not in metrics or name + "_bucket" not in metrics:
            raise AssertionError(f"{name} is missing _sum or _bucket")

    print(
        f"e2e ok requests={args.requests} histogram_count={int(request_count)} +Inf={int(request_inf)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
