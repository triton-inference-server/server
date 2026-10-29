# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import tempfile
from pathlib import Path

import yaml

CASES = [
    (
        'namespace="default",pod=~"triton-a"',
        [
            ("default", "triton-a", "busy", "0+1000x4", "0+1x4"),
            ("default", "triton-a", "idle", "0+0x4", "0+0x4"),
        ],
        [{"labels": '{pod="triton-a"}', "value": 1000}],
    ),
    (
        'namespace="default",pod=~"triton-a"',
        [
            ("default", "triton-a", "low", "0+100x4", "0+1x4"),
            ("default", "triton-a", "high", "0+800x4", "0+2x4"),
        ],
        [{"labels": '{pod="triton-a"}', "value": 300}],
    ),
    (
        'namespace="default",pod=~"triton-a"',
        [("default", "triton-a", "idle", "0+0x4", "0+0x4")],
        [{"labels": '{pod="triton-a"}', "value": 0}],
    ),
    (
        'namespace="default",pod=~"triton-a"',
        [("default", "triton-a", "single", "0+0x3 1000", "0+0x3 1")],
        [{"labels": '{pod="triton-a"}', "value": 1000}],
    ),
    (
        'namespace="default",pod=~"triton-a"',
        [
            ("default", "triton-a", "reset", "0+100x3 100", "0+1x3 1"),
            ("default", "triton-a", "steady", "0+500x4", "0+1x4"),
        ],
        [{"labels": '{pod="triton-a"}', "value": 300}],
    ),
    (
        'namespace="default",pod=~"triton-a|triton-b"',
        [
            ("default", "triton-a", "model", "0+1000x4", "0+1x4"),
            ("default", "triton-b", "model", "0+100x4", "0+1x4"),
        ],
        [
            {"labels": '{pod="triton-a"}', "value": 1000},
            {"labels": '{pod="triton-b"}', "value": 100},
        ],
    ),
    (
        'namespace="default",pod=~"triton-a"',
        [
            ("default", "triton-a", "default-model", "0+1000x4", "0+1x4"),
            ("other", "triton-a", "other-model", "0+100000x4", "0+1x4"),
        ],
        [{"labels": '{pod="triton-a"}', "value": 1000}],
    ),
]


def promql_test(query, label_matchers, series, exp_samples):
    return {
        "interval": "15s",
        "input_series": [
            {
                "series": (
                    f'nv_inference_{metric}{{namespace="{namespace}",'
                    f'pod="{pod}",model="{model}"}}'
                ),
                "values": values,
            }
            for namespace, pod, model, queue, success in series
            for metric, values in (
                ("queue_duration_us", queue),
                ("request_success", success),
            )
        ],
        "promql_expr_test": [
            {
                "expr": query.replace("<<.LabelMatchers>>", label_matchers).replace(
                    "<<.GroupBy>>", "pod"
                ),
                "eval_time": "1m",
                "exp_samples": exp_samples,
            }
        ],
    }


def main():
    chart = Path(__file__).with_name("k8s-onprem")
    values = yaml.safe_load((chart / "values.yaml").read_text(encoding="utf-8"))
    query = values["prometheus-adapter"]["rules"]["custom"][0]["metricsQuery"]
    spec = {
        "tests": [promql_test(query, *case) for case in CASES],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "queue-query.yml"
        test_file.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        subprocess.run(["promtool", "test", "rules", str(test_file)], check=True)


if __name__ == "__main__":
    main()
