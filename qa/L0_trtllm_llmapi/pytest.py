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

"""Shared entrypoint for every L0_trtllm_llmapi scenario.

    ./pytest.py <scenario> [pytest args...]
    # e.g. ./pytest.py ensemble
    #      ./pytest.py unary -vv -k perf_metrics

Each scenario is a subdirectory with its own tests/ and conftest.py, which
serves the model the way that scenario needs; this points pytest at it and
writes its JUnit report to the scenario's output/. Scenarios run one at a
time: each starts its own server on the default ports and GPU.

Runs pytest with -s so that every server start, logged through
qa/common/action_log, is visible for passing tests too. Arguments after the
scenario are passed to pytest after the defaults, so they add to or override
them.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    if len(argv) < 2:
        print("usage: pytest.py <scenario> [pytest args...]", file=sys.stderr)
        return 2

    scenario = argv[1]
    scenario_dir = os.path.join(HERE, scenario)
    if not os.path.isdir(os.path.join(scenario_dir, "tests")):
        print(
            "no such scenario (missing %s/tests/): %s" % (scenario, scenario),
            file=sys.stderr,
        )
        return 2

    output_dir = os.path.join(scenario_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "-s",
            "-rs",
            "--junitxml=output/%s.report.xml" % scenario,
        ]
        + argv[2:],
        cwd=scenario_dir,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
