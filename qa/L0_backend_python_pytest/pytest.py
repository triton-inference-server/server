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
"""Shared entrypoint for every L0_backend_python_pytest scenario.

    ./pytest.py <scenario>   # e.g. ./pytest.py bls

One driver, one image (see Dockerfile): each scenario is a subdirectory
with its own tests/ and conftest.py; this just points pytest at the right
one and writes its JUnit report to that scenario's output/.

Named pytest.py, not test.sh: the CI job builds a derived image from this
directory (see Dockerfile) and runs it with the scenario name as the
container command, so there is no bash indirection to satisfy here -- this
*is* the entrypoint, invoked as `python3 pytest.py <scenario>`.

Runs pytest with -s (capturing off): every scenario's conftest.py prints
each subprocess it runs via qa/common/action_log before running it, and
without -s that only shows up in a failing test's "Captured stdout"
section -- it should be visible for every test, pass or fail, the same
way `bash -ex` made every command in the old test.sh scripts visible.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    if len(argv) != 2:
        print("usage: pytest.py <scenario>", file=sys.stderr)
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
            "--junitxml=output/%s.report.xml" % scenario,
        ],
        cwd=scenario_dir,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
