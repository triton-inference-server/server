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

    ./pytest.py <scenario> [--output-dir DIR] [--copy-to DIR] [pytest args...]
    # e.g. ./pytest.py ensemble
    #      ./pytest.py unary --copy-to /results -vv -k perf_metrics

Each scenario is a subdirectory with its own tests/ and conftest.py, which
serves the model the way that scenario needs; this points pytest at it.
Scenarios run one at a time: each starts its own server on the default ports
and GPU.

Everything a run produces goes to one output directory: the JUnit report
(<scenario>.report.xml), pytest's own log (<scenario>.pytest.log), the server
logs (*_server.log for Triton, *_server_launcher.log for the server's stdout,
where TensorRT-LLM logs) and the generated model repositories. It is
--output-dir, else $LLMAPI_OUTPUT_DIR, else output/<scenario>/ next to this
file; the conftests read it from LLMAPI_OUTPUT_DIR. It is emptied before the
run, so it only ever holds the latest run.

--copy-to DIR (or $LLMAPI_COPY_TO) copies the output directory to
DIR/<scenario> after the run, whether the tests passed or not; point it at a
directory mounted from the host. The copy replaces the previous one and is
given the owner of DIR, so a root container leaves no root-owned files there.

Only directories this script created are ever emptied or replaced: they hold
a marker file, and an existing non-empty directory without it is an error.

pytest.ini next to this file holds the default options (-v -s -rs, pytest log
file level), so running pytest directly in a scenario directory behaves the
same. Remaining arguments are passed to pytest after them, so they add to or
override them.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USAGE = (
    "usage: pytest.py <scenario> [--output-dir DIR] [--copy-to DIR] " "[pytest args...]"
)
OPTIONS = ("--output-dir", "--copy-to")
# Marks a directory as created by this script, and therefore safe to empty.
MARKER = ".llmapi-output"


def split_options(args):
    """Remove this script's OPTIONS (as `--opt DIR` or `--opt=DIR`) from `args`.

    Returns ({option: value}, remaining args in their original order).
    """
    values = {}
    rest = []
    i = 0
    while i < len(args):
        arg = args[i]
        name, eq, value = arg.partition("=")
        if name in OPTIONS:
            if not eq:
                if i + 1 >= len(args):
                    raise ValueError("%s needs a directory" % name)
                i += 1
                value = args[i]
            values[name] = value
        else:
            rest.append(arg)
        i += 1
    return values, rest


def check_replaceable(path):
    """Raise if `path` holds anything this script did not create."""
    if (
        os.path.isdir(path)
        and os.listdir(path)
        and not os.path.exists(os.path.join(path, MARKER))
    ):
        raise ValueError(
            "refusing to empty %s: not empty and not created by pytest.py "
            "(no %s)" % (path, MARKER)
        )


def reset_dir(path):
    """Make `path` an empty, marked directory."""
    check_replaceable(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    open(os.path.join(path, MARKER), "w").close()


def copy_results(output_dir, copy_root, scenario):
    """Replace copy_root/scenario with output_dir, owned like copy_root."""
    target = os.path.join(copy_root, scenario)
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(
        output_dir,
        target,
        symlinks=True,
        ignore=shutil.ignore_patterns(".pytest_cache"),
    )
    owner = os.stat(copy_root)
    if os.geteuid() == 0 and (owner.st_uid, owner.st_gid) != (0, 0):
        for root, dirs, files in os.walk(target):
            for name in [root] + [os.path.join(root, n) for n in dirs + files]:
                os.lchown(name, owner.st_uid, owner.st_gid)
    print("=== %s results copied to %s" % (scenario, target), flush=True)


def main(argv):
    if len(argv) < 2 or argv[1].startswith("-"):
        print(USAGE, file=sys.stderr)
        return 2

    scenario = argv[1]
    scenario_dir = os.path.join(HERE, scenario)
    if not os.path.isdir(os.path.join(scenario_dir, "tests")):
        print(
            "no such scenario (missing %s/tests/): %s" % (scenario, scenario),
            file=sys.stderr,
        )
        return 2

    try:
        options, pytest_args = split_options(argv[2:])
        output_dir = os.path.abspath(
            options.get("--output-dir")
            or os.environ.get("LLMAPI_OUTPUT_DIR")
            or os.path.join(HERE, "output", scenario)
        )
        copy_root = options.get("--copy-to") or os.environ.get("LLMAPI_COPY_TO")
        if copy_root:
            copy_root = os.path.abspath(copy_root)
            if not os.path.isdir(copy_root):
                raise ValueError("--copy-to: no such directory: %s" % copy_root)
            # Checked now rather than after a long run.
            check_replaceable(os.path.join(copy_root, scenario))
        reset_dir(output_dir)
    except ValueError as e:
        print("%s\n%s" % (e, USAGE), file=sys.stderr)
        return 2
    print("=== %s output: %s" % (scenario, output_dir), flush=True)

    env = dict(os.environ, LLMAPI_OUTPUT_DIR=output_dir)
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--junitxml=%s" % os.path.join(output_dir, "%s.report.xml" % scenario),
                "-o",
                "log_file=%s" % os.path.join(output_dir, "%s.pytest.log" % scenario),
                "-o",
                "cache_dir=%s" % os.path.join(output_dir, ".pytest_cache"),
            ]
            + pytest_args,
            cwd=scenario_dir,
            env=env,
        )
    finally:
        if copy_root:
            copy_results(output_dir, copy_root, scenario)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
