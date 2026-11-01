#!/usr/bin/env python3
# Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-main", action="store_true")
    args = parser.parse_args()
    artifact_dir = Path(__file__).resolve().parent
    native = artifact_dir / "perf_analyzer"
    source_ref = (artifact_dir / "perf_analyzer.source_ref").read_text().strip()
    commit = (artifact_dir / "perf_analyzer.commit").read_text().strip()
    mode = (artifact_dir / "perf_analyzer.build_mode").read_text().strip()
    expected = (artifact_dir / "perf_analyzer.sha256").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise RuntimeError("Missing or invalid Perf Analyzer source commit")
    if mode not in ("0", "1"):
        raise RuntimeError("Invalid Perf Analyzer build mode")
    if args.require_main and (source_ref != "main" or mode != "1"):
        raise RuntimeError("This benchmark requires Perf Analyzer built from main")
    if not os.access(native, os.X_OK) or digest(native) != expected:
        raise RuntimeError("Native Perf Analyzer does not match its build checksum")
    spec = importlib.util.find_spec("perf_analyzer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Perf Analyzer Python package is not installed")
    packaged = Path(next(iter(spec.submodule_search_locations))) / "bin/perf_analyzer"
    if not os.access(packaged, os.X_OK) or digest(packaged) != expected:
        raise RuntimeError("Installed PA package differs from the SDK build artifact")
    selected = shutil.which("perf_analyzer")
    if selected is None or Path(selected).resolve() != native.resolve():
        raise RuntimeError(f"PATH selects an unexpected Perf Analyzer: {selected}")
    print(f"Perf Analyzer source: {source_ref}, commit: {commit}, build mode: {mode}")
    print(f"Native binary: {native}; installed package binary: {packaged}")
    print(f"Perf Analyzer SHA256: {expected}", flush=True)
    # Also check runtime library resolution in the final image.
    subprocess.run([str(native), "--version"], check=True)
    subprocess.run([str(packaged), "--version"], check=True)


if __name__ == "__main__":
    main()
