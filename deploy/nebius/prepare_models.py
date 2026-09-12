#!/usr/bin/env python3
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Prepare the fixed DenseNet repository; no dependencies beyond Python 3.10+."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
GPU_CONFIG = (
    "\n# This example serves only version 1 on one GPU.\n"
    "instance_group [{ kind: KIND_GPU count: 1 gpus: [0] }]\n"
    "version_policy: { specific: { versions: [1] } }\n"
    "# Disable reduced-precision TF32 for the FP32 numerical reference check.\n"
    "optimization { execution_accelerators { gpu_execution_accelerator [ {\n"
    '  name: "cuda" parameters { key: "use_tf32" value: "0" }\n'
    "} ] } }\n"
)
CONFIG_PATH = "model_repository/densenet_onnx/config.pbtxt"


def destination(root, relative):
    path = PurePosixPath(relative)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or path.parts[0] not in {"model_repository", "notices"}
    ):
        raise ValueError("Unsafe artifact path in pins.json")
    return root.joinpath(*path.parts)


def fetch(artifact, target):
    """Enforce the byte count as well as the digest, including on partial reads."""
    expected_size = artifact["size_bytes"]
    digest = hashlib.sha256()
    received = 0
    deadline = time.monotonic() + 180
    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "Triton"})
    with urllib.request.urlopen(request, timeout=30) as response, target.open(
        "xb"
    ) as f:
        if not response.geturl().startswith("https://"):
            raise ValueError("Artifact download did not use HTTPS")
        while True:
            if time.monotonic() >= deadline:
                raise ValueError("Artifact download exceeded its deadline")
            chunk = response.read(min(65536, expected_size - received + 1))
            if not chunk:
                break
            received += len(chunk)
            if received > expected_size:
                raise ValueError("Artifact exceeds the pinned size")
            digest.update(chunk)
            f.write(chunk)
    if received != expected_size or digest.hexdigest() != artifact["sha256"]:
        raise ValueError("Artifact size or SHA-256 mismatch; refusing the download")


def prepare(output, pins, download=fetch):
    """Publish a complete build context only after all artifacts are verified."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nebius-models-", dir=output.parent))
    try:
        for artifact in pins["artifacts"]:
            target = destination(staging, artifact["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            download(artifact, target)
        config = staging / CONFIG_PATH
        config.write_text(
            config.read_text(encoding="utf-8") + GPU_CONFIG, encoding="utf-8"
        )
        manifest = []
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                manifest.append(f"{digest}  {file.relative_to(staging).as_posix()}\n")
        (staging / "SHA256SUMS").write_text("".join(manifest), encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(
                "Output was created during preparation; refusing to replace it"
            )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "build")
    args = parser.parse_args()
    try:
        pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
        output = prepare(args.output, pins)
    except (OSError, ValueError, urllib.error.URLError):
        print(
            "Preparation failed; check network, output path and pinned artifacts.",
            file=sys.stderr,
        )
        return 1
    print(f"Prepared {output}; checksums are in SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
