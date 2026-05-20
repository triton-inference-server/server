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

"""
Build a malicious EXECUTION_ENV_PATH tarball for the Python backend path-traversal regression test.

    --mode relative  -> entry '../../<name>' (ARCHIVE_EXTRACT_SECURE_NODOTDOT)
    --mode absolute  -> entry '/tmp/<name>'  (ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS)
"""

import argparse
import io
import os
import sys
import tarfile
import time


def build_archive(mode: str, output: str, marker: str) -> None:
    if mode == "relative":
        entry_name = "../../" + marker[len("/tmp/"):]
    else:
        entry_name = marker

    def add(tar: tarfile.TarFile, name: str, body: bytes, mode: int = 0o644):
        info = tarfile.TarInfo(name=name)
        info.size, info.mode, info.mtime = len(body), mode, int(time.time())
        tar.addfile(info, io.BytesIO(body))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        add(tar, entry_name, b"traversal-test\n")
        # Stand-in for a conda-pack activate script so an unpatched server
        # accepts the model end-to-end rather than failing on the missing
        # entry-point.
        add(tar, "bin/activate", b"#!/bin/bash\n", mode=0o755)

    print(f"[zipslip] wrote {output} (mode={mode}, entry={entry_name!r})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["relative", "absolute"])
    ap.add_argument("--output", required=True,
                    help="path to write the malicious .tar.gz")
    ap.add_argument("--marker", required=True,
                    help="absolute path under /tmp/ where the traversal "
                         "would land on a vulnerable server")
    args = ap.parse_args()

    try:
        build_archive(args.mode, args.output, args.marker)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
