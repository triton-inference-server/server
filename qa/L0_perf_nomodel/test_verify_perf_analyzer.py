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

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPerfAnalyzerArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clients = self.root / "clients"
        self.clients.mkdir()
        verifier = Path(__file__).resolve().parents[2] / "tools/verify_perf_analyzer.py"
        shutil.copy(verifier, self.clients)
        self.payload = b"#!/bin/sh\necho 'Perf Analyzer test build'\n"
        self.native = self.clients / "perf_analyzer"
        self.native.write_bytes(self.payload)
        self.native.chmod(0o755)
        package = self.root / "packages/perf_analyzer"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "bin").mkdir()
        self.packaged = package / "bin/perf_analyzer"
        shutil.copy(self.native, self.packaged)
        self.metadata = {
            "commit": "a" * 40,
            "source_ref": "main",
            "build_mode": "1",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        for suffix, value in self.metadata.items():
            self.write_metadata(suffix, value)
        self.env = dict(
            os.environ,
            PYTHONPATH=str(package.parent),
            PATH=str(self.clients) + os.pathsep + os.environ["PATH"],
        )

    def write_metadata(self, suffix, value):
        (self.clients / ("perf_analyzer." + suffix)).write_text(value + "\n")

    def verify(self):
        return subprocess.run(
            [sys.executable, str(self.clients / "verify_perf_analyzer.py"),
             "--require-main"],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_source_main_package_matches(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_provenance_is_rejected(self):
        for suffix, value in [
            ("source_ref", "r26.10"), ("build_mode", "0"),
            ("commit", "unknown"), ("sha256", "0" * 64),
        ]:
            with self.subTest(suffix=suffix):
                self.write_metadata(suffix, value)
                self.assertNotEqual(self.verify().returncode, 0)
                self.write_metadata(suffix, self.metadata[suffix])

    def test_stale_package_is_rejected(self):
        self.packaged.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.assertNotEqual(self.verify().returncode, 0)

    def test_missing_native_is_rejected(self):
        self.native.unlink()
        self.assertNotEqual(self.verify().returncode, 0)

    def test_shadowing_executable_is_rejected(self):
        shadow = self.root / "shadow"
        shadow.mkdir()
        shutil.copy(self.native, shadow)
        self.env["PATH"] = str(shadow) + os.pathsep + self.env["PATH"]
        self.assertNotEqual(self.verify().returncode, 0)


if __name__ == "__main__":
    unittest.main()
