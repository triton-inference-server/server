#!/usr/bin/env python3
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig
from distutils.dir_util import copy_tree
from tempfile import mkstemp


def fail_if(p, msg):
    if p:
        print("error: {}".format(msg), file=sys.stderr)
        sys.exit(1)


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def touch(path):
    pathlib.Path(path).touch()


def cpdir(src, dest):
    copy_tree(src, dest, preserve_symlinks=1)


def sed(pattern, replace, source, dest=None):
    name = None
    if dest:
        name = dest
    if dest is None:
        fd, name = mkstemp()

    with open(source, "r") as fin, open(name, "w") as fout:
        for line in fin:
            out = re.sub(pattern, replace, line)
            fout.write(out)

    if not dest:
        shutil.copyfile(name, source)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dest-dir", type=str, required=True, help="Destination directory."
    )
    parser.add_argument(
        "--binding-path",
        type=str,
        required=True,
        help="Path to Triton Frontend Python binding.",
    )

    FLAGS = parser.parse_args()

    FLAGS.triton_version = None
    with open("TRITON_VERSION", "r") as vfile:
        FLAGS.triton_version = vfile.readline().strip()

    FLAGS.whl_dir = os.path.join(FLAGS.dest_dir, "wheel")

    print("=== Building in: {}".format(os.getcwd()))
    print("=== Using builddir: {}".format(FLAGS.whl_dir))
    print("Adding package files")
    mkdir(os.path.join(FLAGS.whl_dir, "tritonfrontend"))
    shutil.copy(
        "tritonfrontend/__init__.py", os.path.join(FLAGS.whl_dir, "tritonfrontend")
    )
    # Type checking marker file indicating support for type checkers.
    # https://peps.python.org/pep-0561/
    shutil.copy(
        "tritonfrontend/py.typed", os.path.join(FLAGS.whl_dir, "tritonfrontend")
    )
    cpdir("tritonfrontend/_c", os.path.join(FLAGS.whl_dir, "tritonfrontend", "_c"))
    cpdir("tritonfrontend/_api", os.path.join(FLAGS.whl_dir, "tritonfrontend", "_api"))
    PYBIND_LIB = os.path.basename(FLAGS.binding_path)
    shutil.copyfile(
        FLAGS.binding_path,
        os.path.join(FLAGS.whl_dir, "tritonfrontend", "_c", PYBIND_LIB),
    )

    shutil.copyfile("LICENSE.txt", os.path.join(FLAGS.whl_dir, "LICENSE.txt"))
    shutil.copyfile("setup.py", os.path.join(FLAGS.whl_dir, "setup.py"))

    os.chdir(FLAGS.whl_dir)
    # The wheel ships an arch-specific CPython extension
    # (tritonfrontend/_c/<pybind>.so). Pass --plat-name so the wheel is
    # tagged with the current platform (e.g. linux_x86_64 / linux_aarch64)
    # instead of the misleading "none-any".
    plat_name = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    print("=== Building wheel")
    args = ["python3", "setup.py", "bdist_wheel", "--plat-name", plat_name]
    # PEP 427 "build tag": an optional numeric segment between version
    # and python-tag that lets two wheels of the same version coexist
    # (e.g. reruns of the same CI pipeline). Preferred source is
    # CI_PIPELINE_ID (GitLab) with a BUILD_NUMBER fallback — both are
    # guaranteed to start with a digit as required by PEP 427.
    build_number = os.environ.get("CI_PIPELINE_ID") or os.environ.get("BUILD_NUMBER")
    if build_number:
        args += ["--build-number", build_number]

    wenv = os.environ.copy()
    wenv["VERSION"] = FLAGS.triton_version
    wenv["TRITON_PYBIND"] = PYBIND_LIB
    p = subprocess.Popen(args, env=wenv)
    p.wait()
    fail_if(p.returncode != 0, "setup.py failed")

    # Post-process with auditwheel so the wheel is tagged with a proper
    # manylinux_2_X_<arch> platform (required by canonical PyPI). When
    # auditwheel is unavailable in the build image we keep the
    # linux_<arch> wheel and emit a warning; the Poetry/pip lock-file
    # problem is already solved by the distinct filename, and the tag can
    # be fixed up in a follow-up publish step if needed.
    _repair_wheel_with_auditwheel(FLAGS.whl_dir, FLAGS.dest_dir)

    print(f"=== Output wheel file is in: {FLAGS.dest_dir}")
    touch(os.path.join(FLAGS.dest_dir, "stamp.whl"))


def _repair_wheel_with_auditwheel(whl_dir, dest_dir):
    """Upgrade a linux_<arch> wheel to manylinux_2_X_<arch>.

    Ports the pattern established for tritonclient in TRI-286:
      1. auditwheel repair   — auto-discovers the minimum manylinux tag
         by inspecting glibc symbol requirements of the embedded .so.
      2. python -m wheel tags fallback — used when auditwheel reports
         "no ELF" (the wheel has no native extension, e.g. a downstream
         build disabled bindings). Mirrors the documented fallback.
      3. No-op with warning — when auditwheel is not installed in the
         build image, keep the linux_<arch> wheel as-is so the build
         does not regress.
    """
    if shutil.which("auditwheel") is None:
        print(
            "=== WARNING: auditwheel not found on PATH; keeping linux_<arch> "
            "wheel as-is. Install auditwheel in the build image to produce "
            "PyPI-acceptable manylinux_2_X_<arch> wheels.",
            file=sys.stderr,
        )
        cpdir("dist", dest_dir)
        return

    dist_dir = os.path.join(whl_dir, "dist")
    wheels = [
        os.path.join(dist_dir, w) for w in os.listdir(dist_dir) if w.endswith(".whl")
    ]
    fail_if(not wheels, "no wheel produced by setup.py")

    for wheel_path in wheels:
        print(f"=== Running auditwheel repair on {wheel_path}")
        r = subprocess.run(
            ["auditwheel", "repair", wheel_path, "--wheel-dir", dest_dir],
            capture_output=True,
            text=True,
        )
        # `auditwheel` logs via Python's logging module, which writes to
        # stderr — the "no ELF" sentinel only appears there, not in
        # stdout. See TRI-286 root-cause write-up.
        if r.returncode != 0 and "no ELF" in r.stderr:
            arch = os.uname().machine
            manylinux_tag = f"manylinux_2_28_{arch}"
            print(
                f"=== Pure-Python wheel detected; falling back to wheel tags "
                f"({manylinux_tag})"
            )
            copied = os.path.join(dest_dir, os.path.basename(wheel_path))
            shutil.copy(wheel_path, copied)
            # `wheel tags --remove` replaces the linux_<arch> wheel in
            # dest_dir with the correctly-tagged manylinux one.
            r2 = subprocess.run(
                [
                    "python3",
                    "-m",
                    "wheel",
                    "tags",
                    "--platform-tag",
                    manylinux_tag,
                    "--remove",
                    copied,
                ]
            )
            fail_if(r2.returncode != 0, "wheel tags fallback failed")
        elif r.returncode != 0:
            sys.stderr.write(r.stderr)
            fail_if(True, "auditwheel repair failed")


if __name__ == "__main__":
    main()
