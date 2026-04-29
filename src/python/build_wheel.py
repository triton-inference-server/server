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
    shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True)


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


def _detect_cuda_version():
    """Detect the CUDA toolkit version visible to the build.

    Prefers the CUDA_VERSION env var (set by official NVIDIA base
    images); falls back to parsing /usr/local/cuda/version.json which
    is the canonical location for the installed toolkit. Returns the
    raw string (e.g. "13.2.1") or None when CUDA is not available.
    """
    v = os.environ.get("CUDA_VERSION")
    if v:
        return v
    try:
        import json as _json

        with open("/usr/local/cuda/version.json") as f:
            data = _json.load(f)
        return data.get("cuda", {}).get("version")
    except (OSError, ValueError, KeyError):
        return None


def _compose_version(base_version):
    """Compose the full wheel version string.

    Appends a PEP 440 local-version segment describing the NVIDIA
    container release and CUDA toolkit so consumers can tell an
    nv26.04 wheel from an nv26.05 wheel and a cu132 wheel from a
    cu128 wheel. All sources are optional; local non-CI builds return
    the version unchanged.
    """
    nv = (
        os.environ.get("NVIDIA_UPSTREAM_VERSION")
        or os.environ.get("NVIDIA_TRITON_SERVER_VERSION")
        or os.environ.get("TRITON_CONTAINER_VERSION")
    )
    cuda = _detect_cuda_version()
    print(
        f"=== Wheel local-version inputs: "
        f"NVIDIA_UPSTREAM_VERSION={os.environ.get('NVIDIA_UPSTREAM_VERSION')!r} "
        f"NVIDIA_TRITON_SERVER_VERSION={os.environ.get('NVIDIA_TRITON_SERVER_VERSION')!r} "
        f"TRITON_CONTAINER_VERSION={os.environ.get('TRITON_CONTAINER_VERSION')!r} "
        f"-> nv={nv!r}, cuda={cuda!r}",
        file=sys.stderr,
    )
    local = []
    if nv:
        local.append(f"nv{nv}")
    if cuda:
        parts = cuda.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            local.append(f"cu{parts[0]}{parts[1]}")
    if local:
        return f"{base_version}+{'.'.join(local)}"
    return base_version


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
        shutil.copytree(os.path.join(whl_dir, "dist"), dest_dir, dirs_exist_ok=True)
        return

    dist_dir = os.path.join(whl_dir, "dist")
    wheels = [
        os.path.join(dist_dir, w) for w in os.listdir(dist_dir) if w.endswith(".whl")
    ]
    fail_if(not wheels, "no wheel produced by the build")

    for wheel_path in wheels:
        print(f"=== Running auditwheel repair on {wheel_path}")
        r = subprocess.run(
            ["auditwheel", "repair", wheel_path, "--wheel-dir", dest_dir],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 and "no ELF" in r.stderr:
            arch = os.uname().machine
            manylinux_tag = f"manylinux_2_28_{arch}"
            print(
                f"=== Pure-Python wheel detected; falling back to wheel tags "
                f"({manylinux_tag})"
            )
            copied = os.path.join(dest_dir, os.path.basename(wheel_path))
            shutil.copy(wheel_path, copied)
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
    print("=== Building wheel")
    args = ["python3", "setup.py", "bdist_wheel"]
    # PEP 427 build tag: lets two wheels of the same version coexist
    # (e.g. reruns of the same CI pipeline). Sources, first non-empty
    # and usable wins:
    #   CI_PIPELINE_ID  - GitLab pipeline-scoped ID (preferred).
    #   NVIDIA_BUILD_ID - from build.py's --build-id flag.
    #   BUILD_NUMBER    - generic CI systems.
    # PEP 427 requires the build tag to start with a digit.
    build_tag = (
        os.environ.get("CI_PIPELINE_ID")
        or os.environ.get("NVIDIA_BUILD_ID")
        or os.environ.get("BUILD_NUMBER")
    )
    print(
        f"=== Wheel build-tag inputs: "
        f"CI_PIPELINE_ID={os.environ.get('CI_PIPELINE_ID')!r} "
        f"NVIDIA_BUILD_ID={os.environ.get('NVIDIA_BUILD_ID')!r} "
        f"BUILD_NUMBER={os.environ.get('BUILD_NUMBER')!r} "
        f"-> build-tag={build_tag!r}",
        file=sys.stderr,
    )
    if build_tag and build_tag != "<unknown>" and build_tag[:1].isdigit():
        args += [f"--build-number={build_tag}"]

    wenv = os.environ.copy()
    wenv["VERSION"] = _compose_version(FLAGS.triton_version)
    wenv["TRITON_PYBIND"] = PYBIND_LIB
    p = subprocess.Popen(args, env=wenv)
    p.wait()
    fail_if(p.returncode != 0, "setup.py failed")

    _repair_wheel_with_auditwheel(FLAGS.whl_dir, FLAGS.dest_dir)

    print(f"=== Output wheel file is in: {FLAGS.dest_dir}")
    touch(os.path.join(FLAGS.dest_dir, "stamp.whl"))


if __name__ == "__main__":
    main()
