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
"""variants: python_backend builds with TRITON_ENABLE_GPU=OFF.

Ported 1:1 from qa/L0_backend_python/variants/test.sh: clone python_backend,
configure and build it as a CPU-only variant, assert the build succeeds.
Nothing else -- no server, no inference; the original bash script's only
assertion is the exit code of `make install`.
"""

import os
import shutil
import subprocess

from conftest import (
    OUTPUT_DIR,
    PYTHON_BACKEND_REPO_TAG,
    TRITON_BACKEND_REPO_TAG,
    TRITON_COMMON_REPO_TAG,
    TRITON_CORE_REPO_TAG,
    TRITON_REPO_ORGANIZATION,
)


def _install_build_deps():
    """Mirrors install_build_deps_apt() in qa/L0_backend_python/common.sh.

    The base QA image has no cmake at all (verified: `FileNotFoundError:
    cmake` without this) -- common.sh doesn't rely on whatever cmake the
    image happens to ship, it pins a specific version from Kitware's apt
    repo, same as the original bash.
    """
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "software-properties-common", "rapidjson-dev"],
        check=True,
    )
    subprocess.run(["apt-get", "update", "-q=2"], check=True)
    subprocess.run(["apt-get", "install", "-y", "gpg", "wget"], check=True)
    key = subprocess.run(
        ["wget", "-O", "-", "https://apt.kitware.com/keys/kitware-archive-latest.asc"],
        check=True,
        capture_output=True,
    ).stdout
    keyring = subprocess.run(
        ["gpg", "--dearmor"], input=key, check=True, capture_output=True
    ).stdout
    with open("/usr/share/keyrings/kitware-archive-keyring.gpg", "wb") as f:
        f.write(keyring)
    codename = subprocess.run(
        ["bash", "-c", ". /etc/os-release && echo $UBUNTU_CODENAME"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with open("/etc/apt/sources.list.d/kitware.list", "w") as f:
        f.write(
            "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] "
            "https://apt.kitware.com/ubuntu/ %s main\n" % codename
        )
    subprocess.run(["apt-get", "update", "-q=2"], check=True)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake=4.0.3*",
            "cmake-data=4.0.3*",
        ],
        check=True,
    )


def test_cpu_only_build():
    _install_build_deps()

    clone_dir = os.path.join(OUTPUT_DIR, "python_backend")
    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subprocess.run(
        [
            "git",
            "clone",
            "%s/python_backend" % TRITON_REPO_ORGANIZATION,
            "-b",
            PYTHON_BACKEND_REPO_TAG,
            clone_dir,
        ],
        check=True,
    )

    build_dir = os.path.join(clone_dir, "builddir")
    os.makedirs(build_dir)
    env = dict(os.environ, CMAKE_POLICY_VERSION_MINIMUM="3.5")
    subprocess.run(
        [
            "cmake",
            "-DTRITON_ENABLE_GPU=OFF",
            "-DTRITON_REPO_ORGANIZATION:STRING=%s" % TRITON_REPO_ORGANIZATION,
            "-DTRITON_BACKEND_REPO_TAG=%s" % TRITON_BACKEND_REPO_TAG,
            "-DTRITON_COMMON_REPO_TAG=%s" % TRITON_COMMON_REPO_TAG,
            "-DTRITON_CORE_REPO_TAG=%s" % TRITON_CORE_REPO_TAG,
            "../",
        ],
        cwd=build_dir,
        env=env,
        check=True,
    )
    subprocess.run(["make", "-j18", "install"], cwd=build_dir, env=env, check=True)
