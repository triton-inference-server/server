#!/usr/bin/env python3
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib.util
import multiprocessing
import os
import os.path
import pathlib
import platform
import stat
import subprocess
import sys
from inspect import getsourcefile

import distro
import requests

#
# Build Triton Inference Server.
#

# By default build.py builds the Triton Docker image, but can also be
# used to build without Docker.  See docs/build.md and --help for more
# information.
#
# The TRITON_VERSION file indicates the Triton version and
# DEFAULT_TRITON_VERSION_MAP is used to determine the corresponding container
# version and upstream container version (upstream containers are
# dependencies required by Triton). These versions may be overridden.

# Map from Triton version to corresponding container and component versions.
#
#   triton version ->
#     (triton container version,
#      upstream container version,
#      ORT version,
#      ORT OpenVINO version (use None to disable OpenVINO in ORT),
#      Standalone OpenVINO version,
#      DCGM version
#     )
#
# Currently the OpenVINO versions used in ORT and standalone must
# match because of the way dlopen works with loading the backends. If
# different versions are used then one backend or the other will
# incorrectly load the other version of the openvino libraries.
#

DEFAULT_TRITON_VERSION_MAP = {
    "release_version": "2.56.0dev",
    "triton_container_version": "25.03dev",
    "upstream_container_version": "25.02",
    "ort_version": "1.20.1",
    "ort_openvino_version": "2025.0.0",
    "standalone_openvino_version": "2025.0.0",
    "dcgm_version": "3.3.6",
    "vllm_version": "0.7.0",
    "rhel_py_version": "3.12.3",
}

CORE_BACKENDS = ["ensemble"]

FLAGS = None
EXTRA_CORE_CMAKE_FLAGS = {}
OVERRIDE_CORE_CMAKE_FLAGS = {}
EXTRA_BACKEND_CMAKE_FLAGS = {}
OVERRIDE_BACKEND_CMAKE_FLAGS = {}

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print("<failed to log>", file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    fail_if(True, msg)


def fail_if(p, msg):
    if p:
        print("error: {}".format(msg), file=sys.stderr)
        sys.exit(1)


def target_platform():
    # When called by compose.py, FLAGS will be None
    if FLAGS and FLAGS.target_platform is not None:
        return FLAGS.target_platform
    platform_string = platform.system().lower()
    if platform_string == "linux":
        # Need to inspect the /etc/os-release file to get
        # the distribution of linux
        id_like_list = distro.like().split()
        if "debian" in id_like_list:
            return "linux"
        else:
            return "rhel"
    else:
        return platform_string


def target_machine():
    # When called by compose.py, FLAGS will be None
    if FLAGS and FLAGS.target_machine is not None:
        return FLAGS.target_machine
    return platform.machine().lower()


def container_versions(version, container_version, upstream_container_version):
    if container_version is None:
        container_version = FLAGS.triton_container_version
    if upstream_container_version is None:
        upstream_container_version = FLAGS.upstream_container_version
    return container_version, upstream_container_version


class BuildScript:
    """Utility class for writing build scripts"""

    def __init__(self, filepath, desc=None, verbose=False):
        self._filepath = filepath
        self._file = open(self._filepath, "w")
        self._verbose = verbose
        self.header(desc)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self._file is not None:
            if target_platform() == "windows":
                self.blankln()
                self._file.write("}\n")
                self._file.write("catch {\n")
                self._file.write("    $_;\n")
                self._file.write("    ExitWithCode 1;\n")
                self._file.write("}\n")
            """Close the file"""
            self._file.close()
            self._file = None
            st = os.stat(self._filepath)
            os.chmod(self._filepath, st.st_mode | stat.S_IEXEC)

    def blankln(self):
        self._file.write("\n")

    def commentln(self, cnt):
        self._file.write("#" * cnt + "\n")

    def comment(self, msg=""):
        if not isinstance(msg, str):
            try:
                for m in msg:
                    self._file.write(f"# {msg}\n")
                return
            except TypeError:
                pass
        self._file.write(f"# {msg}\n")

    def comment_verbose(self, msg=""):
        if self._verbose:
            self.comment(msg)

    def header(self, desc=None):
        if target_platform() != "windows":
            self._file.write("#!/usr/bin/env bash\n\n")

        if desc is not None:
            self.comment()
            self.comment(desc)
            self.comment()
            self.blankln()

        self.comment("Exit script immediately if any command fails")
        if target_platform() == "windows":
            self._file.write("$UseStructuredOutput = $false\n")
            self.blankln()
            self._file.write("function ExitWithCode($exitcode) {\n")
            self._file.write("    $host.SetShouldExit($exitcode)\n")
            self._file.write("    exit $exitcode\n")
            self._file.write("}\n")
            self.blankln()
            if self._verbose:
                self._file.write("Set-PSDebug -Trace 1\n")
            self.blankln()
            self._file.write("try {\n")
        else:
            self._file.write("set -e\n")
            if self._verbose:
                self._file.write("set -x\n")
        self.blankln()

    def envvar_ref(self, v):
        if target_platform() == "windows":
            return f"${{env:{v}}}"
        return f"${{{v}}}"

    def cmd(self, clist, check_exitcode=False):
        if isinstance(clist, str):
            self._file.write(f"{clist}\n")
        else:
            for c in clist:
                self._file.write(f"{c} ")
            self.blankln()

        if check_exitcode:
            if target_platform() == "windows":
                self._file.write("if ($LASTEXITCODE -ne 0) {\n")
                self._file.write(
                    '  Write-Output "exited with status code $LASTEXITCODE";\n'
                )
                self._file.write("  ExitWithCode 1;\n")
                self._file.write("}\n")

    def cwd(self, path):
        if target_platform() == "windows":
            self.cmd(f"Set-Location -EV Err -EA Stop {path}")
        else:
            self.cmd(f"cd {path}")

    def cp(self, src, dest):
        if target_platform() == "windows":
            self.cmd(f"Copy-Item -EV Err -EA Stop {src} -Destination {dest}")
        else:
            self.cmd(f"cp {src} {dest}")

    def mkdir(self, path):
        if target_platform() == "windows":
            self.cmd(
                f"New-Item -EV Err -EA Stop -ItemType Directory -Force -Path {path}"
            )
        else:
            self.cmd(f"mkdir -p {pathlib.Path(path)}")

    def rmdir(self, path):
        if target_platform() == "windows":
            self.cmd(f"if (Test-Path -Path {path}) {{")
            self.cmd(f"  Remove-Item -EV Err -EA Stop -Recurse -Force {path}")
            self.cmd("}")
        else:
            self.cmd(f"rm -fr {pathlib.Path(path)}")

    def cpdir(self, src, dest):
        if target_platform() == "windows":
            self.cmd(f"Copy-Item -EV Err -EA Stop -Recurse {src} -Destination {dest}")
        else:
            self.cmd(f"cp -r {src} {dest}")

    def tar(self, subdir, tar_filename):
        if target_platform() == "windows":
            fail("unsupported operation: tar")
        else:
            self.cmd(f"tar zcf {tar_filename} {subdir}")

    def cmake(self, args):
        # Pass some additional envvars into cmake...
        env_args = []
        for k in ("TRT_VERSION", "CMAKE_TOOLCHAIN_FILE", "VCPKG_TARGET_TRIPLET"):
            env_args += [f'"-D{k}={self.envvar_ref(k)}"']
        self.cmd(f'cmake {" ".join(env_args)} {" ".join(args)}', check_exitcode=True)

    def makeinstall(self, target="install"):
        verbose_flag = "-v" if self._verbose else ""
        self.cmd(
            f"cmake --build . --config {FLAGS.build_type} -j{FLAGS.build_parallel} {verbose_flag} -t {target}"
        )

    def gitclone(self, repo, tag, subdir, org):
        clone_dir = subdir
        if not FLAGS.no_force_clone:
            self.rmdir(clone_dir)

        if target_platform() == "windows":
            self.cmd(f"if (-Not (Test-Path -Path {clone_dir})) {{")
        else:
            self.cmd(f"if [[ ! -e {clone_dir} ]]; then")

        # FIXME [DLIS-4045 - Currently the tag starting with "pull/" is not
        # working with "--repo-tag" as the option is not forwarded to the
        # individual repo build correctly.]
        # If 'tag' starts with "pull/" then it must be of form
        # "pull/<pr>/head". We just clone at "main" and then fetch the
        # reference onto a new branch we name "tritonbuildref".
        if tag.startswith("pull/"):
            self.cmd(
                f"  git clone --recursive --depth=1 {org}/{repo}.git {subdir};",
                check_exitcode=True,
            )
            self.cmd("}" if target_platform() == "windows" else "fi")
            self.cwd(subdir)
            self.cmd(f"git fetch origin {tag}:tritonbuildref", check_exitcode=True)
            self.cmd(f"git checkout tritonbuildref", check_exitcode=True)
        else:
            self.cmd(
                f"  git clone --recursive --single-branch --depth=1 -b {tag} {org}/{repo}.git {subdir};",
                check_exitcode=True,
            )
            self.cmd("}" if target_platform() == "windows" else "fi")


def cmake_core_arg(name, type, value):
    # Return cmake -D setting to set name=value for core build. Use
    # command-line specified value if one is given.
    if name in OVERRIDE_CORE_CMAKE_FLAGS:
        value = OVERRIDE_CORE_CMAKE_FLAGS[name]
    if type is None:
        type = ""
    else:
        type = ":{}".format(type)
    return '"-D{}{}={}"'.format(name, type, value)


def cmake_core_enable(name, flag):
    # Return cmake -D setting to set name=flag?ON:OFF for core
    # build. Use command-line specified value for 'flag' if one is
    # given.
    if name in OVERRIDE_CORE_CMAKE_FLAGS:
        value = OVERRIDE_CORE_CMAKE_FLAGS[name]
    else:
        value = "ON" if flag else "OFF"
    return '"-D{}:BOOL={}"'.format(name, value)


def cmake_core_extra_args():
    args = []
    for k, v in EXTRA_CORE_CMAKE_FLAGS.items():
        args.append('"-D{}={}"'.format(k, v))
    return args


def cmake_backend_arg(backend, name, type, value):
    # Return cmake -D setting to set name=value for backend build. Use
    # command-line specified value if one is given.
    if backend in OVERRIDE_BACKEND_CMAKE_FLAGS:
        if name in OVERRIDE_BACKEND_CMAKE_FLAGS[backend]:
            value = OVERRIDE_BACKEND_CMAKE_FLAGS[backend][name]
    if type is None:
        type = ""
    else:
        type = ":{}".format(type)
    return '"-D{}{}={}"'.format(name, type, value)


def cmake_backend_enable(backend, name, flag):
    # Return cmake -D setting to set name=flag?ON:OFF for backend
    # build. Use command-line specified value for 'flag' if one is
    # given.
    value = None
    if backend in OVERRIDE_BACKEND_CMAKE_FLAGS:
        if name in OVERRIDE_BACKEND_CMAKE_FLAGS[backend]:
            value = OVERRIDE_BACKEND_CMAKE_FLAGS[backend][name]
    if value is None:
        value = "ON" if flag else "OFF"
    return '"-D{}:BOOL={}"'.format(name, value)


def cmake_backend_extra_args(backend):
    args = []
    if backend in EXTRA_BACKEND_CMAKE_FLAGS:
        for k, v in EXTRA_BACKEND_CMAKE_FLAGS[backend].items():
            args.append('"-D{}={}"'.format(k, v))
    return args


def cmake_repoagent_arg(name, type, value):
    # For now there is no override for repo-agents
    if type is None:
        type = ""
    else:
        type = ":{}".format(type)
    return '"-D{}{}={}"'.format(name, type, value)


def cmake_repoagent_enable(name, flag):
    # For now there is no override for repo-agents
    value = "ON" if flag else "OFF"
    return '"-D{}:BOOL={}"'.format(name, value)


def cmake_repoagent_extra_args():
    # For now there is no extra args for repo-agents
    args = []
    return args


def cmake_cache_arg(name, type, value):
    # For now there is no override for caches
    if type is None:
        type = ""
    else:
        type = ":{}".format(type)
    return '"-D{}{}={}"'.format(name, type, value)


def cmake_cache_enable(name, flag):
    # For now there is no override for caches
    value = "ON" if flag else "OFF"
    return '"-D{}:BOOL={}"'.format(name, value)


def cmake_cache_extra_args():
    # For now there is no extra args for caches
    args = []
    return args


def core_cmake_args(components, backends, cmake_dir, install_dir):
    cargs = [
        cmake_core_arg("CMAKE_BUILD_TYPE", None, FLAGS.build_type),
        cmake_core_arg("CMAKE_INSTALL_PREFIX", "PATH", install_dir),
        cmake_core_arg("TRITON_VERSION", "STRING", FLAGS.version),
        cmake_core_arg("TRITON_REPO_ORGANIZATION", "STRING", FLAGS.github_organization),
        cmake_core_arg("TRITON_COMMON_REPO_TAG", "STRING", components["common"]),
        cmake_core_arg("TRITON_CORE_REPO_TAG", "STRING", components["core"]),
        cmake_core_arg("TRITON_BACKEND_REPO_TAG", "STRING", components["backend"]),
        cmake_core_arg(
            "TRITON_THIRD_PARTY_REPO_TAG", "STRING", components["thirdparty"]
        ),
    ]

    cargs.append(cmake_core_enable("TRITON_ENABLE_LOGGING", FLAGS.enable_logging))
    cargs.append(cmake_core_enable("TRITON_ENABLE_STATS", FLAGS.enable_stats))
    cargs.append(cmake_core_enable("TRITON_ENABLE_METRICS", FLAGS.enable_metrics))
    cargs.append(
        cmake_core_enable("TRITON_ENABLE_METRICS_GPU", FLAGS.enable_gpu_metrics)
    )
    cargs.append(
        cmake_core_enable("TRITON_ENABLE_METRICS_CPU", FLAGS.enable_cpu_metrics)
    )
    cargs.append(cmake_core_enable("TRITON_ENABLE_TRACING", FLAGS.enable_tracing))
    cargs.append(cmake_core_enable("TRITON_ENABLE_NVTX", FLAGS.enable_nvtx))

    cargs.append(cmake_core_enable("TRITON_ENABLE_GPU", FLAGS.enable_gpu))
    cargs.append(
        cmake_core_arg(
            "TRITON_MIN_COMPUTE_CAPABILITY", None, FLAGS.min_compute_capability
        )
    )

    cargs.append(cmake_core_enable("TRITON_ENABLE_MALI_GPU", FLAGS.enable_mali_gpu))

    cargs.append(cmake_core_enable("TRITON_ENABLE_GRPC", "grpc" in FLAGS.endpoint))
    cargs.append(cmake_core_enable("TRITON_ENABLE_HTTP", "http" in FLAGS.endpoint))
    cargs.append(
        cmake_core_enable("TRITON_ENABLE_SAGEMAKER", "sagemaker" in FLAGS.endpoint)
    )
    cargs.append(
        cmake_core_enable("TRITON_ENABLE_VERTEX_AI", "vertex-ai" in FLAGS.endpoint)
    )

    cargs.append(cmake_core_enable("TRITON_ENABLE_GCS", "gcs" in FLAGS.filesystem))
    cargs.append(cmake_core_enable("TRITON_ENABLE_S3", "s3" in FLAGS.filesystem))
    cargs.append(
        cmake_core_enable(
            "TRITON_ENABLE_AZURE_STORAGE", "azure_storage" in FLAGS.filesystem
        )
    )

    cargs.append(cmake_core_enable("TRITON_ENABLE_ENSEMBLE", "ensemble" in backends))
    cargs.append(cmake_core_enable("TRITON_ENABLE_TENSORRT", "tensorrt" in backends))

    cargs += cmake_core_extra_args()
    cargs.append(cmake_dir)
    return cargs


def repoagent_repo(ra):
    return "{}_repository_agent".format(ra)


def repoagent_cmake_args(images, components, ra, install_dir):
    args = []

    cargs = args + [
        cmake_repoagent_arg("CMAKE_BUILD_TYPE", None, FLAGS.build_type),
        cmake_repoagent_arg("CMAKE_INSTALL_PREFIX", "PATH", install_dir),
        cmake_repoagent_arg(
            "TRITON_REPO_ORGANIZATION", "STRING", FLAGS.github_organization
        ),
        cmake_repoagent_arg("TRITON_COMMON_REPO_TAG", "STRING", components["common"]),
        cmake_repoagent_arg("TRITON_CORE_REPO_TAG", "STRING", components["core"]),
    ]

    cargs.append(cmake_repoagent_enable("TRITON_ENABLE_GPU", FLAGS.enable_gpu))
    cargs += cmake_repoagent_extra_args()
    cargs.append("..")
    return cargs


def cache_repo(cache):
    # example: "local", or "redis"
    return "{}_cache".format(cache)


def cache_cmake_args(images, components, cache, install_dir):
    args = []

    cargs = args + [
        cmake_cache_arg("CMAKE_BUILD_TYPE", None, FLAGS.build_type),
        cmake_cache_arg("CMAKE_INSTALL_PREFIX", "PATH", install_dir),
        cmake_cache_arg(
            "TRITON_REPO_ORGANIZATION", "STRING", FLAGS.github_organization
        ),
        cmake_cache_arg("TRITON_COMMON_REPO_TAG", "STRING", components["common"]),
        cmake_cache_arg("TRITON_CORE_REPO_TAG", "STRING", components["core"]),
    ]

    cargs.append(cmake_cache_enable("TRITON_ENABLE_GPU", FLAGS.enable_gpu))
    cargs += cmake_cache_extra_args()
    cargs.append("..")
    return cargs


def backend_repo(be):
    return "{}_backend".format(be)


def backend_cmake_args(images, components, be, install_dir, library_paths):
    cmake_build_type = FLAGS.build_type

    if be == "onnxruntime":
        args = onnxruntime_cmake_args(images, library_paths)
    elif be == "openvino":
        args = openvino_cmake_args()
    elif be == "tensorflow":
        args = tensorflow_cmake_args(images, library_paths)
    elif be == "python":
        args = python_cmake_args()
    elif be == "dali":
        args = dali_cmake_args()
    elif be == "pytorch":
        args = pytorch_cmake_args(images)
    elif be == "armnn_tflite":
        args = armnn_tflite_cmake_args()
    elif be == "fil":
        args = fil_cmake_args(images)
        # DLIS-4618: FIL backend fails debug build, so override it for now.
        cmake_build_type = "Release"
    elif be == "fastertransformer":
        args = fastertransformer_cmake_args()
    elif be == "tensorrt":
        args = tensorrt_cmake_args()
    elif be == "tensorrtllm":
        args = tensorrtllm_cmake_args(images)
    else:
        args = []

    cargs = args + [
        cmake_backend_arg(be, "CMAKE_BUILD_TYPE", None, cmake_build_type),
        cmake_backend_arg(be, "CMAKE_INSTALL_PREFIX", "PATH", install_dir),
        cmake_backend_arg(
            be, "TRITON_REPO_ORGANIZATION", "STRING", FLAGS.github_organization
        ),
        cmake_backend_arg(be, "TRITON_COMMON_REPO_TAG", "STRING", components["common"]),
        cmake_backend_arg(be, "TRITON_CORE_REPO_TAG", "STRING", components["core"]),
        cmake_backend_arg(
            be, "TRITON_BACKEND_REPO_TAG", "STRING", components["backend"]
        ),
    ]

    cargs.append(cmake_backend_enable(be, "TRITON_ENABLE_GPU", FLAGS.enable_gpu))
    cargs.append(
        cmake_backend_enable(be, "TRITON_ENABLE_MALI_GPU", FLAGS.enable_mali_gpu)
    )
    cargs.append(cmake_backend_enable(be, "TRITON_ENABLE_STATS", FLAGS.enable_stats))
    cargs.append(
        cmake_backend_enable(be, "TRITON_ENABLE_METRICS", FLAGS.enable_metrics)
    )

    # [DLIS-4950] always enable below once Windows image is updated with CUPTI
    # cargs.append(cmake_backend_enable(be, 'TRITON_ENABLE_MEMORY_TRACKER', True))
    if (target_platform() == "windows") and (not FLAGS.no_container_build):
        print(
            "Warning: Detected docker build is used for Windows, backend utility 'device memory tracker' will be disabled due to missing library in CUDA Windows docker image."
        )
        cargs.append(cmake_backend_enable(be, "TRITON_ENABLE_MEMORY_TRACKER", False))
    elif target_platform() == "igpu":
        print(
            "Warning: Detected iGPU build, backend utility 'device memory tracker' will be disabled as iGPU doesn't contain required version of the library."
        )
        cargs.append(cmake_backend_enable(be, "TRITON_ENABLE_MEMORY_TRACKER", False))
    elif FLAGS.enable_gpu:
        cargs.append(cmake_backend_enable(be, "TRITON_ENABLE_MEMORY_TRACKER", True))

    cargs += cmake_backend_extra_args(be)
    if be == "tensorrtllm":
        cargs.append("-S ../inflight_batcher_llm -B .")

    else:
        cargs.append("..")
    return cargs


def python_cmake_args():
    cargs = []
    if target_platform() == "rhel":
        cargs.append(
            cmake_backend_arg(
                "python", "PYBIND11_PYTHON_VERSION", "STRING", FLAGS.rhel_py_version
            )
        )

    return cargs


def pytorch_cmake_args(images):
    if "pytorch" in images:
        image = images["pytorch"]
    else:
        image = "nvcr.io/nvidia/pytorch:{}-py3".format(FLAGS.upstream_container_version)
    cargs = [
        cmake_backend_arg("pytorch", "TRITON_PYTORCH_DOCKER_IMAGE", None, image),
    ]

    # TODO: TPRD-372 TorchTRT extension is not currently supported by our manylinux build
    # TODO: TPRD-373 NVTX extension is not currently supported by our manylinux build
    if target_platform() != "rhel":
        if FLAGS.enable_gpu:
            cargs.append(
                cmake_backend_enable("pytorch", "TRITON_PYTORCH_ENABLE_TORCHTRT", True)
            )
        cargs.append(
            cmake_backend_enable("pytorch", "TRITON_ENABLE_NVTX", FLAGS.enable_nvtx)
        )
    return cargs


def onnxruntime_cmake_args(images, library_paths):
    cargs = [
        cmake_backend_arg(
            "onnxruntime",
            "TRITON_BUILD_ONNXRUNTIME_VERSION",
            None,
            os.getenv("TRITON_BUILD_ONNXRUNTIME_VERSION")
            if os.getenv("TRITON_BUILD_ONNXRUNTIME_VERSION")
            else FLAGS.ort_version,
        )
    ]

    # TRITON_ENABLE_GPU is already set for all backends in backend_cmake_args()
    if FLAGS.enable_gpu:
        # TODO: TPRD-712 TensorRT is not currently supported by our RHEL build for SBSA.
        if target_platform() != "rhel" or (
            target_platform() == "rhel" and target_machine() == "x86_64"
        ):
            cargs.append(
                cmake_backend_enable(
                    "onnxruntime", "TRITON_ENABLE_ONNXRUNTIME_TENSORRT", True
                )
            )

    if target_platform() == "windows":
        if "base" in images:
            cargs.append(
                cmake_backend_arg(
                    "onnxruntime", "TRITON_BUILD_CONTAINER", None, images["base"]
                )
            )
    else:
        if "base" in images:
            cargs.append(
                cmake_backend_arg(
                    "onnxruntime", "TRITON_BUILD_CONTAINER", None, images["base"]
                )
            )
        else:
            cargs.append(
                cmake_backend_arg(
                    "onnxruntime",
                    "TRITON_BUILD_CONTAINER_VERSION",
                    None,
                    FLAGS.triton_container_version,
                )
            )

        # TODO: TPRD-333 OpenVino extension is not currently supported by our manylinux build
        if (
            (target_machine() != "aarch64")
            and (target_platform() != "rhel")
            and (FLAGS.ort_openvino_version is not None)
        ):
            cargs.append(
                cmake_backend_enable(
                    "onnxruntime", "TRITON_ENABLE_ONNXRUNTIME_OPENVINO", True
                )
            )
            cargs.append(
                cmake_backend_arg(
                    "onnxruntime",
                    "TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION",
                    None,
                    FLAGS.ort_openvino_version,
                )
            )

        if (target_platform() == "igpu") or (target_platform() == "rhel"):
            cargs.append(
                cmake_backend_arg(
                    "onnxruntime",
                    "TRITON_BUILD_TARGET_PLATFORM",
                    None,
                    target_platform(),
                )
            )

    return cargs


def openvino_cmake_args():
    cargs = [
        cmake_backend_arg(
            "openvino",
            "TRITON_BUILD_OPENVINO_VERSION",
            None,
            FLAGS.standalone_openvino_version,
        )
    ]
    if target_platform() == "windows":
        if "base" in images:
            cargs.append(
                cmake_backend_arg(
                    "openvino", "TRITON_BUILD_CONTAINER", None, images["base"]
                )
            )
    else:
        if "base" in images:
            cargs.append(
                cmake_backend_arg(
                    "openvino", "TRITON_BUILD_CONTAINER", None, images["base"]
                )
            )
        else:
            cargs.append(
                cmake_backend_arg(
                    "openvino",
                    "TRITON_BUILD_CONTAINER_VERSION",
                    None,
                    FLAGS.upstream_container_version,
                )
            )
    return cargs


def tensorrt_cmake_args():
    cargs = [
        cmake_backend_enable("tensorrt", "TRITON_ENABLE_NVTX", FLAGS.enable_nvtx),
    ]
    if target_platform() == "windows":
        cargs.append(
            cmake_backend_arg(
                "tensorrt", "TRITON_TENSORRT_INCLUDE_PATHS", None, "c:/TensorRT/include"
            )
        )

    return cargs


def tensorflow_cmake_args(images, library_paths):
    backend_name = "tensorflow"
    extra_args = []

    # If a specific TF image is specified use it, otherwise pull from NGC.
    if backend_name in images:
        image = images[backend_name]
    else:
        image = "nvcr.io/nvidia/tensorflow:{}-tf2-py3".format(
            FLAGS.upstream_container_version
        )
    extra_args = [
        cmake_backend_arg(backend_name, "TRITON_TENSORFLOW_DOCKER_IMAGE", None, image)
    ]
    return extra_args


def dali_cmake_args():
    return [
        cmake_backend_enable("dali", "TRITON_DALI_SKIP_DOWNLOAD", False),
    ]


def fil_cmake_args(images):
    cargs = [cmake_backend_enable("fil", "TRITON_FIL_DOCKER_BUILD", True)]
    if "base" in images:
        cargs.append(
            cmake_backend_arg("fil", "TRITON_BUILD_CONTAINER", None, images["base"])
        )
    else:
        cargs.append(
            cmake_backend_arg(
                "fil",
                "TRITON_BUILD_CONTAINER_VERSION",
                None,
                FLAGS.upstream_container_version,
            )
        )

    return cargs


def armnn_tflite_cmake_args():
    return [
        cmake_backend_arg("armnn_tflite", "JOBS", None, multiprocessing.cpu_count()),
    ]


def fastertransformer_cmake_args():
    print("Warning: FasterTransformer backend is not officially supported.")
    cargs = [
        cmake_backend_arg(
            "fastertransformer", "CMAKE_EXPORT_COMPILE_COMMANDS", None, 1
        ),
        cmake_backend_arg("fastertransformer", "ENABLE_FP8", None, "OFF"),
    ]
    return cargs


def tensorrtllm_cmake_args(images):
    cargs = []
    cargs.append(cmake_backend_enable("tensorrtllm", "USE_CXX11_ABI", True))
    return cargs


def install_dcgm_libraries(dcgm_version, target_machine):
    if dcgm_version == "":
        fail(
            "unable to determine default repo-tag, DCGM version not known for {}".format(
                FLAGS.version
            )
        )
        return ""
    else:
        # RHEL has the same install instructions for both aarch64 and x86
        if target_platform() == "rhel":
            if target_machine == "aarch64":
                return """
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo \\
    && dnf clean expire-cache \\
    && dnf install -y datacenter-gpu-manager-{}
""".format(
                    dcgm_version, dcgm_version
                )
            else:
                return """
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \\
    && dnf clean expire-cache \\
    && dnf install -y datacenter-gpu-manager-{}
""".format(
                    dcgm_version, dcgm_version
                )
        else:
            if target_machine == "aarch64":
                return """
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN curl -o /tmp/cuda-keyring.deb \\
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \\
      && apt install /tmp/cuda-keyring.deb \\
      && rm /tmp/cuda-keyring.deb \\
      && apt-get update \\
      && apt-get install -y datacenter-gpu-manager=1:{}
""".format(
                    dcgm_version, dcgm_version
                )
            else:
                return """
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN curl -o /tmp/cuda-keyring.deb \\
          https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \\
      && apt install /tmp/cuda-keyring.deb \\
      && rm /tmp/cuda-keyring.deb \\
      && apt-get update \\
      && apt-get install -y datacenter-gpu-manager=1:{}
""".format(
                    dcgm_version, dcgm_version
                )


def create_dockerfile_buildbase_rhel(ddir, dockerfile_name, argmap):
    df = """
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}
""".format(
        argmap["TRITON_VERSION"],
        argmap["TRITON_CONTAINER_VERSION"],
        argmap["BASE_IMAGE"],
    )

    df += """
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
ENV PIP_BREAK_SYSTEM_PACKAGES=1
"""
    df += """
# Install docker docker buildx
RUN yum install -y ca-certificates curl gnupg yum-utils \\
      && yum-config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo \\
      && yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
#   && yum install -y docker.io docker-buildx-plugin

# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip and libarchive-dev is needed by python backend
# libxml2-dev is needed for Azure Storage
# scons is needed for armnn_tflite backend build dep
RUN yum install -y \\
            autoconf \\
            automake \\
            bzip2-devel \\
            ca-certificates \\
            git \\
            gperf \\
            gperftools-devel \\
            libarchive-devel \\
            libb64-devel \\
            libcurl-devel \\
            libtool \\
            libxml2-devel \\
            ncurses-devel \\
            numactl-devel \\
            openssl-devel \\
            pkg-config \\
            python3-pip \\
            python3-scons \\
            python3-setuptools \\
            rapidjson-devel \\
            re2-devel \\
            readline-devel \\
            unzip \\
            wget \\
            xz-devel \\
            zlib-devel
"""
    if os.getenv("CCACHE_REMOTE_ONLY") and os.getenv("CCACHE_REMOTE_STORAGE"):
        df += """
RUN curl -k -s -L https://github.com/ccache/ccache/archive/refs/tags/v4.10.2.tar.gz -o /tmp/ccache.tar.gz \\
    && tar -xzf /tmp/ccache.tar.gz -C /tmp \\
    && cmake -D CMAKE_BUILD_TYPE=Release -S /tmp/ccache-4.10.2 -B /tmp/build \\
    && cmake --build /tmp/build -j$(nproc) -t install \\
    && rm -rf /tmp/ccache.tar.gz /tmp/ccache-4.10.2 /tmp/build

ENV CCACHE_REMOTE_ONLY="true" \\
    CCACHE_REMOTE_STORAGE="{}" \\
    CMAKE_CXX_COMPILER_LAUNCHER="ccache" \\
    CMAKE_C_COMPILER_LAUNCHER="ccache" \\
    CMAKE_CUDA_COMPILER_LAUNCHER="ccache"

RUN ccache -p
""".format(
            os.getenv("CCACHE_REMOTE_STORAGE")
        )
    # Requires openssl-devel to be installed first for pyenv build to be successful
    df += change_default_python_version_rhel(FLAGS.rhel_py_version)
    df += """

RUN pip3 install --upgrade pip \\
      && pip3 install --upgrade \\
          build \\
          wheel \\
          setuptools \\
          docker \\
          virtualenv \\
          patchelf==0.17.2

# Install boost version >= 1.78 for boost::span
# Current libboost-dev apt packages are < 1.78, so install from tar.gz
RUN wget -O /tmp/boost.tar.gz \\
          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \\
      && (cd /tmp && tar xzf boost.tar.gz) \\
      && mv /tmp/boost_1_80_0/boost /usr/include/boost

# Server build requires recent version of CMake (FetchContent required)
# Might not need this if the installed version of cmake is high enough for our build.
# RUN apt update -q=2 \\
#       && apt install -y gpg wget \\
#       && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \\
#       && . /etc/os-release \\
#       && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \\
#       && apt-get update -q=2 \\
#       && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7*
"""
    if FLAGS.enable_gpu:
        df += install_dcgm_libraries(argmap["DCGM_VERSION"], target_machine())
    df += """
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
"""

    df += """
WORKDIR /workspace
RUN rm -fr *
COPY . .
ENTRYPOINT []
"""

    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_buildbase(ddir, dockerfile_name, argmap):
    df = """
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}
""".format(
        argmap["TRITON_VERSION"],
        argmap["TRITON_CONTAINER_VERSION"],
        argmap["BASE_IMAGE"],
    )

    df += """
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
ENV PIP_BREAK_SYSTEM_PACKAGES=1
"""
    # Install the windows- or linux-specific buildbase dependencies
    if target_platform() == "windows":
        df += """
RUN python3 -m pip install build

SHELL ["cmd", "/S", "/C"]
"""
    else:
        df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Install docker docker buildx
RUN apt-get update \\
      && apt-get install -y ca-certificates curl gnupg \\
      && install -m 0755 -d /etc/apt/keyrings \\
      && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \\
      && chmod a+r /etc/apt/keyrings/docker.gpg \\
      && echo \\
          "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \\
          "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \\
          tee /etc/apt/sources.list.d/docker.list > /dev/null \\
      && apt-get update \\
      && apt-get install -y docker.io docker-buildx-plugin

# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip and libarchive-dev is needed by python backend
# libxml2-dev is needed for Azure Storage
# scons is needed for armnn_tflite backend build dep
RUN apt-get update \\
      && apt-get install -y --no-install-recommends \\
            ca-certificates \\
            autoconf \\
            automake \\
            build-essential \\
            git \\
            gperf \\
            libre2-dev \\
            libssl-dev \\
            libtool \\
            libcurl4-openssl-dev \\
            libb64-dev \\
            libgoogle-perftools-dev \\
            python3-dev \\
            python3-pip \\
            python3-wheel \\
            python3-setuptools \\
            rapidjson-dev \\
            scons \\
            software-properties-common \\
            pkg-config \\
            unzip \\
            wget \\
            zlib1g-dev \\
            libarchive-dev \\
            libxml2-dev \\
            libnuma-dev \\
            wget \\
      && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade \\
          build \\
          docker \\
          virtualenv \\
          patchelf==0.17.2

# Install boost version >= 1.78 for boost::span
# Current libboost-dev apt packages are < 1.78, so install from tar.gz
RUN wget -O /tmp/boost.tar.gz \\
          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \\
      && (cd /tmp && tar xzf boost.tar.gz) \\
      && mv /tmp/boost_1_80_0/boost /usr/include/boost

# Server build requires recent version of CMake (FetchContent required)
RUN apt update -q=2 \\
      && apt install -y gpg wget \\
      && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \\
      && . /etc/os-release \\
      && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \\
      && apt-get update -q=2 \\
      && apt-get install -y --no-install-recommends cmake=3.28.3* cmake-data=3.28.3*
"""

        if FLAGS.enable_gpu:
            df += install_dcgm_libraries(argmap["DCGM_VERSION"], target_machine())

    df += """
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
"""

    if os.getenv("CCACHE_REMOTE_ONLY") and os.getenv("CCACHE_REMOTE_STORAGE"):
        df += """
ENV CCACHE_REMOTE_ONLY="true" \\
    CCACHE_REMOTE_STORAGE="{}" \\
    CMAKE_CXX_COMPILER_LAUNCHER="ccache" \\
    CMAKE_C_COMPILER_LAUNCHER="ccache" \\
    CMAKE_CUDA_COMPILER_LAUNCHER="ccache"

RUN apt-get update \\
      && apt-get install -y --no-install-recommends ccache && ccache -p \\
      && rm -rf /var/lib/apt/lists/*
""".format(
            os.getenv("CCACHE_REMOTE_STORAGE")
        )

    # Copy in the triton source. We remove existing contents first in
    # case the FROM container has something there already.
    if target_platform() == "windows":
        df += """
WORKDIR /workspace
RUN rmdir /S/Q * || exit 0
COPY . .
"""
    else:
        df += """
WORKDIR /workspace
RUN rm -fr *
COPY . .
ENTRYPOINT []
"""

    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_cibase(ddir, dockerfile_name, argmap):
    df = """
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}
""".format(
        argmap["TRITON_VERSION"],
        argmap["TRITON_CONTAINER_VERSION"],
        argmap["BASE_IMAGE"],
    )

    df += """
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

COPY build/ci /workspace

WORKDIR /workspace

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
"""

    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_linux(
    ddir, dockerfile_name, argmap, backends, repoagents, caches, endpoints
):
    df = """
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}

""".format(
        argmap["TRITON_VERSION"],
        argmap["TRITON_CONTAINER_VERSION"],
        argmap["BASE_IMAGE"],
    )

    # PyTorch and TensorFlow backends need extra CUDA and other
    # dependencies during runtime that are missing in the CPU-only base container.
    # These dependencies must be copied from the Triton Min image.
    if not FLAGS.enable_gpu and (("pytorch" in backends) or ("tensorflow" in backends)):
        df += """
############################################################################
##  Triton Min image
############################################################################
FROM {} AS min_container

""".format(
            argmap["GPU_BASE_IMAGE"]
        )

    df += """
############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1
"""

    df += dockerfile_prepare_container_linux(
        argmap, backends, FLAGS.enable_gpu, target_machine()
    )

    df += """
WORKDIR /opt
COPY --chown=1000:1000 build/install tritonserver

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .
RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \\
    "tritonserver-*.whl" | xargs -I {} pip install --upgrade {}[all] && \\
    find /opt/tritonserver/python -maxdepth 1 -type f -name \\
    "tritonfrontend-*.whl" | xargs -I {} pip install --upgrade {}[all]

RUN pip3 install -r python/openai/requirements.txt

"""
    if not FLAGS.no_core_build:
        # Add feature labels for SageMaker endpoint
        if "sagemaker" in endpoints:
            df += """
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
COPY --chown=1000:1000 docker/sagemaker/serve /usr/bin/.
"""
    # This is required since libcublasLt.so is not present during the build
    # stage of the PyTorch backend
    if not FLAGS.enable_gpu and ("pytorch" in backends):
        df += """
RUN patchelf --add-needed /usr/local/cuda/lib64/stubs/libcublasLt.so.12 backends/pytorch/libtorch_cuda.so
"""
    if "tensorrtllm" in backends:
        df += """
# Install required packages for TRT-LLM models
# Remove contents that are not needed in runtime
# Setuptools has breaking changes in version 70.0.0, so fix it to 69.5.1
# The generated code in grpc_service_pb2_grpc.py depends on grpcio>=1.64.0, so fix it to 1.64.0
RUN ldconfig && \\
    ARCH="$(uname -i)" && \\
    rm -fr ${TRT_ROOT}/bin ${TRT_ROOT}/targets/${ARCH}-linux-gnu/bin ${TRT_ROOT}/data && \\
    rm -fr ${TRT_ROOT}/doc ${TRT_ROOT}/onnx_graphsurgeon ${TRT_ROOT}/python && \\
    rm -fr ${TRT_ROOT}/samples ${TRT_ROOT}/targets/${ARCH}-linux-gnu/samples && \\
    pip3 install --no-cache-dir transformers && \\
    find /usr -name libtensorrt_llm.so -exec dirname {} \; > /etc/ld.so.conf.d/tensorrt-llm.conf && \\
    find /opt/tritonserver -name libtritonserver.so -exec dirname {} \; > /etc/ld.so.conf.d/triton-tensorrtllm-worker.conf && \\
    pip3 install --no-cache-dir  grpcio-tools==1.64.0 && \\
    pip3 uninstall -y setuptools
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:$LD_LIBRARY_PATH

# There are some ucc issues when spawning mpi processes with ompi v4.1.7a1.
# Downgrade to ompi v4.1.5rc2 to avoid the issue.
RUN rm -fr /opt/hpcx/ompi
COPY --from=nvcr.io/nvidia/tritonserver:24.02-py3-min /opt/hpcx/ompi /opt/hpcx/ompi
"""
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def dockerfile_prepare_container_linux(argmap, backends, enable_gpu, target_machine):
    gpu_enabled = 1 if enable_gpu else 0
    # Common steps to produce docker images shared by build.py and compose.py.
    # Sets environment variables, installs dependencies and adds entrypoint
    df = """
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
# Remove once https://github.com/openucx/ucx/pull/9148 is available
# in the min container.
ENV UCX_MEM_EVENTS no
"""

    # Necessary for libtorch.so to find correct HPCX libraries
    if "pytorch" in backends:
        df += """
ENV LD_LIBRARY_PATH /opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib/:${LD_LIBRARY_PATH}
"""

    backend_dependencies = ""
    # libgomp1 is needed by both onnxruntime and pytorch backends
    if ("onnxruntime" in backends) or ("pytorch" in backends):
        backend_dependencies = "libgomp1"

    # libgfortran5 is needed by pytorch backend on ARM
    if ("pytorch" in backends) and (target_machine == "aarch64"):
        backend_dependencies += " libgfortran5"
    # openssh-server is needed for fastertransformer
    if "fastertransformer" in backends:
        backend_dependencies += " openssh-server"

    df += """
ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    {gpu_enabled}

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true \\
      && userdel ubuntu > /dev/null 2>&1 || true \\
      && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \\
          useradd $TRITON_SERVER_USER; \\
        fi \\
      && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \\
      && [ `id -g $TRITON_SERVER_USER` -eq 1000 ]
""".format(
        gpu_enabled=gpu_enabled
    )

    if target_platform() == "rhel":
        df += """
# Common dependencies.
RUN yum install -y \\
        git \\
        gperf \\
        re2-devel \\
        openssl-devel \\
        libtool \\
        libcurl-devel \\
        libb64-devel \\
        gperftools-devel \\
        wget \\
        python3-pip \\
        numactl-devel

RUN pip3 install patchelf==0.17.2

"""
    else:
        df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN apt-get update \\
      && apt-get install -y --no-install-recommends \\
              clang \\
              curl \\
              dirmngr \\
              git \\
              gperf \\
              libb64-0d \\
              libcurl4-openssl-dev \\
              libgoogle-perftools-dev \\
              libjemalloc-dev \\
              libnuma-dev \\
              software-properties-common \\
              wget \\
              {backend_dependencies} \\
              python3-pip \\
      && rm -rf /var/lib/apt/lists/*
""".format(
            backend_dependencies=backend_dependencies
        )

    df += """
# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
ENV TCMALLOC_RELEASE_RATE 200
"""

    if "fastertransformer" in backends:
        be = "fastertransformer"
        url = "https://raw.githubusercontent.com/triton-inference-server/fastertransformer_backend/{}/docker/create_dockerfile_and_build.py".format(
            backends[be]
        )
        response = requests.get(url)
        spec = importlib.util.spec_from_loader(
            "fastertransformer_buildscript", loader=None, origin=url
        )
        fastertransformer_buildscript = importlib.util.module_from_spec(spec)
        exec(response.content, fastertransformer_buildscript.__dict__)
        df += fastertransformer_buildscript.create_postbuild(is_multistage_build=False)

    if enable_gpu:
        df += install_dcgm_libraries(argmap["DCGM_VERSION"], target_machine)
        # This segment will break the RHEL SBSA build. Need to determine whether
        # this is necessary to incorporate.
        if target_platform() != "rhel":
            df += """
# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \\
    && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \\
    && ldconfig \\
    && rm -f ${_CUDA_COMPAT_PATH}/lib
"""
    else:
        df += add_cpu_libs_to_linux_dockerfile(backends, target_machine)

    # Add dependencies needed for python backend
    if "python" in backends:
        if target_platform() == "rhel":
            df += """
# python3, python3-pip and some pip installs required for the python backend
RUN yum install -y \\
        libarchive-devel \\
        openssl-devel \\
        readline-devel
"""
            # Requires openssl-devel to be installed first for pyenv build to be successful
            df += change_default_python_version_rhel(FLAGS.rhel_py_version)
            df += """
RUN pip3 install --upgrade pip \\
    && pip3 install --upgrade \\
        wheel \\
        setuptools \\
        \"numpy<2\" \\
        virtualenv
"""
        else:
            df += """
# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update \\
      && apt-get install -y --no-install-recommends \\
            python3 \\
            libarchive-dev \\
            python3-pip \\
            python3-wheel \\
            python3-setuptools \\
            libpython3-dev \\
      && pip3 install --upgrade \\
            \"numpy<2\" \\
            virtualenv \\
      && rm -rf /var/lib/apt/lists/*
"""
    if "tensorrtllm" in backends:
        df += """
# Updating the openssh-client to fix for the CVE-2024-6387. This can be removed when trtllm uses a later CUDA container(12.5 or later)
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        openssh-client \\
    && rm -rf /var/lib/apt/lists/*
    """

    if "vllm" in backends:
        df += f"""
ARG BUILD_PUBLIC_VLLM="true"
ARG VLLM_INDEX_URL
ARG PYTORCH_TRITON_URL

RUN --mount=type=secret,id=req,target=/run/secrets/requirements \\
    if [ "$BUILD_PUBLIC_VLLM" = "false" ]; then \\
        pip3 install --no-cache-dir \\
        mkl==2021.1.1 \\
        mkl-include==2021.1.1 \\
        mkl-devel==2021.1.1 \\
        && pip3 install --no-cache-dir --progress-bar on --index-url $VLLM_INDEX_URL -r /run/secrets/requirements \\
        # Need to install in-house build of pytorch-triton to support triton_key definition used by torch 2.5.1
        && cd /tmp \\
        && wget $PYTORCH_TRITON_URL \\
        && pip install --no-cache-dir /tmp/pytorch_triton-*.whl \\
        && rm /tmp/pytorch_triton-*.whl; \\
    else \\
        # public vLLM needed for vLLM backend
        pip3 install vllm=={DEFAULT_TRITON_VERSION_MAP["vllm_version"]}; \\
    fi

ARG PYVER=3.12
ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/lib/python${{PYVER}}/dist-packages/torch/lib:${{LD_LIBRARY_PATH}}
"""

    if "dali" in backends:
        df += """
# Update Python path to include DALI
ENV PYTHONPATH=/opt/tritonserver/backends/dali/wheel/dali:$PYTHONPATH
"""

    df += """
WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/
"""

    # The CPU-only build uses ubuntu as the base image, and so the
    # entrypoint files are not available in /opt/nvidia in the base
    # image, so we must provide them ourselves.
    if not enable_gpu:
        df += """
COPY docker/cpu_only/ /opt/nvidia/
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
"""

    df += """
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
""".format(
        argmap["NVIDIA_BUILD_ID"], argmap["NVIDIA_BUILD_ID"], argmap["NVIDIA_BUILD_REF"]
    )
    return df


def add_cpu_libs_to_linux_dockerfile(backends, target_machine):
    df = ""
    libs_arch = "aarch64" if target_machine == "aarch64" else "x86_64"
    if "pytorch" in backends:
        # Add extra dependencies for pytorch backend.
        # Note: Even though the build is CPU-only, the version of pytorch
        # we are using depend upon libraries like cuda and cudnn. Since
        # these dependencies are not present in the ubuntu base image,
        # we must copy these from the Triton min container ourselves.
        cuda_arch = "sbsa" if target_machine == "aarch64" else "x86_64"
        df += """
RUN mkdir -p /usr/local/cuda/lib64/stubs
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusparse.so /usr/local/cuda/lib64/stubs/libcusparse.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusolver.so /usr/local/cuda/lib64/stubs/libcusolver.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcurand.so /usr/local/cuda/lib64/stubs/libcurand.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcufft.so /usr/local/cuda/lib64/stubs/libcufft.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublas.so /usr/local/cuda/lib64/stubs/libcublas.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.11

RUN mkdir -p /usr/local/cuda/targets/{cuda_arch}-linux/lib
COPY --from=min_container /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/targets/{cuda_arch}-linux/lib/.
COPY --from=min_container /usr/local/cuda/lib64/libcupti.so.12 /usr/local/cuda/targets/{cuda_arch}-linux/lib/.
COPY --from=min_container /usr/local/cuda/lib64/libnvToolsExt.so.1 /usr/local/cuda/targets/{cuda_arch}-linux/lib/.
COPY --from=min_container /usr/local/cuda/lib64/libnvJitLink.so.12 /usr/local/cuda/targets/{cuda_arch}-linux/lib/.

RUN mkdir -p /opt/hpcx/ucc/lib/ /opt/hpcx/ucx/lib/
COPY --from=min_container /opt/hpcx/ucc/lib/libucc.so.1 /opt/hpcx/ucc/lib/libucc.so.1
COPY --from=min_container /opt/hpcx/ucx/lib/libucm.so.0 /opt/hpcx/ucx/lib/libucm.so.0
COPY --from=min_container /opt/hpcx/ucx/lib/libucp.so.0 /opt/hpcx/ucx/lib/libucp.so.0
COPY --from=min_container /opt/hpcx/ucx/lib/libucs.so.0 /opt/hpcx/ucx/lib/libucs.so.0
COPY --from=min_container /opt/hpcx/ucx/lib/libuct.so.0 /opt/hpcx/ucx/lib/libuct.so.0

COPY --from=min_container /usr/lib/{libs_arch}-linux-gnu/libcudnn.so.9 /usr/lib/{libs_arch}-linux-gnu/libcudnn.so.9

# patchelf is needed to add deps of libcublasLt.so.12 to libtorch_cuda.so
RUN apt-get update \\
      && apt-get install -y --no-install-recommends openmpi-bin
RUN pip3 install patchelf==0.17.2

ENV LD_LIBRARY_PATH /usr/local/cuda/targets/{cuda_arch}-linux/lib:/usr/local/cuda/lib64/stubs:${{LD_LIBRARY_PATH}}
""".format(
            cuda_arch=cuda_arch, libs_arch=libs_arch
        )

    if ("pytorch" in backends) or ("tensorflow" in backends):
        # Add NCCL dependency for tensorflow/pytorch backend.
        # Note: Even though the build is CPU-only, the version of
        # tensorflow/pytorch we are using depends upon the NCCL library.
        # Since this dependency is not present in the ubuntu base image,
        # we must copy it from the Triton min container ourselves.
        df += """
COPY --from=min_container /usr/lib/{libs_arch}-linux-gnu/libnccl.so.2 /usr/lib/{libs_arch}-linux-gnu/libnccl.so.2
""".format(
            libs_arch=libs_arch
        )

    return df


def change_default_python_version_rhel(version):
    df = f"""
# The python library version available for install via 'yum install python3.X-devel' does not
# match the version of python inside the RHEL base container. This means that python packages
# installed within the container will not be picked up by the python backend stub process pybind
# bindings. It must instead must be installed via pyenv.
ENV PYENV_ROOT=/opt/pyenv_build
RUN curl https://pyenv.run | bash
ENV PATH="${{PYENV_ROOT}}/bin:$PATH"
RUN eval "$(pyenv init -)"
RUN CONFIGURE_OPTS=\"--with-openssl=/usr/lib64\" && pyenv install {version} \\
    && cp ${{PYENV_ROOT}}/versions/{version}/lib/libpython3* /usr/lib64/

# RHEL image has several python versions. It's important
# to set the correct version, otherwise, packages that are
# pip installed will not be found during testing.
ENV PYVER={version} PYTHONPATH=/opt/python/v
RUN ln -sf ${{PYENV_ROOT}}/versions/${{PYVER}}* ${{PYTHONPATH}}
ENV PYBIN=${{PYTHONPATH}}/bin
ENV PYTHON_BIN_PATH=${{PYBIN}}/python${{PYVER}} PATH=${{PYBIN}}:${{PATH}}
"""
    return df


def create_dockerfile_windows(
    ddir, dockerfile_name, argmap, backends, repoagents, caches
):
    df = """
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${{BASE_IMAGE}}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
LABEL com.nvidia.tritonserver.version="${{TRITON_SERVER_VERSION}}"

RUN setx path "%path%;C:\opt\tritonserver\bin"

""".format(
        argmap["TRITON_VERSION"],
        argmap["TRITON_CONTAINER_VERSION"],
        argmap["BASE_IMAGE"],
    )
    df += """
WORKDIR /opt
RUN rmdir /S/Q tritonserver || exit 0
COPY --chown=1000:1000 build/install tritonserver

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .

"""
    df += """
ENTRYPOINT []
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
""".format(
        argmap["NVIDIA_BUILD_ID"], argmap["NVIDIA_BUILD_ID"], argmap["NVIDIA_BUILD_REF"]
    )

    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_build_dockerfiles(
    container_build_dir, images, backends, repoagents, caches, endpoints
):
    if "base" in images:
        base_image = images["base"]
        if target_platform() == "rhel":
            print(
                "warning: RHEL is not an officially supported target and you will probably experience errors attempting to build this container."
            )
    elif target_platform() == "windows":
        base_image = "mcr.microsoft.com/dotnet/framework/sdk:4.8"
    elif target_platform() == "rhel":
        raise KeyError("A base image must be specified when targeting RHEL")
    elif FLAGS.enable_gpu:
        base_image = "nvcr.io/nvidia/tritonserver:{}-py3-min".format(
            FLAGS.upstream_container_version
        )
    else:
        base_image = "ubuntu:24.04"

    dockerfileargmap = {
        "NVIDIA_BUILD_REF": "" if FLAGS.build_sha is None else FLAGS.build_sha,
        "NVIDIA_BUILD_ID": "<unknown>" if FLAGS.build_id is None else FLAGS.build_id,
        "TRITON_VERSION": FLAGS.version,
        "TRITON_CONTAINER_VERSION": FLAGS.container_version,
        "BASE_IMAGE": base_image,
        "DCGM_VERSION": FLAGS.dcgm_version,
    }

    # For CPU-only image we need to copy some cuda libraries and dependencies
    # since we are using PyTorch and TensorFlow containers that
    # are not CPU-only.
    if (
        not FLAGS.enable_gpu
        and (("pytorch" in backends) or ("tensorflow" in backends))
        and (target_platform() != "windows")
    ):
        if "gpu-base" in images:
            gpu_base_image = images["gpu-base"]
        else:
            gpu_base_image = "nvcr.io/nvidia/tritonserver:{}-py3-min".format(
                FLAGS.upstream_container_version
            )
        dockerfileargmap["GPU_BASE_IMAGE"] = gpu_base_image

    if target_platform() == "rhel":
        create_dockerfile_buildbase_rhel(
            FLAGS.build_dir, "Dockerfile.buildbase", dockerfileargmap
        )
    else:
        create_dockerfile_buildbase(
            FLAGS.build_dir, "Dockerfile.buildbase", dockerfileargmap
        )

    if target_platform() == "windows":
        create_dockerfile_windows(
            FLAGS.build_dir,
            "Dockerfile",
            dockerfileargmap,
            backends,
            repoagents,
            caches,
        )
    else:
        create_dockerfile_linux(
            FLAGS.build_dir,
            "Dockerfile",
            dockerfileargmap,
            backends,
            repoagents,
            caches,
            endpoints,
        )

    # Dockerfile used for the creating the CI base image.
    create_dockerfile_cibase(FLAGS.build_dir, "Dockerfile.cibase", dockerfileargmap)


def create_docker_build_script(script_name, container_install_dir, container_ci_dir):
    with BuildScript(
        os.path.join(FLAGS.build_dir, script_name),
        verbose=FLAGS.verbose,
        desc=("Docker-based build script for Triton Inference Server"),
    ) as docker_script:
        #
        # Build base image... tritonserver_buildbase
        #
        docker_script.commentln(8)
        docker_script.comment("Create Triton base build image")
        docker_script.comment(
            "This image contains all dependencies necessary to build Triton"
        )
        docker_script.comment()

        cachefrommap = [
            "tritonserver_buildbase",
            "tritonserver_buildbase_cache0",
            "tritonserver_buildbase_cache1",
        ]

        baseargs = [
            "docker",
            "build",
            "-t",
            "tritonserver_buildbase",
            "-f",
            os.path.join(FLAGS.build_dir, "Dockerfile.buildbase"),
        ]

        if not FLAGS.no_container_pull:
            baseargs += [
                "--pull",
            ]

        # Windows docker runs in a VM and memory needs to be specified
        # explicitly (at least for some configurations of docker).
        if target_platform() == "windows":
            if FLAGS.container_memory:
                baseargs += ["--memory", FLAGS.container_memory]

        if target_platform() != "windows":
            baseargs += ["--cache-from={}".format(k) for k in cachefrommap]

        baseargs += ["."]

        docker_script.cwd(THIS_SCRIPT_DIR)
        docker_script.cmd(baseargs, check_exitcode=True)

        #
        # Build...
        #
        docker_script.blankln()
        docker_script.commentln(8)
        docker_script.comment("Run build in tritonserver_buildbase container")
        docker_script.comment("Mount a directory into the container where the install")
        docker_script.comment("artifacts will be placed.")
        docker_script.comment()

        # Don't use '-v' to communicate the built artifacts out of the
        # build, because we want this code to work even if run within
        # Docker (i.e. docker-in-docker) and not just if run directly
        # from host.
        runargs = [
            "docker",
            "run",
            "-w",
            "/workspace/build",
            "--name",
            "tritonserver_builder",
        ]

        if not FLAGS.no_container_interactive:
            runargs += ["-it"]

        if target_platform() == "windows":
            if FLAGS.container_memory:
                runargs += ["--memory", FLAGS.container_memory]
            runargs += ["-v", "\\\\.\pipe\docker_engine:\\\\.\pipe\docker_engine"]
        else:
            runargs += ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
            if FLAGS.use_user_docker_config:
                if os.path.exists(FLAGS.use_user_docker_config):
                    runargs += [
                        "-v",
                        os.path.expanduser(
                            FLAGS.use_user_docker_config + ":/root/.docker/config.json"
                        ),
                    ]

        runargs += ["tritonserver_buildbase"]

        if target_platform() == "windows":
            runargs += ["powershell.exe", "-noexit", "-File", "./cmake_build.ps1"]
        else:
            runargs += ["./cmake_build"]

        # Remove existing tritonserver_builder container...
        if target_platform() == "windows":
            docker_script.cmd(["docker", "rm", "tritonserver_builder"])
        else:
            docker_script._file.write(
                'if [ "$(docker ps -a | grep tritonserver_builder)" ]; then  docker rm -f tritonserver_builder; fi\n'
            )

        docker_script.cmd(runargs, check_exitcode=True)

        docker_script.cmd(
            [
                "docker",
                "cp",
                "tritonserver_builder:/tmp/tritonbuild/install",
                FLAGS.build_dir,
            ],
            check_exitcode=True,
        )
        docker_script.cmd(
            [
                "docker",
                "cp",
                "tritonserver_builder:/tmp/tritonbuild/ci",
                FLAGS.build_dir,
            ],
            check_exitcode=True,
        )

        #
        # Final image... tritonserver
        #
        docker_script.blankln()
        docker_script.commentln(8)
        docker_script.comment("Create final tritonserver image")
        docker_script.comment()

        finalargs = [
            "docker",
            "build",
        ]
        if secrets:
            finalargs += [
                f"--secret id=req,src={requirements}",
                f"--build-arg VLLM_INDEX_URL={vllm_index_url}",
                f"--build-arg PYTORCH_TRITON_URL={pytorch_triton_url}",
                f"--build-arg BUILD_PUBLIC_VLLM={build_public_vllm}",
            ]
        finalargs += [
            "-t",
            "tritonserver",
            "-f",
            os.path.join(FLAGS.build_dir, "Dockerfile"),
            ".",
        ]
        docker_script.cwd(THIS_SCRIPT_DIR)
        docker_script.cmd(finalargs, check_exitcode=True)

        #
        # CI base image... tritonserver_cibase
        #
        docker_script.blankln()
        docker_script.commentln(8)
        docker_script.comment("Create CI base image")
        docker_script.comment()

        cibaseargs = [
            "docker",
            "build",
            "-t",
            "tritonserver_cibase",
            "-f",
            os.path.join(FLAGS.build_dir, "Dockerfile.cibase"),
            ".",
        ]

        docker_script.cwd(THIS_SCRIPT_DIR)
        docker_script.cmd(cibaseargs, check_exitcode=True)


def core_build(
    cmake_script, repo_dir, cmake_dir, build_dir, install_dir, components, backends
):
    repo_build_dir = os.path.join(build_dir, "tritonserver", "build")
    repo_install_dir = os.path.join(build_dir, "tritonserver", "install")

    cmake_script.commentln(8)
    cmake_script.comment("Triton core library and tritonserver executable")
    cmake_script.comment()
    cmake_script.mkdir(repo_build_dir)
    cmake_script.cwd(repo_build_dir)
    cmake_script.cmake(
        core_cmake_args(components, backends, cmake_dir, repo_install_dir)
    )
    cmake_script.makeinstall()

    if target_platform() == "windows":
        cmake_script.mkdir(os.path.join(install_dir, "bin"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "bin", "tritonserver.exe"),
            os.path.join(install_dir, "bin"),
        )
        cmake_script.cp(
            os.path.join(repo_install_dir, "bin", "tritonserver.dll"),
            os.path.join(install_dir, "bin"),
        )
        cmake_script.cp(
            os.path.join(repo_install_dir, "lib", "tritonserver.lib"),
            os.path.join(install_dir, "bin"),
        )
    elif target_platform() == "rhel":
        cmake_script.mkdir(os.path.join(install_dir, "bin"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "bin", "tritonserver"),
            os.path.join(install_dir, "bin"),
        )
        cmake_script.mkdir(os.path.join(install_dir, "lib64"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "lib64", "libtritonserver.so"),
            os.path.join(install_dir, "lib64"),
        )
    else:
        cmake_script.mkdir(os.path.join(install_dir, "bin"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "bin", "tritonserver"),
            os.path.join(install_dir, "bin"),
        )
        cmake_script.mkdir(os.path.join(install_dir, "lib"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "lib", "libtritonserver.so"),
            os.path.join(install_dir, "lib"),
        )
    # [FIXME] Placing the tritonserver and tritonfrontend wheel files in 'python' for now,
    # should be uploaded to pip registry to be able to install directly
    cmake_script.mkdir(os.path.join(install_dir, "python"))
    cmake_script.cp(
        os.path.join(repo_install_dir, "python", "triton*.whl"),
        os.path.join(install_dir, "python"),
    )

    cmake_script.mkdir(os.path.join(install_dir, "include", "triton"))
    cmake_script.cpdir(
        os.path.join(repo_install_dir, "include", "triton", "core"),
        os.path.join(install_dir, "include", "triton", "core"),
    )

    cmake_script.cpdir(
        os.path.join(repo_dir, "python", "openai"), os.path.join(install_dir, "python")
    )

    cmake_script.cp(os.path.join(repo_dir, "LICENSE"), install_dir)
    cmake_script.cp(os.path.join(repo_dir, "TRITON_VERSION"), install_dir)

    # If requested, package the source code for all OSS used to build
    # For windows, Triton is not delivered as a container so skip for
    # windows platform.
    if target_platform() != "windows":
        if (
            (not FLAGS.no_container_build)
            and (not FLAGS.no_core_build)
            and (not FLAGS.no_container_source)
        ):
            cmake_script.mkdir(os.path.join(install_dir, "third-party-src"))
            cmake_script.cwd(repo_build_dir)
            cmake_script.tar(
                "third-party-src",
                os.path.join(install_dir, "third-party-src", "src.tar.gz"),
            )
            cmake_script.cp(
                os.path.join(repo_dir, "docker", "README.third-party-src"),
                os.path.join(install_dir, "third-party-src", "README"),
            )

    cmake_script.comment()
    cmake_script.comment("end Triton core library and tritonserver executable")
    cmake_script.commentln(8)
    cmake_script.blankln()


def tensorrtllm_prebuild(cmake_script):
    # Export the TRT_ROOT environment variable
    cmake_script.cmd("export TRT_ROOT=/usr/local/tensorrt")
    cmake_script.cmd("export ARCH=$(uname -m)")
    cmake_script.cmd(
        'export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH}"'
    )


def tensorrtllm_postbuild(cmake_script, repo_install_dir, tensorrtllm_be_dir):
    # TODO: Update the CMakeLists.txt of TRT-LLM backend to install the artifacts to the correct location
    cmake_destination_dir = os.path.join(repo_install_dir, "backends/tensorrtllm")
    cmake_script.mkdir(cmake_destination_dir)

    # Copy over the TRT-LLM backend libraries
    cmake_script.cp(
        os.path.join(tensorrtllm_be_dir, "build", "libtriton_tensorrtllm*.so"),
        cmake_destination_dir,
    )
    cmake_script.cp(
        os.path.join(tensorrtllm_be_dir, "build", "trtllmExecutorWorker"),
        cmake_destination_dir,
    )


def backend_build(
    be,
    cmake_script,
    tag,
    build_dir,
    install_dir,
    github_organization,
    images,
    components,
    library_paths,
):
    repo_build_dir = os.path.join(build_dir, be, "build")
    repo_install_dir = os.path.join(build_dir, be, "install")

    cmake_script.commentln(8)
    cmake_script.comment(f"'{be}' backend")
    cmake_script.comment("Delete this section to remove backend from build")
    cmake_script.comment()
    cmake_script.mkdir(build_dir)
    cmake_script.cwd(build_dir)
    cmake_script.gitclone(backend_repo(be), tag, be, github_organization)

    if be == "tensorrtllm":
        tensorrtllm_prebuild(cmake_script)

    cmake_script.mkdir(repo_build_dir)
    cmake_script.cwd(repo_build_dir)
    cmake_script.cmake(
        backend_cmake_args(images, components, be, repo_install_dir, library_paths)
    )
    cmake_script.makeinstall()

    if be == "tensorrtllm":
        tensorrtllm_be_dir = os.path.join(build_dir, be)
        tensorrtllm_postbuild(cmake_script, repo_install_dir, tensorrtllm_be_dir)

    cmake_script.mkdir(os.path.join(install_dir, "backends"))
    cmake_script.rmdir(os.path.join(install_dir, "backends", be))

    # The python library version available for install via 'yum install python3.X-devel' does not
    # match the version of python inside the RHEL base container. This means that python packages
    # installed within the container will not be picked up by the python backend stub process pybind
    # bindings. It must instead must be installed via pyenv. We package it here for better usability.
    if target_platform() == "rhel" and be == "python":
        major_minor_version = ".".join((FLAGS.rhel_py_version).split(".")[:2])
        version_matched_files = "/usr/lib64/libpython" + major_minor_version + "*"
        cmake_script.cp(
            version_matched_files, os.path.join(repo_install_dir, "backends", be)
        )

    cmake_script.cpdir(
        os.path.join(repo_install_dir, "backends", be),
        os.path.join(install_dir, "backends"),
    )

    cmake_script.comment()
    cmake_script.comment(f"end '{be}' backend")
    cmake_script.commentln(8)
    cmake_script.blankln()


def backend_clone(
    be,
    clone_script,
    tag,
    build_dir,
    install_dir,
    github_organization,
):
    clone_script.commentln(8)
    clone_script.comment(f"'{be}' backend")
    clone_script.comment("Delete this section to remove backend from build")
    clone_script.comment()
    clone_script.mkdir(build_dir)
    clone_script.cwd(build_dir)
    clone_script.gitclone(backend_repo(be), tag, be, github_organization)

    repo_target_dir = os.path.join(install_dir, "backends")
    clone_script.mkdir(repo_target_dir)
    backend_dir = os.path.join(repo_target_dir, be)
    clone_script.rmdir(backend_dir)
    clone_script.mkdir(backend_dir)

    clone_script.cp(
        os.path.join(build_dir, be, "src", "model.py"),
        backend_dir,
    )
    clone_script.cpdir(
        os.path.join(build_dir, be, "src", "utils"),
        backend_dir,
    )

    clone_script.comment()
    clone_script.comment(f"end '{be}' backend")
    clone_script.commentln(8)
    clone_script.blankln()


def repo_agent_build(
    ra, cmake_script, build_dir, install_dir, repoagent_repo, repoagents
):
    repo_build_dir = os.path.join(build_dir, ra, "build")
    repo_install_dir = os.path.join(build_dir, ra, "install")

    cmake_script.commentln(8)
    cmake_script.comment(f"'{ra}' repository agent")
    cmake_script.comment("Delete this section to remove repository agent from build")
    cmake_script.comment()
    cmake_script.mkdir(build_dir)
    cmake_script.cwd(build_dir)
    cmake_script.gitclone(
        repoagent_repo(ra), repoagents[ra], ra, FLAGS.github_organization
    )

    cmake_script.mkdir(repo_build_dir)
    cmake_script.cwd(repo_build_dir)
    cmake_script.cmake(repoagent_cmake_args(images, components, ra, repo_install_dir))
    cmake_script.makeinstall()

    cmake_script.mkdir(os.path.join(install_dir, "repoagents"))
    cmake_script.rmdir(os.path.join(install_dir, "repoagents", ra))
    cmake_script.cpdir(
        os.path.join(repo_install_dir, "repoagents", ra),
        os.path.join(install_dir, "repoagents"),
    )
    cmake_script.comment()
    cmake_script.comment(f"end '{ra}' repository agent")
    cmake_script.commentln(8)
    cmake_script.blankln()


def cache_build(cache, cmake_script, build_dir, install_dir, cache_repo, caches):
    repo_build_dir = os.path.join(build_dir, cache, "build")
    repo_install_dir = os.path.join(build_dir, cache, "install")

    cmake_script.commentln(8)
    cmake_script.comment(f"'{cache}' cache")
    cmake_script.comment("Delete this section to remove cache from build")
    cmake_script.comment()
    cmake_script.mkdir(build_dir)
    cmake_script.cwd(build_dir)
    cmake_script.gitclone(
        cache_repo(cache), caches[cache], cache, FLAGS.github_organization
    )

    cmake_script.mkdir(repo_build_dir)
    cmake_script.cwd(repo_build_dir)
    cmake_script.cmake(cache_cmake_args(images, components, cache, repo_install_dir))
    cmake_script.makeinstall()

    cmake_script.mkdir(os.path.join(install_dir, "caches"))
    cmake_script.rmdir(os.path.join(install_dir, "caches", cache))
    cmake_script.cpdir(
        os.path.join(repo_install_dir, "caches", cache),
        os.path.join(install_dir, "caches"),
    )
    cmake_script.comment()
    cmake_script.comment(f"end '{cache}' cache")
    cmake_script.commentln(8)
    cmake_script.blankln()


def cibase_build(
    cmake_script, repo_dir, cmake_dir, build_dir, install_dir, ci_dir, backends
):
    repo_install_dir = os.path.join(build_dir, "tritonserver", "install")

    cmake_script.commentln(8)
    cmake_script.comment("Collect Triton CI artifacts")
    cmake_script.comment()

    cmake_script.mkdir(ci_dir)

    # On windows we are not yet using a CI/QA docker image for
    # testing, so don't do anything...
    if target_platform() == "windows":
        return

    # The core build produces some artifacts that are needed for CI
    # testing, so include those in the install.
    cmake_script.cpdir(os.path.join(repo_dir, "qa"), ci_dir)
    cmake_script.cpdir(os.path.join(repo_dir, "deploy"), ci_dir)
    cmake_script.mkdir(os.path.join(ci_dir, "docs"))
    cmake_script.cpdir(
        os.path.join(repo_dir, "docs", "examples"), os.path.join(ci_dir, "docs")
    )
    cmake_script.mkdir(os.path.join(ci_dir, "src", "test"))
    cmake_script.cpdir(
        os.path.join(repo_dir, "src", "test", "models"),
        os.path.join(ci_dir, "src", "test"),
    )
    # Skip copying the artifacts in the bin, lib, and python as those directories will
    # be missing when the core build is not enabled.
    if not FLAGS.no_core_build:
        cmake_script.cpdir(os.path.join(repo_install_dir, "bin"), ci_dir)
        cmake_script.mkdir(os.path.join(ci_dir, "lib"))
        cmake_script.cp(
            os.path.join(repo_install_dir, "lib", "libtritonrepoagent_relocation.so"),
            os.path.join(ci_dir, "lib"),
        )
        cmake_script.cpdir(os.path.join(repo_install_dir, "python"), ci_dir)

    # Some of the backends are needed for CI testing
    cmake_script.mkdir(os.path.join(ci_dir, "backends"))
    for be in ("identity", "repeat", "square"):
        be_install_dir = os.path.join(build_dir, be, "install", "backends", be)
        if target_platform() == "windows":
            cmake_script.cmd(f"if (Test-Path -Path {be_install_dir}) {{")
        else:
            cmake_script.cmd(f"if [[ -e {be_install_dir} ]]; then")
        cmake_script.cpdir(be_install_dir, os.path.join(ci_dir, "backends"))
        cmake_script.cmd("}" if target_platform() == "windows" else "fi")

    # Some of the unit-test built backends are needed for CI testing
    cmake_script.mkdir(os.path.join(ci_dir, "tritonbuild", "tritonserver", "backends"))
    for be in (
        "query",
        "implicit_state",
        "sequence",
        "dyna_sequence",
        "distributed_addsub",
        "iterative_sequence",
    ):
        be_install_dir = os.path.join(repo_install_dir, "backends", be)
        if target_platform() == "windows":
            cmake_script.cmd(f"if (Test-Path -Path {be_install_dir}) {{")
        else:
            cmake_script.cmd(f"if [[ -e {be_install_dir} ]]; then")
        cmake_script.cpdir(
            be_install_dir,
            os.path.join(ci_dir, "tritonbuild", "tritonserver", "backends"),
        )
        cmake_script.cmd("}" if target_platform() == "windows" else "fi")

    # The onnxruntime_backend build produces some artifacts that
    # are needed for CI testing.
    if "onnxruntime" in backends:
        ort_install_dir = os.path.join(build_dir, "onnxruntime", "install")
        cmake_script.mkdir(os.path.join(ci_dir, "qa", "L0_custom_ops"))
        if target_platform() != "igpu":
            cmake_script.cp(
                os.path.join(ort_install_dir, "test", "libcustom_op_library.so"),
                os.path.join(ci_dir, "qa", "L0_custom_ops"),
            )
            cmake_script.cp(
                os.path.join(ort_install_dir, "test", "custom_op_test.onnx"),
                os.path.join(ci_dir, "qa", "L0_custom_ops"),
            )
        # [WIP] other way than wildcard?
        backend_tests = os.path.join(build_dir, "onnxruntime", "test", "*")
        cmake_script.cpdir(backend_tests, os.path.join(ci_dir, "qa"))

    # Need the build area for some backends so that they can be
    # rebuilt with specific options.
    cmake_script.mkdir(os.path.join(ci_dir, "tritonbuild"))
    for be in ("identity", "python"):
        if be in backends:
            cmake_script.rmdir(os.path.join(build_dir, be, "build"))
            cmake_script.rmdir(os.path.join(build_dir, be, "install"))
            cmake_script.cpdir(
                os.path.join(build_dir, be), os.path.join(ci_dir, "tritonbuild")
            )

    cmake_script.comment()
    cmake_script.comment("end Triton CI artifacts")
    cmake_script.commentln(8)
    cmake_script.blankln()


def finalize_build(cmake_script, install_dir, ci_dir):
    cmake_script.cmd(f"chmod -R a+rw {install_dir}")
    cmake_script.cmd(f"chmod -R a+rw {ci_dir}")


def enable_all():
    if target_platform() != "windows":
        all_backends = [
            "ensemble",
            "identity",
            "square",
            "repeat",
            "tensorflow",
            "onnxruntime",
            "python",
            "dali",
            "pytorch",
            "openvino",
            "fil",
            "tensorrt",
        ]
        all_repoagents = ["checksum"]
        all_caches = ["local", "redis"]
        all_filesystems = ["gcs", "s3", "azure_storage"]
        all_endpoints = ["http", "grpc", "sagemaker", "vertex-ai"]

        FLAGS.enable_logging = True
        FLAGS.enable_stats = True
        FLAGS.enable_metrics = True
        FLAGS.enable_gpu_metrics = True
        FLAGS.enable_cpu_metrics = True
        FLAGS.enable_tracing = True
        FLAGS.enable_nvtx = True
        FLAGS.enable_gpu = True
    else:
        all_backends = [
            "ensemble",
            "identity",
            "square",
            "repeat",
            "onnxruntime",
            "openvino",
            "tensorrt",
        ]
        all_repoagents = ["checksum"]
        all_caches = ["local", "redis"]
        all_filesystems = []
        all_endpoints = ["http", "grpc"]

        FLAGS.enable_logging = True
        FLAGS.enable_stats = True
        FLAGS.enable_tracing = True
        FLAGS.enable_gpu = True

    requested_backends = []
    for be in FLAGS.backend:
        parts = be.split(":")
        requested_backends += [parts[0]]
    for be in all_backends:
        if be not in requested_backends:
            FLAGS.backend += [be]

    requested_repoagents = []
    for ra in FLAGS.repoagent:
        parts = ra.split(":")
        requested_repoagents += [parts[0]]
    for ra in all_repoagents:
        if ra not in requested_repoagents:
            FLAGS.repoagent += [ra]

    requested_caches = []
    for cache in FLAGS.cache:
        parts = cache.split(":")
        requested_caches += [parts[0]]
    for cache in all_caches:
        if cache not in requested_caches:
            FLAGS.cache += [cache]

    for fs in all_filesystems:
        if fs not in FLAGS.filesystem:
            FLAGS.filesystem += [fs]

    for ep in all_endpoints:
        if ep not in FLAGS.endpoint:
            FLAGS.endpoint += [ep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        required=False,
        help="Disable console output.",
    )
    group_qv.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Enable verbose output.",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        required=False,
        help="Output the build scripts, but do not perform build.",
    )
    parser.add_argument(
        "--no-container-build",
        action="store_true",
        required=False,
        help="Do not use Docker container for build.",
    )
    parser.add_argument(
        "--use-user-docker-config",
        default=None,
        required=False,
        help="Path to the Docker configuration file to be used when performing container build.",
    )
    parser.add_argument(
        "--no-container-interactive",
        action="store_true",
        required=False,
        help='Do not use -it argument to "docker run" when performing container build.',
    )
    parser.add_argument(
        "--no-container-pull",
        action="store_true",
        required=False,
        help="Do not use Docker --pull argument when building container.",
    )
    parser.add_argument(
        "--container-memory",
        default=None,
        required=False,
        help="Value for Docker --memory argument. Used only for windows builds.",
    )
    parser.add_argument(
        "--target-platform",
        required=False,
        default=None,
        help='Target platform for build, can be "linux", "rhel", "windows" or "igpu". If not specified, build targets the current platform.',
    )
    parser.add_argument(
        "--target-machine",
        required=False,
        default=None,
        help="Target machine/architecture for build. If not specified, build targets the current machine/architecture.",
    )

    parser.add_argument(
        "--build-id",
        type=str,
        required=False,
        help="Build ID associated with the build.",
    )
    parser.add_argument(
        "--build-sha", type=str, required=False, help="SHA associated with the build."
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        required=False,
        help="Build directory. All repo clones and builds will be performed in this directory.",
    )
    parser.add_argument(
        "--install-dir",
        type=str,
        required=False,
        default=None,
        help="Install directory, default is <builddir>/opt/tritonserver.",
    )
    parser.add_argument(
        "--cmake-dir",
        type=str,
        required=False,
        help="Directory containing the CMakeLists.txt file for Triton server.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        required=False,
        default="/tmp",
        help="Temporary directory used for building inside docker. Default is /tmp.",
    )
    parser.add_argument(
        "--library-paths",
        action="append",
        required=False,
        default=None,
        help="Specify library paths for respective backends in build as <backend-name>[:<library_path>].",
    )
    parser.add_argument(
        "--build-type",
        required=False,
        default="Release",
        help='Build type, one of "Release", "Debug", "RelWithDebInfo" or "MinSizeRel". Default is "Release".',
    )
    parser.add_argument(
        "-j",
        "--build-parallel",
        type=int,
        required=False,
        default=None,
        help="Build parallelism. Defaults to 2 * number-of-cores.",
    )

    parser.add_argument(
        "--github-organization",
        type=str,
        required=False,
        default="https://github.com/triton-inference-server",
        help='The GitHub organization containing the repos used for the build. Defaults to "https://github.com/triton-inference-server".',
    )
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        help="The Triton version. If not specified defaults to the value in the TRITON_VERSION file.",
    )
    parser.add_argument(
        "--container-version",
        type=str,
        required=False,
        help="The Triton container version to build. If not specified the container version will be chosen automatically based on --version value.",
    )
    parser.add_argument(
        "--container-prebuild-command",
        type=str,
        required=False,
        help="When performing a container build, this command will be executed within the container just before the build it performed.",
    )
    parser.add_argument(
        "--no-container-source",
        action="store_true",
        required=False,
        help="Do not include OSS source code in Docker container.",
    )
    parser.add_argument(
        "--image",
        action="append",
        required=False,
        help='Use specified Docker image in build as <image-name>,<full-image-name>. <image-name> can be "base", "gpu-base", "tensorflow", or "pytorch".',
    )

    parser.add_argument(
        "--enable-all",
        action="store_true",
        required=False,
        help="Enable all standard released Triton features, backends, repository agents, caches, endpoints and file systems.",
    )
    parser.add_argument(
        "--enable-logging", action="store_true", required=False, help="Enable logging."
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        required=False,
        help="Enable statistics collection.",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        required=False,
        help="Enable metrics reporting.",
    )
    parser.add_argument(
        "--enable-gpu-metrics",
        action="store_true",
        required=False,
        help="Include GPU metrics in reported metrics.",
    )
    parser.add_argument(
        "--enable-cpu-metrics",
        action="store_true",
        required=False,
        help="Include CPU metrics in reported metrics.",
    )
    parser.add_argument(
        "--enable-tracing", action="store_true", required=False, help="Enable tracing."
    )
    parser.add_argument(
        "--enable-nvtx", action="store_true", required=False, help="Enable NVTX."
    )
    parser.add_argument(
        "--enable-gpu", action="store_true", required=False, help="Enable GPU support."
    )
    parser.add_argument(
        "--enable-mali-gpu",
        action="store_true",
        required=False,
        help="Enable ARM MALI GPU support.",
    )
    parser.add_argument(
        "--min-compute-capability",
        type=str,
        required=False,
        default="6.0",
        help="Minimum CUDA compute capability supported by server.",
    )

    parser.add_argument(
        "--endpoint",
        action="append",
        required=False,
        help='Include specified endpoint in build. Allowed values are "grpc", "http", "vertex-ai" and "sagemaker".',
    )
    parser.add_argument(
        "--filesystem",
        action="append",
        required=False,
        help='Include specified filesystem in build. Allowed values are "gcs", "azure_storage" and "s3".',
    )
    parser.add_argument(
        "--no-core-build",
        action="store_true",
        required=False,
        help="Do not build Triton core shared library or executable.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        required=False,
        help='Include specified backend in build as <backend-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version YY.MM -> branch rYY.MM); otherwise the default <repo-tag> is "main" (e.g. version YY.MMdev -> branch main).',
    )
    parser.add_argument(
        "--repo-tag",
        action="append",
        required=False,
        help='The version of a component to use in the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", "backend" or "thirdparty". <repo-tag> indicates the git tag/branch to use for the build. Currently <repo-tag> does not support pull-request reference. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version YY.MM -> branch rYY.MM); otherwise the default <repo-tag> is "main" (e.g. version YY.MMdev -> branch main).',
    )
    parser.add_argument(
        "--repoagent",
        action="append",
        required=False,
        help='Include specified repo agent in build as <repoagent-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version YY.MM -> branch rYY.MM); otherwise the default <repo-tag> is "main" (e.g. version YY.MMdev -> branch main).',
    )
    parser.add_argument(
        "--cache",
        action="append",
        required=False,
        help='Include specified cache in build as <cache-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version YY.MM -> branch rYY.MM); otherwise the default <repo-tag> is "main" (e.g. version YY.MMdev -> branch main).',
    )
    parser.add_argument(
        "--no-force-clone",
        action="store_true",
        default=False,
        help="Do not create fresh clones of repos that have already been cloned.",
    )
    parser.add_argument(
        "--extra-core-cmake-arg",
        action="append",
        required=False,
        help="Extra CMake argument as <name>=<value>. The argument is passed to CMake as -D<name>=<value> and is included after all CMake arguments added by build.py for the core builds.",
    )
    parser.add_argument(
        "--override-core-cmake-arg",
        action="append",
        required=False,
        help="Override specified CMake argument in the build as <name>=<value>. The argument is passed to CMake as -D<name>=<value>. This flag only impacts CMake arguments that are used by build.py. To unconditionally add a CMake argument to the core build use --extra-core-cmake-arg.",
    )
    parser.add_argument(
        "--extra-backend-cmake-arg",
        action="append",
        required=False,
        help="Extra CMake argument for a backend build as <backend>:<name>=<value>. The argument is passed to CMake as -D<name>=<value> and is included after all CMake arguments added by build.py for the backend.",
    )
    parser.add_argument(
        "--override-backend-cmake-arg",
        action="append",
        required=False,
        help="Override specified backend CMake argument in the build as <backend>:<name>=<value>. The argument is passed to CMake as -D<name>=<value>. This flag only impacts CMake arguments that are used by build.py. To unconditionally add a CMake argument to the backend build use --extra-backend-cmake-arg.",
    )
    parser.add_argument(
        "--release-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["release_version"],
        help="This flag sets the release version for Triton Inference Server to be built. Default: the latest released version.",
    )
    parser.add_argument(
        "--triton-container-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["triton_container_version"],
        help="This flag sets the container version for Triton Inference Server to be built. Default: the latest released version.",
    )
    parser.add_argument(
        "--upstream-container-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["upstream_container_version"],
        help="This flag sets the upstream container version for Triton Inference Server to be built. Default: the latest released version.",
    )
    parser.add_argument(
        "--ort-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["ort_version"],
        help="This flag sets the ORT version for Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--ort-openvino-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["ort_openvino_version"],
        help="This flag sets the OpenVino version for Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--standalone-openvino-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["standalone_openvino_version"],
        help="This flag sets the standalon OpenVino version for Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--dcgm-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["dcgm_version"],
        help="This flag sets the DCGM version for Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--vllm-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["vllm_version"],
        help="This flag sets the vLLM version for Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--rhel-py-version",
        required=False,
        default=DEFAULT_TRITON_VERSION_MAP["rhel_py_version"],
        help="This flag sets the Python version for RHEL platform of Triton Inference Server to be built. Default: the latest supported version.",
    )
    parser.add_argument(
        "--build-secret",
        action="append",
        required=False,
        nargs=2,
        metavar=("key", "value"),
        help="Add build secrets in the form of <key> <value>. These secrets are used during the build process for vllm. The secrets are passed to the Docker build step as `--secret id=<key>`. The following keys are expected and their purposes are described below:\n\n"
        "  - 'req': A file containing a list of dependencies for pip (e.g., requirements.txt).\n"
        "  - 'vllm_index_url': The index URL for the pip install.\n"
        "  - 'pytorch_triton_url': The location of the PyTorch wheel to download.\n"
        "  - 'build_public_vllm': A flag (default is 'true') indicating whether to build the public VLLM version.\n\n"
        "Ensure that the required environment variables for these secrets are set before running the build.",
    )
    FLAGS = parser.parse_args()

    if FLAGS.image is None:
        FLAGS.image = []
    if FLAGS.repo_tag is None:
        FLAGS.repo_tag = []
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.endpoint is None:
        FLAGS.endpoint = []
    if FLAGS.filesystem is None:
        FLAGS.filesystem = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []
    if FLAGS.cache is None:
        FLAGS.cache = []
    if FLAGS.library_paths is None:
        FLAGS.library_paths = []
    if FLAGS.extra_core_cmake_arg is None:
        FLAGS.extra_core_cmake_arg = []
    if FLAGS.override_core_cmake_arg is None:
        FLAGS.override_core_cmake_arg = []
    if FLAGS.override_backend_cmake_arg is None:
        FLAGS.override_backend_cmake_arg = []
    if FLAGS.extra_backend_cmake_arg is None:
        FLAGS.extra_backend_cmake_arg = []
    if FLAGS.build_secret is None:
        FLAGS.build_secret = []

    # if --enable-all is specified, then update FLAGS to enable all
    # settings, backends, repo-agents, caches, file systems, endpoints, etc.
    if FLAGS.enable_all:
        enable_all()

    # When doing a docker build, --build-dir, --install-dir and
    # --cmake-dir must not be set. We will use the build/ subdir
    # within the server/ repo that contains this build.py script for
    # --build-dir. If not doing a docker build, --build-dir must be
    # set.
    if FLAGS.no_container_build:
        if FLAGS.build_dir is None:
            fail("--no-container-build requires --build-dir")
        if FLAGS.install_dir is None:
            FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")
        if FLAGS.cmake_dir is None:
            FLAGS.cmake_dir = THIS_SCRIPT_DIR
    else:
        if FLAGS.build_dir is not None:
            fail("--build-dir must not be set for container-based build")
        if FLAGS.install_dir is not None:
            fail("--install-dir must not be set for container-based build")
        if FLAGS.cmake_dir is not None:
            fail("--cmake-dir must not be set for container-based build")
        FLAGS.build_dir = os.path.join(THIS_SCRIPT_DIR, "build")

    # Determine the versions. Start with Triton version, if --version
    # is not explicitly specified read from TRITON_VERSION file.
    if FLAGS.version is None:
        FLAGS.version = DEFAULT_TRITON_VERSION_MAP["release_version"]

    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2

    log("Building Triton Inference Server")
    log("platform {}".format(target_platform()))
    log("machine {}".format(target_machine()))
    log("version {}".format(FLAGS.version))
    log("build dir {}".format(FLAGS.build_dir))
    log("install dir {}".format(FLAGS.install_dir))
    log("cmake dir {}".format(FLAGS.cmake_dir))

    # Determine the default repo-tag that should be used for images,
    # backends, repo-agents, and caches if a repo-tag is not given
    # explicitly. For release branches we use the release branch as
    # the default, otherwise we use 'main'.
    default_repo_tag = "main"
    cver = FLAGS.upstream_container_version
    if cver is None:
        cver = FLAGS.triton_container_version
    if not cver.endswith("dev"):
        default_repo_tag = "r" + cver
    log("default repo-tag: {}".format(default_repo_tag))

    # For other versions use the TRITON_VERSION_MAP unless explicitly
    # given.
    FLAGS.container_version, FLAGS.upstream_container_version = container_versions(
        FLAGS.version, FLAGS.container_version, FLAGS.upstream_container_version
    )

    log("container version {}".format(FLAGS.container_version))
    log("upstream container version {}".format(FLAGS.upstream_container_version))

    for ep in FLAGS.endpoint:
        log(f'endpoint "{ep}"')
    for fs in FLAGS.filesystem:
        log(f'filesystem "{fs}"')

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    for be in FLAGS.backend:
        parts = be.split(":")
        if len(parts) == 1:
            parts.append(default_repo_tag)
        if parts[0] == "tensorflow1":
            fail(
                "Starting from Triton version 23.04, support for TensorFlow 1 has been discontinued. Please switch to Tensorflow 2."
            )
        if parts[0] == "tensorflow2":
            parts[0] = "tensorflow"
        log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        backends[parts[0]] = parts[1]

    if "vllm" in backends:
        if "python" not in backends:
            log(
                "vLLM backend requires Python backend, adding Python backend with tag {}".format(
                    backends["vllm"]
                )
            )
            backends["python"] = backends["vllm"]

    secrets = dict(getattr(FLAGS, "build_secret", []))
    if secrets:
        requirements = secrets.get("req", "")
        vllm_index_url = secrets.get("vllm_index_url", "")
        pytorch_triton_url = secrets.get("pytorch_triton_url", "")
        build_public_vllm = secrets.get("build_public_vllm", "true")
        log('Build Arg for BUILD_PUBLIC_VLLM: "{}"'.format(build_public_vllm))

    # Initialize map of repo agents to build and repo-tag for each.
    repoagents = {}
    for be in FLAGS.repoagent:
        parts = be.split(":")
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('repoagent "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        repoagents[parts[0]] = parts[1]

    # Initialize map of caches to build and repo-tag for each.
    caches = {}
    for be in FLAGS.cache:
        parts = be.split(":")
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('cache "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        caches[parts[0]] = parts[1]

    # Initialize map of docker images.
    images = {}
    for img in FLAGS.image:
        parts = img.split(",")
        fail_if(
            len(parts) != 2, "--image must specify <image-name>,<full-image-registry>"
        )
        fail_if(
            parts[0]
            not in ["base", "gpu-base", "pytorch", "tensorflow", "tensorflow2"],
            "unsupported value for --image",
        )
        log('image "{}": "{}"'.format(parts[0], parts[1]))
        if parts[0] == "tensorflow2":
            parts[0] = "tensorflow"
        images[parts[0]] = parts[1]

    # Initialize map of library paths for each backend.
    library_paths = {}
    for lpath in FLAGS.library_paths:
        parts = lpath.split(":")
        if len(parts) == 2:
            log('backend "{}" library path "{}"'.format(parts[0], parts[1]))
            if parts[0] == "tensorflow2":
                parts[0] = "tensorflow"
            library_paths[parts[0]] = parts[1]

    # Parse any explicitly specified cmake arguments
    for cf in FLAGS.extra_core_cmake_arg:
        parts = cf.split("=")
        fail_if(len(parts) != 2, "--extra-core-cmake-arg must specify <name>=<value>")
        log('CMake core extra "-D{}={}"'.format(parts[0], parts[1]))
        EXTRA_CORE_CMAKE_FLAGS[parts[0]] = parts[1]

    for cf in FLAGS.override_core_cmake_arg:
        parts = cf.split("=")
        fail_if(
            len(parts) != 2, "--override-core-cmake-arg must specify <name>=<value>"
        )
        log('CMake core override "-D{}={}"'.format(parts[0], parts[1]))
        OVERRIDE_CORE_CMAKE_FLAGS[parts[0]] = parts[1]

    for cf in FLAGS.extra_backend_cmake_arg:
        parts = cf.split(":", 1)
        fail_if(
            len(parts) != 2,
            "--extra-backend-cmake-arg must specify <backend>:<name>=<value>",
        )
        be = parts[0]
        parts = parts[1].split("=", 1)
        fail_if(
            len(parts) != 2,
            "--extra-backend-cmake-arg must specify <backend>:<name>=<value>",
        )
        fail_if(
            be not in backends,
            '--extra-backend-cmake-arg specifies backend "{}" which is not included in build'.format(
                be
            ),
        )
        log('backend "{}" CMake extra "-D{}={}"'.format(be, parts[0], parts[1]))
        if be not in EXTRA_BACKEND_CMAKE_FLAGS:
            EXTRA_BACKEND_CMAKE_FLAGS[be] = {}
        EXTRA_BACKEND_CMAKE_FLAGS[be][parts[0]] = parts[1]

    for cf in FLAGS.override_backend_cmake_arg:
        parts = cf.split(":", 1)
        fail_if(
            len(parts) != 2,
            "--override-backend-cmake-arg must specify <backend>:<name>=<value>",
        )
        be = parts[0]
        parts = parts[1].split("=", 1)
        fail_if(
            len(parts) != 2,
            "--override-backend-cmake-arg must specify <backend>:<name>=<value>",
        )
        fail_if(
            be not in backends,
            '--override-backend-cmake-arg specifies backend "{}" which is not included in build'.format(
                be
            ),
        )
        log('backend "{}" CMake override "-D{}={}"'.format(be, parts[0], parts[1]))
        if be not in OVERRIDE_BACKEND_CMAKE_FLAGS:
            OVERRIDE_BACKEND_CMAKE_FLAGS[be] = {}
        OVERRIDE_BACKEND_CMAKE_FLAGS[be][parts[0]] = parts[1]

    # Initialize map of common components and repo-tag for each.
    components = {
        "common": default_repo_tag,
        "core": default_repo_tag,
        "backend": default_repo_tag,
        "thirdparty": default_repo_tag,
    }
    for be in FLAGS.repo_tag:
        parts = be.split(":")
        fail_if(len(parts) != 2, "--repo-tag must specify <component-name>:<repo-tag>")
        fail_if(
            parts[0] not in components,
            '--repo-tag <component-name> must be "common", "core", "backend", or "thirdparty"',
        )
        components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Set the build, install, and cmake directories to use for the
    # generated build scripts and Dockerfiles. If building without
    # Docker, these are the directories specified on the cmdline. If
    # building with Docker, we change these to be directories within
    # FLAGS.tmp_dir inside the Docker container.
    script_repo_dir = THIS_SCRIPT_DIR
    script_build_dir = FLAGS.build_dir
    script_install_dir = script_ci_dir = FLAGS.install_dir
    script_cmake_dir = FLAGS.cmake_dir
    if not FLAGS.no_container_build:
        # FLAGS.tmp_dir may be specified with "\" on Windows, adjust
        # to "/" for docker usage.
        script_build_dir = os.path.normpath(
            os.path.join(FLAGS.tmp_dir, "tritonbuild").replace("\\", "/")
        )
        script_install_dir = os.path.normpath(os.path.join(script_build_dir, "install"))
        script_ci_dir = os.path.normpath(os.path.join(script_build_dir, "ci"))
        if target_platform() == "windows":
            script_repo_dir = script_cmake_dir = os.path.normpath("c:/workspace")
        else:
            script_repo_dir = script_cmake_dir = "/workspace"

    script_name = "cmake_build"
    if target_platform() == "windows":
        script_name += ".ps1"

    # Write the build script that invokes cmake for the core, backends, repo-agents, and caches.
    pathlib.Path(FLAGS.build_dir).mkdir(parents=True, exist_ok=True)
    with BuildScript(
        os.path.join(FLAGS.build_dir, script_name),
        verbose=FLAGS.verbose,
        desc=("Build script for Triton Inference Server"),
    ) as cmake_script:
        # Run the container pre-build command if the cmake build is
        # being done within the build container.
        if not FLAGS.no_container_build and FLAGS.container_prebuild_command:
            cmake_script.cmd(FLAGS.container_prebuild_command, check_exitcode=True)
            cmake_script.blankln()

        # Commands to build the core shared library and the server executable.
        if not FLAGS.no_core_build:
            core_build(
                cmake_script,
                script_repo_dir,
                script_cmake_dir,
                script_build_dir,
                script_install_dir,
                components,
                backends,
            )

        # Commands to build each backend...
        for be in backends:
            # Core backends are not built separately from core so skip...
            if be in CORE_BACKENDS:
                continue

            # If armnn_tflite backend, source from external repo for git clone
            if be == "armnn_tflite":
                github_organization = "https://gitlab.com/arm-research/smarter/"
            else:
                github_organization = FLAGS.github_organization

            if be == "vllm":
                backend_clone(
                    be,
                    cmake_script,
                    backends[be],
                    script_build_dir,
                    script_install_dir,
                    github_organization,
                )
            else:
                backend_build(
                    be,
                    cmake_script,
                    backends[be],
                    script_build_dir,
                    script_install_dir,
                    github_organization,
                    images,
                    components,
                    library_paths,
                )

        # Commands to build each repo agent...
        for ra in repoagents:
            repo_agent_build(
                ra,
                cmake_script,
                script_build_dir,
                script_install_dir,
                repoagent_repo,
                repoagents,
            )

        # Commands to build each cache...
        for cache in caches:
            cache_build(
                cache,
                cmake_script,
                script_build_dir,
                script_install_dir,
                cache_repo,
                caches,
            )

        # Commands needed only when building with Docker...
        if not FLAGS.no_container_build:
            # Commands to collect all the build artifacts needed for CI
            # testing.
            cibase_build(
                cmake_script,
                script_repo_dir,
                script_cmake_dir,
                script_build_dir,
                script_install_dir,
                script_ci_dir,
                backends,
            )

            # When building with Docker the install and ci artifacts
            # written to the build-dir while running the docker container
            # may have root ownership, so give them permissions to be
            # managed by all users on the host system.
            if target_platform() != "windows":
                finalize_build(cmake_script, script_install_dir, script_ci_dir)

    # If --no-container-build is not specified then we perform the
    # actual build within a docker container and from that create the
    # final tritonserver docker image. For the build we need to
    # generate a few Dockerfiles and a top-level script that drives
    # the build process.
    if not FLAGS.no_container_build:
        script_name = "docker_build"
        if target_platform() == "windows":
            script_name += ".ps1"

        create_build_dockerfiles(
            script_build_dir, images, backends, repoagents, caches, FLAGS.endpoint
        )
        create_docker_build_script(script_name, script_install_dir, script_ci_dir)

    # In not dry-run, execute the script to perform the build...  If a
    # container-based build is requested use 'docker_build' script,
    # otherwise build directly on this system using cmake script.
    if not FLAGS.dryrun:
        if target_platform() == "windows":
            p = subprocess.Popen(
                ["powershell.exe", "-noexit", "-File", f"./{script_name}"],
                cwd=FLAGS.build_dir,
            )
        else:
            p = subprocess.Popen([f"./{script_name}"], cwd=FLAGS.build_dir)
        p.wait()
        fail_if(p.returncode != 0, "build failed")
