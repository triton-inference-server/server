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

"""The CPU and GPUs a run sees, for the report header and the JUnit report.

The same scenario runs on different machines (A100, R100, VR200; x86 and
aarch64), and a CI allocation can be smaller than the node, so the run
records what it actually got. Nothing here raises: a probe that fails is
reported as unknown, so a machine without nvidia-smi still runs the tests.
"""

import os
import platform
import subprocess

NVIDIA_SMI_TIMEOUT_S = 30
GPU_FIELDS = (
    "index",
    "name",
    "memory.total",
    "compute_cap",
    "driver_version",
    "pci.bus_id",
)


def _cpu_model():
    # x86 names its CPU in /proc/cpuinfo; aarch64 kernels usually do not, but
    # lscpu reports a model name for both.
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in out.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def cpu_info():
    """{arch, model, cpus}; cpus is what this process may run on (the
    container's or Slurm's allocation), followed by the host's count."""
    try:
        usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable = None
    return {
        "arch": platform.machine(),
        "model": _cpu_model(),
        "cpus": "%s of %s" % (usable or "unknown", os.cpu_count() or "unknown"),
    }


def gpu_info():
    """(list of per-GPU dicts keyed by GPU_FIELDS, error or None)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(GPU_FIELDS),
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [], "nvidia-smi failed: %s" % e
    gpus = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(GPU_FIELDS):
            gpus.append(dict(zip(GPU_FIELDS, values)))
    return gpus, None


def describe():
    """Lines for the pytest report header and (name, value) pairs for the
    JUnit report, computed once."""
    cpu = cpu_info()
    gpus, error = gpu_info()
    lines = ["cpu: %(arch)s, %(model)s, %(cpus)s CPUs usable" % cpu]
    properties = [
        ("cpu_arch", cpu["arch"]),
        ("cpu_model", cpu["model"]),
        ("cpus", cpu["cpus"]),
        ("gpu_count", str(len(gpus))),
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        lines.append("CUDA_VISIBLE_DEVICES=%s" % visible)
        properties.append(("cuda_visible_devices", visible))
    if error:
        lines.append("gpus: unknown (%s)" % error)
        properties.append(("gpu_error", error))
    else:
        lines.append(
            "gpus: %d, driver %s"
            % (len(gpus), gpus[0]["driver_version"] if gpus else "none")
        )
        for gpu in gpus:
            lines.append(
                "  gpu %(index)s: %(name)s, %(memory.total)s, "
                "compute capability %(compute_cap)s, %(pci.bus_id)s" % gpu
            )
            properties.append(
                (
                    "gpu%s" % gpu["index"],
                    "%(name)s, %(memory.total)s, cc %(compute_cap)s" % gpu,
                )
            )
        if gpus:
            properties.append(("gpu_driver", gpus[0]["driver_version"]))
    return lines, properties
