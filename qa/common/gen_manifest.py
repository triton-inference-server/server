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

"""Build and maintain per-model manifest.json files.

A manifest records what produced a model and how large it is, so a consumer can
decide what to fetch without fetching it. One file per model directory, beside
config.pbtxt.

Used two ways.

**From a generator script**, the same way gen_common is used. Fingerprint the
tree before generating, then emit manifests for whatever the script touched:

    import gen_manifest

    if __name__ == "__main__":
        ...
        manifest_baseline = gen_manifest.snapshot_model_dirs(FLAGS.models_dir)
        ...generate models...
        gen_manifest.emit_manifests(FLAGS.models_dir, manifest_baseline)

The baseline is what keeps each script honest. Every stage writes into the same
repositories -- qa_model_repository receives models from all four -- so a script
that stamped the whole tree would relabel models built by the stage before it
with its own image and framework. Comparing against the baseline restricts the
write to directories this script actually created or added to.

**From the command line**, to stamp or re-measure a finished tree:

    python3 gen_manifest.py --tree /tmp/26.08 --image ubuntu:24.04   # phase 1
    python3 gen_manifest.py --tree /tmp/26.08 --update-sizes         # phase 2

Sizes are written in a second pass because a model directory receives files from
more than one stage: the size recorded by the stage that created a model is only
final once every later stage has run.

Either way it reports one line per model as it goes -- a full tree is ~976 models
and the pass is not instantaneous, so a silent run is indistinguishable from a
hung one. `--quiet` keeps just the summary line, and `--summary PATH` writes the
whole pass as JSON so CI can assert on counts and totals rather than parsing them
back out of log text.

Standard library only. This is imported inside bare containers where the only
third-party packages installed are the framework being generated for.
"""

import argparse
import collections
import functools
import io
import json
import os
import pathlib
import platform
import re
import struct
import subprocess
import sys
import zipfile

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "2.0"
KIND = "triton-qa-model"

# Set by the driver, per stage. None of these is discoverable from inside the
# container, so an unset variable means the field is recorded as null rather
# than guessed at.
ENV_IMAGE = "TRITON_MODEL_GEN_IMAGE"
ENV_FRAMEWORK = "TRITON_MODEL_GEN_FRAMEWORK"
ENV_RUNTIME = "TRITON_MODEL_GEN_RUNTIME"
ENV_SEMVER = "TRITON_SEMVER"
ENV_VERBOSE = "TRITON_MODEL_GEN_VERBOSE"
ENV_UPSTREAM = "NVIDIA_UPSTREAM_VERSION"
# TRITON_VERSION used to be read here as the container train. It was not one:
# the driver defaulted it to the train, CI exported it as the semver, so this
# field recorded the semver on every manifest CI produced. Each version now has
# a variable carrying one meaning, and none of them is TRITON_VERSION.

KIB = 1024
MIB = 1024**2
GIB = 1024**3

# Size tiers, as documented in README.md. Boundaries sit on real gaps in the
# distribution: 88% of models are config-only, and a handful hold ~96% of bytes.
SIZE_TIERS = (
    (10 * KIB, "xs"),
    (1 * MIB, "s"),
    (100 * MIB, "m"),
    (1 * GIB, "l"),
)
LARGEST_TIER = "xl"

# config.pbtxt declares a backend two different ways. Normalise to the BACKENDS
# vocabulary used by qa/L0_*/test.sh where one exists, else the backend's own
# name.
PLATFORM_TO_BACKEND = {
    "tensorrt_plan": "plan",
    "pytorch_libtorch": "libtorch",
    # AOTInductor models declare platform: torch_aoti and backend: pytorch. They
    # are served by the pytorch backend, so provenance is libtorch; the verbatim
    # platform field keeps the distinction, which matters because AOTI artifacts
    # embed device code and are not portable across compute capabilities.
    "torch_aoti": "libtorch",
    "onnxruntime_onnx": "onnx",
    "caffe2_netdef": "netdef",
    "ensemble": "ensemble",
}
BACKEND_TO_BACKEND = {
    "tensorrt": "plan",
    "pytorch": "libtorch",
    "onnxruntime": "onnx",
}
# Last resort for models that declare neither, e.g. qa_scalar_models/onnx_scalar_*.
SUFFIX_TO_BACKEND = {
    ".onnx": "onnx",
    ".plan": "plan",
    ".pt": "libtorch",
    ".graphdef": "graphdef",
    ".netdef": "netdef",
    ".xml": "openvino",
}

# The importable module whose __version__ describes each backend's artifacts.
BACKEND_TO_MODULE = {
    "onnx": "onnx",
    "openvino": "openvino",
    "libtorch": "torch",
    "plan": "tensorrt",
}

# Repository groups copied forward from STATIC_MODELS_SOURCE_VERSION rather than
# generated. Note torchtrt_model_store ends in _model_store but IS generated, so
# the name alone cannot decide this.
STATIC_REPOSITORIES = frozenset(
    [
        "c2_model_store",
        "libtorch_model_store",
        "libtorch_model_store2",
        "libtorch_tensor_structure",
        "onnx_model_store",
        "onnx_model_store2",
        "openvino_model_store",
        # perf_model_store ships as two artifacts split by backend. Both the
        # merged directory and the split ones are listed: a tree unpacked from
        # the older artifacts still has the former.
        "perf_model_store",
        "perf_model_store_libtorch",
        "perf_model_store_onnx",
        "pytorch_model_store",
    ]
)


class ManifestError(Exception):
    """A manifest could not be built or read."""


# platform and backend are the verbatim config.pbtxt declarations, either of
# which may be absent; provenance is the serving backend derived from them.
# Keeping all three means the raw declaration survives normalisation -- see the
# torch_aoti case, where platform and backend disagree and both are meaningful.
ModelConfig = collections.namedtuple(
    "ModelConfig", "name platform backend provenance provenance_source"
)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def iter_model_dirs(tree):
    """Yield every model directory under `tree`, at any depth.

    Model directories are not at a uniform depth: most sit at
    <repository>/<model>/, but ensemble models live one level deeper under
    <repository>/<sub-repository>/<model>/, and qa_custom_ops nests as well.
    A model is any directory holding a config.pbtxt.
    """
    root = pathlib.Path(tree)
    if not root.is_dir():
        return
    for config in sorted(root.rglob("config.pbtxt")):
        yield config.parent


def resolve_repository(model_dir, tree):
    """The top-level repository group a model belongs to."""
    try:
        relative = (
            pathlib.Path(model_dir).resolve().relative_to(pathlib.Path(tree).resolve())
        )
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


# --------------------------------------------------------------------------
# config.pbtxt
# --------------------------------------------------------------------------


def _read_top_level_field(text, key):
    """Read a column-0 key from config.pbtxt.

    Anchored at the start of a line on purpose: `name` also appears inside
    input/output blocks, where it names a tensor rather than the model.
    """
    match = re.search(r'^%s\s*:\s*"([^"]*)"' % key, text, re.MULTILINE)
    return match.group(1) if match else None


def _infer_backend_from_artifacts(model_dir):
    for path in sorted(pathlib.Path(model_dir).rglob("*")):
        if path.is_file() and path.suffix in SUFFIX_TO_BACKEND:
            return SUFFIX_TO_BACKEND[path.suffix]
    return "unknown"


def read_model_config(model_dir):
    """Read a model's identity from config.pbtxt as a ModelConfig."""
    model_dir = pathlib.Path(model_dir)
    config = model_dir / "config.pbtxt"
    try:
        text = config.read_text(errors="replace")
    except OSError as error:
        raise ManifestError("cannot read {}: {}".format(config, error))

    # The directory name is authoritative in Triton, and is the documented
    # fallback for the few models that omit a top-level name.
    name = _read_top_level_field(text, "name") or model_dir.name
    declared_platform = _read_top_level_field(text, "platform")
    declared_backend = _read_top_level_field(text, "backend")

    if declared_platform:
        provenance = PLATFORM_TO_BACKEND.get(declared_platform, declared_platform)
        source = "config.pbtxt:platform"
    elif declared_backend:
        provenance = BACKEND_TO_BACKEND.get(declared_backend, declared_backend)
        source = "config.pbtxt:backend"
    else:
        provenance = _infer_backend_from_artifacts(model_dir)
        source = "artifact"

    return ModelConfig(
        name=name,
        platform=declared_platform,
        backend=declared_backend,
        provenance=provenance,
        provenance_source=source,
    )


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------


def measure_model_bytes(model_dir):
    """Recursive size of a model, excluding the manifest itself.

    Excluding the manifest keeps update_manifest_sizes() idempotent: writing
    the file cannot change the number it reports.
    """
    total = 0
    for root, _, files in os.walk(str(model_dir)):
        for name in files:
            if name == MANIFEST_NAME:
                continue
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # A dangling symlink or a file removed mid-walk is not fatal.
                continue
    return total


def resolve_size_tier(total):
    """Map a byte count onto the t-shirt tier documented in README.md."""
    for limit, tier in SIZE_TIERS:
        if total < limit:
            return tier
    return LARGEST_TIER


def format_size(total):
    """A byte count in the largest unit that keeps it readable."""
    for unit, limit in (("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if total >= limit:
            return "{:.1f} {}".format(total / float(limit), unit)
    return "{} B".format(total)


# --------------------------------------------------------------------------
# build environment
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def per_model_logging(declared=None):
    """Whether to print one line per model as it is stamped.

    Off by default. A full tree is ~976 models across ~20 passes, so this was
    ~1700 lines burying the generators' own output. What a reader actually
    needs from a running pass -- that it is progressing, and what it is
    recording -- is the properties block and the per-repository line, which
    together cost about 300 lines. Set TRITON_MODEL_GEN_VERBOSE=1 when a
    specific model's size or tier is the thing being chased.
    """
    if declared is not None:
        return declared
    return os.environ.get(ENV_VERBOSE, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def resolve_upstream_version(declared=None):
    """The container train, from the environment when not passed in."""
    return declared or os.environ.get(ENV_UPSTREAM)


# ModelProto.opset_import, from onnx.proto. The default domain is spelled two
# ways; anything else is a custom operator set and not what "the opset" means.
ONNX_IR_VERSION_FIELD = 1
ONNX_OPSET_FIELD = 8
ONNX_DEFAULT_DOMAINS = ("", "ai.onnx")
TENSORRT_PLAN_MAGIC = 0x74727466


def _read_varint(handle):
    """One protobuf varint, or None at EOF or on a malformed run."""
    shift = value = 0
    while shift <= 63:
        byte = handle.read(1)
        if not byte:
            return None
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
        shift += 7
    return None


def _parse_opset_entry(blob):
    """domain and version out of one OperatorSetIdProto."""
    domain, version = "", None
    handle = io.BytesIO(blob)
    while True:
        tag = _read_varint(handle)
        if tag is None:
            break
        field, wire = tag >> 3, tag & 7
        if wire == 2:
            length = _read_varint(handle)
            if length is None:
                break
            data = handle.read(length)
            if field == 1:
                domain = data.decode("utf-8", "replace")
        elif wire == 0:
            value = _read_varint(handle)
            if value is None:
                break
            if field == 2:
                version = value
        elif wire == 1:
            handle.seek(8, io.SEEK_CUR)
        elif wire == 5:
            handle.seek(4, io.SEEK_CUR)
        else:
            break
    return domain, version


def probe_onnx_header(path):
    """`{ir_version, opset}` from an .onnx file's ModelProto, or {}.

    Both are properties of the file rather than of whatever produced it, so
    they are recorded for static models too. The producer fields are read past
    but not kept: they name the exporter -- tf2onnx in this corpus -- which is
    not the ONNX version anyone means.
    """
    found = {}
    try:
        with open(path, "rb") as handle:
            while True:
                tag = _read_varint(handle)
                if tag is None:
                    break
                field, wire = tag >> 3, tag & 7
                if wire == 0:
                    value = _read_varint(handle)
                    if value is None:
                        break
                    if field == ONNX_IR_VERSION_FIELD:
                        found["onnx_version"] = value
                elif wire == 2:
                    length = _read_varint(handle)
                    if length is None:
                        break
                    if field == ONNX_OPSET_FIELD:
                        domain, version = _parse_opset_entry(handle.read(length))
                        if version is not None and domain in ONNX_DEFAULT_DOMAINS:
                            found["onnx_opset_version"] = version
                    else:
                        handle.seek(length, io.SEEK_CUR)
                elif wire == 1:
                    handle.seek(8, io.SEEK_CUR)
                elif wire == 5:
                    handle.seek(4, io.SEEK_CUR)
                else:
                    break
    except OSError:
        return {}
    return found


def probe_openvino_ir_version(path):
    """The IR version an OpenVINO .xml declares, or None.

    `<net ... version="11">` is the only version these files carry -- the
    product version is not written into the IR -- so this is the OpenVINO
    version *of the artifact*, not of the toolkit that produced it.

    Only the opening tag is read: the graph body runs to megabytes and none of
    it is needed.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    match = re.search(r"<net\b[^>]*?\bversion=\"(\d+)\"", head)
    return int(match.group(1)) if match else None


def probe_torchscript_version(path):
    """The TorchScript archive version inside a .pt, or None.

    A .pt is a zip whose `<name>/version` entry holds the serialization format
    version. That is what the artifact declares; the torch release that wrote
    it is not recorded in the archive.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entry = next(
                (n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "version"),
                None,
            )
            if entry is None:
                return None
            text = archive.read(entry).decode("ascii", "replace").strip()
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return int(text) if text.isdigit() else None


def probe_tensorrt_version(path):
    """The TensorRT version a serialized .plan declares, or None.

    The plan begins with the magic 0x74727466 ("ftrt"), and four bytes at
    offset 0x18 carry major, minor, patch and build. That offset is read from
    the format rather than from documentation NVIDIA publishes, so the magic is
    checked first and anything implausible is discarded -- a wrong version here
    would be worse than an absent one, since a plan is only loadable by the
    runtime that matches it.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(28)
    except OSError:
        return None
    if len(head) < 28 or struct.unpack_from("<I", head, 0)[0] != TENSORRT_PLAN_MAGIC:
        return None
    major, minor, patch, build = head[24:28]
    if not 1 <= major <= 99:
        return None
    return "{}.{}.{}.{}".format(major, minor, patch, build)


def _first_file(model_dir, pattern):
    """The first matching *file*, sorted.

    Files only: an ONNX model with external data is a *directory* named
    model.onnx holding model.onnx plus its weights, and the directory sorts
    ahead of the file inside it.
    """
    return next((p for p in sorted(model_dir.rglob(pattern)) if p.is_file()), None)


def probe_artifact_versions(model_dir):
    """Per-format version blocks for whichever artifacts a model directory holds."""
    blocks = {}

    onnx_file = _first_file(model_dir, "*.onnx")
    if onnx_file is not None:
        header = probe_onnx_header(onnx_file)
        if header:
            blocks["onnx"] = header

    openvino_file = _first_file(model_dir, "*.xml")
    if openvino_file is not None:
        version = probe_openvino_ir_version(openvino_file)
        if version is not None:
            blocks["openvino"] = {"openvino_version": version}

    torch_file = _first_file(model_dir, "*.pt")
    if torch_file is not None:
        version = probe_torchscript_version(torch_file)
        if version is not None:
            blocks["pytorch"] = {"pytorch_version": version}

    plan_file = _first_file(model_dir, "*.plan")
    if plan_file is not None:
        version = probe_tensorrt_version(plan_file)
        if version is not None:
            blocks["tensorrt"] = {"tensorrt_version": version}

    return blocks


def probe_os_release():
    """'{NAME}:{VERSION_ID}', e.g. Ubuntu:24.04.

    Cached: the machine does not change while a script runs, and this is called
    once per model.
    """
    try:
        info = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        return "{}:{}".format(platform.system(), platform.release())
    return "{}:{}".format(
        info.get("NAME", "unknown"), info.get("VERSION_ID", "unknown")
    )


def _read_root_mount_source():
    """The backing path of `/`, from /proc/self/mountinfo, or ''.

    Field 3 is the directory within the filesystem that is mounted, field 4 the
    mount point. See Documentation/filesystems/proc.rst.
    """
    try:
        with open("/proc/self/mountinfo") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 4 and fields[4] == "/":
                    return fields[3]
    except OSError:
        pass
    return ""


@functools.lru_cache(maxsize=1)
def probe_container_runtime():
    """'docker', 'enroot', or None if neither can be established.

    Detected rather than declared, so the field is right even when the driver
    passes nothing -- but set TRITON_MODEL_GEN_RUNTIME if you want certainty,
    because only that path is exact.

    Docker leaves an empty /.dockerenv at the root of every image. enroot leaves
    no marker at all: it exports nothing into the container and deliberately
    strips the ENROOT_* variables it was invoked with, so testing the
    environment both misses real enroot containers and false-positives on any
    host whose shell has ENROOT_CONFIG_PATH set. What it does leave is the
    shape of the mount: / is bind-mounted from $ENROOT_DATA_PATH/<container>,
    which by default has an `enroot` path component. Retuning ENROOT_DATA_PATH
    defeats this, which is why it is the last resort rather than the first.
    """
    declared = os.environ.get(ENV_RUNTIME)
    if declared:
        return declared
    if os.path.exists("/.dockerenv"):
        return "docker"
    if "enroot" in _read_root_mount_source().split("/"):
        return "enroot"
    return None


def _import_optional(name):
    """Import a module, or None if it is absent.

    Not a bare `try: import cuda`. The cuda-python distribution installs `cuda`
    as an implicit namespace package, so that import succeeds even when nothing
    is installed under it, leaving an empty module whose __file__ is None. The
    caller would then fail later with AttributeError instead of a clean skip.
    """
    import importlib
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def format_cuda_version(raw):
    """The CUDA version as `<major>.<minor>`, from either probe's spelling.

    cuDriverGetVersion returns a packed integer -- 13010 is 13.1.0 -- while the
    CUDA_VERSION an image declares is already dotted, and may carry a patch
    component. Normalising both to `13.1` gives one field a consumer can
    compare without knowing which path produced it; `cuda_runtime` keeps the
    raw value for anyone who needs the patch level.
    """
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if "." in text:
        parts = text.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else None
    if not text.isdigit():
        return None
    packed = int(text)
    return "{}.{}".format(packed // 1000, (packed % 1000) // 10)


def _probe_gpu_cuda_python():
    """GPU details from cuda-python and NVML, or None if either is absent.

    Preferred over nvidia-smi: no subprocess, and cuDriverGetVersion is the
    authoritative CUDA version rather than whatever CUDA_VERSION the container
    image happens to declare.
    """
    core = _import_optional("cuda.core")
    if core is None or not hasattr(core, "Device"):
        return None
    try:
        device = core.Device(0)
        capability = device.compute_capability
        details = {
            "name": device.name,
            # Two spellings of the same fact, both in circulation. arch is the
            # SM number used by nvcc -arch=sm_89 and by cuda.core.Device.arch;
            # compute_capability is the dotted form CI's MODEL_TYPE uses
            # (A100-x86-8.0) and nvidia-smi --query-gpu=compute_cap reports.
            "arch": "{}{}".format(capability.major, capability.minor),
            "compute_capability": "{}.{}".format(capability.major, capability.minor),
            "driver_version": None,
            "cuda_runtime": None,
            "cuda_version": None,
        }
    except Exception:
        # Any driver-level failure means we cannot describe the GPU; fall back.
        return None

    bindings = _import_optional("cuda.bindings.driver")
    if bindings is not None:
        try:
            _, version = bindings.cuDriverGetVersion()
            details["cuda_runtime"] = str(version)
            details["cuda_version"] = format_cuda_version(version)
        except Exception:
            pass

    # The display driver version is an NVML concept; cuda-python does not
    # expose it, and cuDriverGetVersion reports something different.
    nvml = _import_optional("pynvml")
    if nvml is not None:
        try:
            nvml.nvmlInit()
            try:
                details["driver_version"] = nvml.nvmlSystemGetDriverVersion()
            finally:
                nvml.nvmlShutdown()
        except Exception:
            pass

    return details


def _probe_gpu_nvidia_smi():
    """GPU details by shelling out, for hosts without cuda-python.

    nvidia-smi is present wherever a GPU is, including images that carry no
    Python CUDA bindings at all.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) < 3:
        return None
    # compute_cap is the dotted form; the SM number is it without the dot,
    # which holds for two-digit majors too (10.0 -> 100).
    capability = fields[1]
    return {
        "name": fields[0],
        "arch": capability.replace(".", "") or None,
        "compute_capability": capability or None,
        "driver_version": fields[2],
        # Only what the image declares; cuDriverGetVersion is authoritative but
        # needs cuda-python, which this path exists precisely to do without.
        "cuda_runtime": os.environ.get("CUDA_VERSION"),
        "cuda_version": format_cuda_version(os.environ.get("CUDA_VERSION")),
    }


@functools.lru_cache(maxsize=1)
def _probe_gpu_cached():
    """Build-host GPU, or None when there is no GPU at all.

    Cached deliberately: called once per model, and over a full tree of ~960
    models an uncached nvidia-smi probe cost ~33 s of subprocess time. Worse, a
    hanging nvidia-smi would apply its timeout per model rather than once.
    """
    return _probe_gpu_cuda_python() or _probe_gpu_nvidia_smi()


def probe_gpu():
    """Build-host GPU as a fresh dict, or None.

    Wraps the cached probe so callers cannot mutate the shared cached value.
    """
    cached = _probe_gpu_cached()
    return dict(cached) if cached is not None else None


@functools.lru_cache(maxsize=None)
def probe_framework_version(framework):
    """Version of the library that produced a framework's artifacts, or None.

    The installed version, not the pin in pyproject.toml -- those diverge, and
    it is the installed one that determines what the artifacts look like. The
    OpenVINO stores show why: models built by different library versions carry
    different IR versions (10 vs 11) for otherwise identical inputs.

    Read in-process, because this is imported by the generator script that has
    already imported the framework it is generating for.
    """
    module_name = BACKEND_TO_MODULE.get(framework)
    if module_name is None:
        return None
    module = _import_optional(module_name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def build_manifest(
    model_dir,
    tree=None,
    image=None,
    container_runtime=None,
    framework=None,
    framework_version=None,
    generator=None,
    origin=None,
    triton_version=None,
    upstream_version=None,
    include_size=False,
):
    """Build a manifest for one model directory.

    `origin` defaults to 'static' for the version-pinned stores and 'dynamic'
    otherwise, which needs `tree` to identify the repository group.
    """
    model_dir = pathlib.Path(model_dir)
    config = read_model_config(model_dir)

    if origin is None:
        repository = resolve_repository(model_dir, tree) if tree else ""
        origin = "static" if repository in STATIC_REPOSITORIES else "dynamic"

    # A static model was built elsewhere, at an unknown time, on unknown
    # hardware -- it is copied forward, not generated here. Probing this host
    # for its GPU, platform and library version would answer a question about
    # the machine that happened to stamp the manifest, and record it as though
    # it described the artifact. Compute capability is exactly the field a
    # consumer filters on to decide whether a plan or AOTI artifact is portable
    # to their GPU, so a confident wrong answer there is worse than none.
    built_here = origin != "static"

    model = {
        "name": config.name,
        # The stage and library that produced this model. Distinct from
        # provenance, which is the backend that serves it: the ONNX stage also
        # produces ensemble and identity models.
        "framework": framework,
        "framework_version": framework_version if built_here else None,
        # The script that wrote it, which provenance alone does not identify --
        # eight different generators emit onnx models.
        "generator": generator,
        "platform": config.platform,
        "backend": config.backend,
        "provenance": config.provenance,
        "provenance_source": config.provenance_source,
        "origin": origin,
        "triton_version": triton_version,
        "upstream_container_version": upstream_version,
        "gpu": probe_gpu() if built_here else None,
    }
    # What the artifacts themselves declare. Recorded even for static models,
    # unlike the build-environment fields: these are properties of the files,
    # not of whatever produced them, and they decide whether a runtime can load
    # the model at all. A block appears only when its format is present and
    # readable, so a libtorch model carries no `onnx` key.
    model.update(probe_artifact_versions(model_dir))

    if include_size:
        total = measure_model_bytes(model_dir)
        model["size_bytes"] = total
        model["size_tier"] = resolve_size_tier(total)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "properties": {
            # Runtime-neutral on purpose. Naming the block after the engine --
            # "docker" or "enroot" -- would make the schema shape depend on the
            # data, forcing every consumer to probe both keys to find the image.
            # enroot pulls docker:// references too, so image_name means the
            # same thing either way; only the engine differs.
            "container": {
                "image_name": image if built_here else None,
                "runtime": container_runtime if built_here else None,
            },
            "platform": {
                "platform": platform.system() if built_here else None,
                "os_release": probe_os_release() if built_here else None,
                # Architecture survives: a static artifact is fetched for an
                # arch, and the tree it came from is arch-specific even when
                # nothing else about the build host is known.
                "arch": platform.machine().lower(),
            },
            "model": model,
        },
    }


def write_manifest(model_dir, manifest):
    """Write a manifest beside the model's config.pbtxt."""
    path = pathlib.Path(model_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_manifest(model_dir):
    """Read a model's manifest."""
    path = pathlib.Path(model_dir) / MANIFEST_NAME
    try:
        return json.loads(path.read_text())
    except OSError as error:
        raise ManifestError("no manifest at {}: {}".format(path, error))
    except ValueError as error:
        raise ManifestError("invalid manifest at {}: {}".format(path, error))


def update_manifest_sizes(model_dir):
    """Phase 2: add size_bytes and size_tier to an existing manifest.

    Idempotent -- measure_model_bytes() ignores the manifest, so re-running
    produces a byte-identical file.
    """
    manifest = load_manifest(model_dir)
    model = manifest.setdefault("properties", {}).setdefault("model", {})
    total = measure_model_bytes(model_dir)
    model["size_bytes"] = total
    model["size_tier"] = resolve_size_tier(total)
    write_manifest(model_dir, manifest)
    return manifest


SUMMARY_NAME = "manifest-summary.json"
SUMMARY_KIND = "triton-qa-model-manifest-summary"


def describe_model(model_dir, tree, manifest):
    """One model as a summary record."""
    model = manifest["properties"]["model"]
    try:
        path = str(
            pathlib.Path(model_dir).resolve().relative_to(pathlib.Path(tree).resolve())
        )
    except ValueError:
        path = str(model_dir)
    return {
        "name": model.get("name"),
        "path": path,
        "repository": resolve_repository(model_dir, tree),
        "provenance": model.get("provenance"),
        "framework": model.get("framework"),
        "size_bytes": model.get("size_bytes"),
        "size_tier": model.get("size_tier"),
    }


def build_summary(tree, records, written=0, unchanged=0, skipped=0, **fields):
    """A machine-readable account of a manifest pass.

    Exists so CI can assert on counts and totals directly instead of parsing
    them back out of log text.
    """
    by_tier = collections.Counter()
    by_provenance = collections.Counter()
    total = 0
    for record in records:
        if record.get("size_tier"):
            by_tier[record["size_tier"]] += 1
        if record.get("provenance"):
            by_provenance[record["provenance"]] += 1
        total += record.get("size_bytes") or 0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "tree": str(tree),
        "written": written,
        "unchanged": unchanged,
        "skipped": skipped,
        "model_count": len(records),
        "total_bytes": total,
        "by_size_tier": dict(sorted(by_tier.items())),
        "by_provenance": dict(sorted(by_provenance.items())),
        "models": sorted(records, key=lambda record: record["path"]),
    }
    summary.update({key: value for key, value in fields.items() if value is not None})
    return summary


def write_summary(target, summary):
    """Write a summary to a path, or to stdout when target is '-'."""
    text = json.dumps(summary, indent=2) + "\n"
    if str(target) == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
        return None
    path = pathlib.Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# generator-script hook
# --------------------------------------------------------------------------


def _fingerprint_model_dir(model_dir):
    """A cheap value that changes when a model directory's contents change.

    manifest.json is excluded so that emitting a manifest does not itself count
    as a change on a later pass.
    """
    count = 0
    total = 0
    newest = 0
    for root, _, files in os.walk(str(model_dir)):
        for name in files:
            if name == MANIFEST_NAME:
                continue
            try:
                stat = os.stat(os.path.join(root, name))
            except OSError:
                continue
            count += 1
            total += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
    return (count, total, newest)


def snapshot_model_dirs(tree):
    """Fingerprint every model directory under `tree`.

    Call before generating; pass the result to emit_manifests() afterwards so
    only the models this script created or added to are stamped. Returns an
    empty mapping for a tree that does not exist yet, which is the common case
    on a clean build and correctly means "everything found later is new".
    """
    return {
        str(model_dir): _fingerprint_model_dir(model_dir)
        for model_dir in iter_model_dirs(tree)
    }


def emit_manifests(
    tree,
    baseline=None,
    framework=None,
    generator=None,
    image=None,
    container_runtime=None,
    triton_version=None,
    upstream_version=None,
    include_size=True,
    dry_run=False,
    verbose=True,
    per_model=None,
    summary_path=None,
):
    """Write manifests for the models under `tree` that changed since `baseline`.

    The one call a generator script makes. Unset arguments are resolved from the
    environment, and `framework` falls back per model to the backend that serves
    it, so a script needs to pass nothing but the tree and its baseline.

    Announces the properties it will record and reports each repository as it
    finishes, so a long pass shows progress rather than going silent; pass
    verbose=False for just the summary line. Per-model lines are off unless
    per_model=True or TRITON_MODEL_GEN_VERBOSE is set -- see per_model_logging.
    `summary_path` additionally writes the whole pass as JSON.

    Never raises. A generation run that has already produced its models must not
    fail over a metadata file, so problems are reported on stdout and counted in
    the summary line; the phase-2 pass over the assembled tree is where missing
    manifests get caught.

    Returns the number of manifests written.
    """
    try:
        return _emit_manifests(
            tree,
            baseline=baseline,
            framework=framework,
            generator=generator,
            image=image,
            container_runtime=container_runtime,
            triton_version=triton_version,
            upstream_version=upstream_version,
            include_size=include_size,
            dry_run=dry_run,
            verbose=verbose,
            per_model=per_model,
            summary_path=summary_path,
        )
    except Exception as error:
        # Deliberately broad: see the docstring. A generation run that has
        # already produced its models must not die over a metadata file.
        _log("ERROR: no manifests written under {}: {}".format(tree, error))
        return 0


def _emit_manifests(
    tree,
    baseline,
    framework,
    generator,
    image,
    container_runtime,
    triton_version,
    upstream_version,
    include_size,
    dry_run,
    verbose=True,
    per_model=None,
    summary_path=None,
):
    if generator is None:
        generator = os.path.basename(sys.argv[0]) or None
    if image is None:
        image = os.environ.get(ENV_IMAGE)
    if container_runtime is None:
        container_runtime = probe_container_runtime()
    if triton_version is None:
        triton_version = os.environ.get(ENV_SEMVER)
    if upstream_version is None:
        upstream_version = resolve_upstream_version()
    declared_framework = framework or os.environ.get(ENV_FRAMEWORK)

    show_models = per_model_logging(per_model)
    if verbose:
        _log_properties(
            tree,
            generator,
            declared_framework,
            image,
            container_runtime,
            triton_version,
            upstream_version,
        )

    written = unchanged = skipped = 0
    records = []
    for model_dir in iter_model_dirs(tree):
        if baseline is not None:
            key = str(model_dir)
            if key in baseline and baseline[key] == _fingerprint_model_dir(model_dir):
                unchanged += 1
                continue
        try:
            config = read_model_config(model_dir)
            # Without a declared stage, the backend that serves the model is the
            # best available answer and is right for the single-framework runs.
            resolved = declared_framework or config.provenance
            manifest = build_manifest(
                model_dir,
                tree=tree,
                image=image,
                container_runtime=container_runtime,
                framework=resolved,
                framework_version=probe_framework_version(resolved),
                generator=generator,
                triton_version=triton_version,
                upstream_version=upstream_version,
                include_size=include_size,
            )
            if not dry_run:
                write_manifest(model_dir, manifest)
            written += 1
            record = describe_model(model_dir, tree, manifest)
            records.append(record)
            if show_models:
                _log_model(record)
        except ManifestError as error:
            _log("skipped {}: {}".format(model_dir, error))
            skipped += 1

    verb = "would write" if dry_run else "wrote"
    _log(
        "{} {} manifest(s) under {} ({} unchanged, {} skipped)".format(
            verb, written, tree, unchanged, skipped
        )
    )
    if summary_path and not dry_run:
        target = write_summary(
            summary_path,
            build_summary(
                tree,
                records,
                written=written,
                unchanged=unchanged,
                skipped=skipped,
                generator=generator,
                framework=declared_framework,
                image=image,
                container_runtime=container_runtime,
            ),
        )
        if target:
            _log("summary written to {}".format(target))
    return written


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def _log(message):
    print("[manifest] {}".format(message), flush=True)


def _log_properties(
    tree,
    generator,
    framework,
    image,
    container_runtime,
    triton_version,
    upstream_version,
    framework_version=None,
):
    """The properties this pass will record, once, before it starts stamping.

    Everything here ends up in every manifest the pass writes. Printing it as
    generation runs is what makes a log answer "which openvino was this built
    against?" without unpacking a model and reading its manifest.

    `framework` is None when the caller did not declare a stage, in which case
    it is resolved per model from the backend that serves it -- and its version
    with it, so neither can be reported up front.
    """
    gpu = probe_gpu() or {}
    if framework_version is None and framework:
        framework_version = probe_framework_version(framework)
    properties = [
        ("tree", tree),
        ("generator", generator),
        ("framework", framework or "per model (from config.pbtxt)"),
        ("framework version", framework_version),
        ("container image", image),
        ("container runtime", container_runtime),
        ("platform", "{} {}".format(probe_os_release(), platform.machine().lower())),
        ("gpu", gpu.get("name")),
        ("gpu compute capability", gpu.get("compute_capability")),
        ("driver version", gpu.get("driver_version")),
        ("triton version", triton_version),
        ("upstream container version", upstream_version),
    ]
    _log("properties recorded by this pass:")
    for name, value in properties:
        _log("  {:<28} {}".format(name, value if value else "-"))


def _log_model(record):
    """One line per model, as it is stamped.

    A full tree is ~976 models and the pass is not instant, so without this the
    run looks hung. The fields are the ones a reader is deciding on: what serves
    it, how big it is, and which tier that puts it in.
    """
    _log(
        "  {:<64} {:<10} {:>10}  {}".format(
            record["path"][:64],
            record["provenance"] or "-",
            (
                format_size(record["size_bytes"])
                if record["size_bytes"] is not None
                else "-"
            ),
            record["size_tier"] or "-",
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tree", required=True, type=pathlib.Path, help="model repository root"
    )
    parser.add_argument(
        "--repository",
        action="append",
        default=[],
        help="limit to this repository group, repeatable",
    )
    parser.add_argument(
        "--provenance",
        action="append",
        default=[],
        help="limit to models resolving to this backend, repeatable",
    )
    parser.add_argument(
        "--update-sizes",
        action="store_true",
        help="phase 2: add size_bytes/size_tier to existing manifests",
    )
    parser.add_argument("--image", help="container image that built these models")
    parser.add_argument("--framework", help="stage that generated these models")
    parser.add_argument(
        "--framework-version", help="version of that stage's library, as installed"
    )
    parser.add_argument("--generator", help="script that generated these models")
    parser.add_argument(
        "--container-runtime",
        choices=["docker", "enroot"],
        help="engine that ran the build image",
    )
    parser.add_argument("--origin", choices=["dynamic", "static"])
    parser.add_argument("--triton-version", help="Triton semver, e.g. 2.72.0dev")
    parser.add_argument("--upstream-version", help="container train, e.g. 26.07")
    parser.add_argument(
        "--dry-run", action="store_true", help="report without writing anything"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the properties block, keep the summary line",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print one line per model; off by default because a full "
        "tree is ~976 of them [{}]".format(ENV_VERBOSE),
    )
    parser.add_argument(
        "--summary",
        help="write a JSON account of this pass to PATH, or to stdout for '-'",
    )
    args = parser.parse_args(argv)

    if not args.tree.is_dir():
        parser.error("no such tree: {}".format(args.tree))

    # Same resolution the library path uses, so a manifest does not depend on
    # which of the two entry points wrote it.
    args.upstream_version = resolve_upstream_version(args.upstream_version)
    if args.triton_version is None:
        args.triton_version = os.environ.get(ENV_SEMVER)

    # The size pass rewrites only size_bytes/size_tier, leaving every property
    # below untouched, so reporting them there would describe this invocation
    # rather than what the manifests actually say.
    if not args.quiet and not args.update_sizes:
        _log_properties(
            args.tree,
            args.generator,
            args.framework,
            args.image,
            args.container_runtime or probe_container_runtime(),
            args.triton_version,
            args.upstream_version,
            framework_version=args.framework_version,
        )

    show_models = per_model_logging(args.verbose or None)

    touched = skipped = 0
    records = []
    for model_dir in iter_model_dirs(args.tree):
        if args.repository:
            if resolve_repository(model_dir, args.tree) not in args.repository:
                continue
        try:
            if args.update_sizes:
                if args.provenance:
                    current = load_manifest(model_dir)["properties"]["model"].get(
                        "provenance"
                    )
                    if current not in args.provenance:
                        continue
                manifest = (
                    update_manifest_sizes(model_dir)
                    if not args.dry_run
                    else load_manifest(model_dir)
                )
            else:
                config = read_model_config(model_dir)
                if args.provenance and config.provenance not in args.provenance:
                    continue
                framework = args.framework or config.provenance
                version = args.framework_version
                if version is None:
                    version = probe_framework_version(framework)
                manifest = build_manifest(
                    model_dir,
                    tree=args.tree,
                    image=args.image,
                    container_runtime=args.container_runtime
                    or probe_container_runtime(),
                    framework=framework,
                    framework_version=version,
                    generator=args.generator,
                    origin=args.origin,
                    triton_version=args.triton_version,
                    upstream_version=args.upstream_version,
                )
                if not args.dry_run:
                    write_manifest(model_dir, manifest)
            touched += 1
            record = describe_model(model_dir, args.tree, manifest)
            records.append(record)
            if show_models:
                _log_model(record)
        except ManifestError as error:
            _log("skipped {}: {}".format(model_dir, error))
            skipped += 1

    verb = "would update" if args.dry_run else "updated"
    _log("{} {} manifest(s), skipped {}".format(verb, touched, skipped))

    if args.summary:
        target = write_summary(
            args.summary,
            build_summary(
                args.tree,
                records,
                written=touched,
                skipped=skipped,
                generator=args.generator,
                framework=args.framework,
                image=args.image,
            ),
        )
        if target:
            _log("summary written to {}".format(target))
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
