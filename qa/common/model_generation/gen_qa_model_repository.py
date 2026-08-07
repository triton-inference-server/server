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

"""Generate the QA model repositories, one framework at a time.

A Python wrapper over the same work the `gen_qa_model_repository` shell driver
does: build four framework stages -- OpenVINO, ONNX, PyTorch, TensorRT -- each
inside its own container, writing into a shared tree of model repositories.

What it adds over the shell driver:

  * **Selective generation.** Run one framework, or any subset, instead of all
    four. `--openvino`, `--onnx`, `--pytorch`, `--tensorrt`, or `--all`.
  * **Every environment variable is also a flag.** `--triton-version`,
    `--ubuntu-image`, `--onnx-opset` and the rest default to the variable the
    shell driver reads, so existing CI keeps working unchanged, but anything
    can be overridden per invocation without exporting.
  * **`--list` and `--dry-run`**, so what a flag combination would do is
    inspectable without running a multi-hour GPU job.
  * **Manifests.** Each stage is told its image, framework and runtime, so the
    manifest.json every generator writes records them; a final pass fills in
    sizes once all stages have finished.

Stage order is fixed at OpenVINO -> ONNX -> PyTorch -> TensorRT regardless of
flag order, and stages never run in parallel. All four write into the *same*
repositories -- qa_model_repository receives models from every one -- so the
order is load-bearing, not cosmetic.

Both container engines the shell driver supports are kept. enroot is not
vestigial: it is how the SLURM B200 job builds, with TRITON_MODELS_USE_DOCKER=0.

    ./gen_qa_model_repository.py --list
    ./gen_qa_model_repository.py --openvino --dry-run
    ./gen_qa_model_repository.py --onnx --onnx-opset 17
    ./gen_qa_model_repository.py --all --runtime enroot

Standard library only, and runs on the host, where the only things that need to
exist are bash and one of docker or enroot.
"""

import argparse
import collections
import datetime
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

COLOR_ERROR = "\033[31m"
COLOR_INFO = "\033[94m"
COLOR_RESET = "\033[0m"
COLOR_STATUS = "\033[36m"
COLOR_WARNING = "\033[33m"

# The same codes as text, for embedding in a generated bash script. Written as
# an escape *sequence* rather than a raw escape byte so the stage scripts stay
# plain ASCII and readable in a diff, exactly as the shell driver emits them.
SH_COLOR_INFO = "\\033[94m"
SH_COLOR_RESET = "\\033[0m"
SH_COLOR_WARNING = "\\033[33m"

# Defaults, matching the shell driver's `${VAR:=default}` values.
DEFAULT_TRITON_VERSION = "26.07"
DEFAULT_ONNX_VERSION = "1.20.1"
DEFAULT_ONNX_OPSET = "0"
DEFAULT_OPENVINO_VERSION = "2024.5.0"
DEFAULT_UBUNTU_IMAGE = "ubuntu:22.04"
DEFAULT_NVIDIA_VISIBLE_DEVICES = "0"
DEFAULT_PROJECT_NAME = "tritonserver"
DEFAULT_OUTPUT_DIR = "/tmp"

# Files outside this directory the generators need at run time. Staged flat
# into the source directory alongside the generators:
#   test_util.py        imported by 14 generators, shared with the L0_* suites
#   resnet50_labels.txt read by gen_qa_models.py as $TRITON_GENSRCDIR/<name>
SHARED_SOURCES = ("../test_util.py", "../resnet50_labels.txt")

# Repository directories created up front, in the shell driver's order.
# Two deliberate quirks preserved from it: qa_noshape_model_repository is
# created but no generator ever writes to it, and qa_custom_ops is absent --
# the PyTorch stage creates that one itself, two levels deep.
REPOSITORY_DIRS = (
    "qa_model_repository",
    "qa_variable_model_repository",
    "qa_identity_model_repository",
    "qa_identity_big_model_repository",
    "qa_shapetensor_model_repository",
    "qa_reshape_model_repository",
    "qa_sequence_model_repository",
    "qa_dyna_sequence_model_repository",
    "qa_dyna_sequence_implicit_model_repository",
    "qa_variable_sequence_model_repository",
    "qa_ensemble_model_repository",
    "qa_noshape_model_repository",
    "qa_trt_plugin_model_repository",
    "qa_ragged_model_repository",
    "qa_trt_format_model_repository",
    "qa_trt_data_dependent_model_repository",
    "qa_sequence_implicit_model_repository",
    "qa_variable_sequence_implicit_model_repository",
    "qa_sequence_initial_state_implicit_model_repository",
    "qa_variable_sequence_initial_state_implicit_model_repository",
    "torchtrt_model_store",
    "qa_scalar_models",
    "qa_dynamic_batch_image_model_repository",
)

CUSTOM_OPS_DIR = "qa_custom_ops/libtorch_custom_ops"

STAGE_ORDER = ("openvino", "onnx", "pytorch", "tensorrt")


# --------------------------------------------------------------------------
# logging, in the shell driver's format so CI logs stay recognisable
# --------------------------------------------------------------------------


def _log(colour, level, message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        "{}{} - [ {} ] - {} {}".format(colour, stamp, level, message, COLOR_RESET),
        flush=True,
    )


def log_info(message):
    _log(COLOR_INFO, "INFO", message)


def log_status(message):
    _log(COLOR_STATUS, "STATUS", message)


def log_warning(message):
    _log(COLOR_WARNING, "WARNING", message)


def log_error(message):
    _log(COLOR_ERROR, "ERROR", message)


class GenerationError(Exception):
    """A stage, or the setup around it, failed."""


# --------------------------------------------------------------------------
# stage description
# --------------------------------------------------------------------------


class Generate:
    """One generator invocation, plus the chmod that follows it."""

    def __init__(self, script, flags=(), repository=None, chmod=True):
        self.script = script
        self.flags = list(flags)
        self.repository = repository
        self.chmod = chmod

    def render(self, ctx):
        target = "{}/{}/{}".format(ctx.build_dir, ctx.triton_version, self.repository)
        command = ["python3", "{}/{}".format(ctx.source_dir, self.script)]
        command += self.flags
        command.append("--models_dir={}".format(target))
        lines = [" ".join(command)]
        if self.chmod:
            lines.append("chmod -R 777 {}".format(target))
        return lines

    def describe(self):
        return "{} {} -> {}".format(
            self.script, " ".join(self.flags) or "(no flags)", self.repository
        )


class Shell:
    """A raw shell fragment a generator invocation cannot express."""

    def __init__(self, text, summary):
        self.text = text
        self.summary = summary

    def render(self, ctx):
        return self.text.strip("\n").splitlines()

    def describe(self):
        return "shell: {}".format(self.summary)


class Skipped:
    """A generator a framework cannot handle, kept as data with its reason.

    Recorded rather than deleted so `--list` reports the gap instead of it
    silently not existing.
    """

    def __init__(self, script, reason):
        self.script = script
        self.reason = reason

    def render(self, ctx):
        return ["echo '[skipped] {}: {}'".format(self.script, self.reason)]

    def describe(self):
        return "SKIPPED {} ({})".format(self.script, self.reason)


Stage = collections.namedtuple(
    "Stage", "key title image needs_root prelude setup steps"
)


def build_stages(ctx):
    """The four stages, with every host-side value already substituted.

    Built as a function rather than a template so nothing goes through a shell
    heredoc. In the shell driver the heredocs are unquoted, so every `$VAR` in
    them expands on the *host* unless escaped -- the source of three latent
    bugs there. Here the distinction is explicit: a Python f-string/format is a
    host-side value, and a literal `$VAR` in the text is evaluated in the
    container.
    """
    version_root = "{}/{}".format(ctx.build_dir, ctx.triton_version)
    custom_ops_root = "{}/{}".format(version_root, CUSTOM_OPS_DIR)
    plugin_root = "{}/qa_trt_plugin_model_repository".format(version_root)
    data_dependent_root = "{}/qa_trt_data_dependent_model_repository".format(
        version_root
    )
    opset = ["--onnx_opset={}".format(ctx.onnx_opset)]

    openvino = Stage(
        key="openvino",
        title="OpenVINO",
        image=ctx.ubuntu_image,
        needs_root=True,
        prelude=[],
        setup=[
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update && \\\n"
            "    apt-get install -y --no-install-recommends \\\n"
            "        build-essential \\\n"
            "        cmake \\\n"
            "        libprotobuf-dev \\\n"
            "        protobuf-compiler \\\n"
            "        python3 \\\n"
            "        python3-dev \\\n"
            "        python3-pip \\\n"
            "        wget \\\n"
            "        gnupg2 \\\n"
            "        software-properties-common",
            "ln -s /usr/bin/python3 /usr/bin/python",
            'pip3 install "numpy<=1.23.5" setuptools "ml_dtypes<=0.5.4"',
            "pip3 install openvino=={}".format(ctx.openvino_version),
        ],
        steps=[
            Generate("gen_qa_models.py", ["--openvino"], "qa_model_repository"),
            Skipped("gen_qa_identity_models.py", "variable shape tensors, DLIS-2827"),
            Generate(
                "gen_qa_reshape_models.py",
                ["--openvino"],
                "qa_reshape_model_repository",
            ),
            Skipped("gen_qa_sequence_models.py", "sequence/control inputs, DLIS-2864"),
            Skipped(
                "gen_qa_dyna_sequence_models.py", "sequence/control inputs, DLIS-2864"
            ),
        ],
    )

    onnx = Stage(
        key="onnx",
        title="ONNX",
        image=ctx.ubuntu_image,
        needs_root=True,
        prelude=[],
        setup=[
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update && \\\n"
            "        apt-get install -y --no-install-recommends build-essential cmake"
            " libprotobuf-dev \\\n"
            "                protobuf-compiler python3 python3-dev python3-pip",
            "ln -s /usr/bin/python3 /usr/bin/python",
            # TODO: Remove the pins, DLIS-3838. Kept verbatim for parity: the
            # protobuf cap contradicts onnx's own protobuf>=4.25.1 and is
            # overwritten by the --upgrade below, and numpy<=1.23.5 has no
            # cp312 wheel, so both only survive because the default image is
            # ubuntu:22.04 with Python 3.10.
            'pip3 install "protobuf<=3.20.1" "numpy<=1.23.5" "ml_dtypes<=0.5.4"'
            "  # TODO: Remove current line DLIS-3838",
            "pip3 install --upgrade onnx=={}".format(ctx.onnx_version),
        ],
        steps=[
            Generate("gen_qa_models.py", ["--onnx"] + opset, "qa_model_repository"),
            Generate(
                "gen_qa_models.py",
                ["--onnx"] + opset + ["--variable"],
                "qa_variable_model_repository",
            ),
            Generate(
                "gen_qa_identity_models.py",
                ["--onnx"] + opset,
                "qa_identity_model_repository",
            ),
            Generate(
                "gen_qa_reshape_models.py",
                ["--onnx"] + opset + ["--variable"],
                "qa_reshape_model_repository",
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--onnx"] + opset,
                "qa_sequence_model_repository",
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--onnx"] + opset + ["--variable"],
                "qa_variable_sequence_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--onnx", "--initial-state", "zero"] + opset,
                "qa_sequence_initial_state_implicit_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--onnx", "--initial-state", "zero"] + opset + ["--variable"],
                "qa_variable_sequence_initial_state_implicit_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--onnx"] + opset,
                "qa_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--onnx"] + opset + ["--variable"],
                "qa_variable_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_dyna_sequence_models.py",
                ["--onnx"] + opset,
                "qa_dyna_sequence_model_repository",
            ),
            Generate(
                "gen_qa_dyna_sequence_implicit_models.py",
                ["--onnx"] + opset,
                "qa_dyna_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_ragged_models.py",
                ["--onnx"] + opset,
                "qa_ragged_model_repository",
            ),
            Generate("gen_qa_ort_scalar_models.py", opset, "qa_scalar_models"),
            # The ensembles. Six sub-repositories under one group, and the only
            # stage that builds them. These carry no --onnx/--onnx_opset flag,
            # and are chmod'ed once at the end rather than per call.
            Generate(
                "gen_qa_models.py",
                ["--ensemble"],
                "qa_ensemble_model_repository/qa_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_models.py",
                ["--ensemble", "--variable"],
                "qa_ensemble_model_repository/qa_variable_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_reshape_models.py",
                ["--ensemble"],
                "qa_ensemble_model_repository/qa_reshape_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_identity_models.py",
                ["--ensemble"],
                "qa_ensemble_model_repository/qa_identity_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--ensemble"],
                "qa_ensemble_model_repository/qa_sequence_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--ensemble", "--variable"],
                "qa_ensemble_model_repository/qa_variable_sequence_model_repository",
                chmod=False,
            ),
            Shell(
                "chmod -R 777 {}/qa_ensemble_model_repository".format(version_root),
                "chmod the ensemble group",
            ),
        ],
    )

    pytorch_steps = [
        Generate(
            "gen_qa_models.py", ["--libtorch"], "qa_model_repository", chmod=False
        ),
        Generate(
            "gen_qa_models.py", ["--torch-aoti"], "qa_model_repository", chmod=False
        ),
        Generate("gen_qa_models.py", ["--torchvision-aoti"], "qa_model_repository"),
        Generate(
            "gen_qa_models.py",
            ["--libtorch", "--variable"],
            "qa_variable_model_repository",
        ),
        Generate(
            "gen_qa_identity_models.py", ["--libtorch"], "qa_identity_model_repository"
        ),
        Generate(
            "gen_qa_reshape_models.py",
            ["--libtorch", "--variable"],
            "qa_reshape_model_repository",
        ),
        Generate(
            "gen_qa_sequence_models.py", ["--libtorch"], "qa_sequence_model_repository"
        ),
        Generate(
            "gen_qa_sequence_models.py",
            ["--libtorch", "--variable"],
            "qa_variable_sequence_model_repository",
        ),
        Generate(
            "gen_qa_implicit_models.py",
            ["--libtorch"],
            "qa_sequence_implicit_model_repository",
        ),
        Generate(
            "gen_qa_implicit_models.py",
            ["--torch-aoti"],
            "qa_sequence_implicit_model_repository",
        ),
        Generate(
            "gen_qa_implicit_models.py",
            ["--libtorch", "--variable"],
            "qa_variable_sequence_implicit_model_repository",
        ),
        Generate(
            "gen_qa_dyna_sequence_models.py",
            ["--libtorch"],
            "qa_dyna_sequence_model_repository",
        ),
    ]
    # torch_tensorrt is not available on iGPU. Decided here, on the host, where
    # MODEL_TYPE is actually known.
    if ctx.model_type != "igpu":
        pytorch_steps.append(
            Generate("gen_qa_torchtrt_models.py", [], "torchtrt_model_store")
        )
    else:
        pytorch_steps.append(Skipped("gen_qa_torchtrt_models.py", "MODEL_TYPE=igpu"))
    pytorch_steps += [
        Generate(
            "gen_qa_ragged_models.py", ["--libtorch"], "qa_ragged_model_repository"
        ),
        Generate(
            "gen_qa_image_models.py",
            ["--resnet50", "--resnet152", "--vgg19"],
            "qa_dynamic_batch_image_model_repository",
        ),
        Shell(
            # Concatenated rather than formatted: the text carries literal
            # ${TORCH_EXTENSIONS_DIR} for the container to expand, and brace
            # escaping to protect it from str.format is easy to get wrong.
            "export TORCH_EXTENSIONS_DIR=/tmp/.cache/torch_extensions/\n"
            "mkdir -p ${TORCH_EXTENSIONS_DIR}\n"
            "python3 " + ctx.source_dir + "/gen_qa_custom_ops_models.py"
            " --libtorch --models_dir=" + custom_ops_root + "\n"
            "mkdir -p " + custom_ops_root + "/libtorch_modulo/\n"
            "cp ${TORCH_EXTENSIONS_DIR}/custom_modulo/custom_modulo.so "
            + custom_ops_root
            + "/libtorch_modulo/.\n"
            "chmod -R 777 " + custom_ops_root,
            "custom ops, plus the built custom_modulo.so",
        ),
    ]

    pytorch = Stage(
        key="pytorch",
        title="PyTorch",
        image=ctx.pytorch_image,
        needs_root=False,
        # Before `set -e`, exactly as in the shell driver: a failure to install
        # onnxscript does not abort the stage.
        prelude=["pip3 install onnxscript"],
        setup=[
            # Literal $PATH, so it is the *container's* PATH that gets cuda
            # appended. The shell driver's unquoted heredoc expanded this on
            # the host and shipped the host's PATH into the container.
            'export PATH="$PATH:/usr/local/cuda-13.0/bin"',
        ],
        steps=pytorch_steps,
    )

    tensorrt = Stage(
        key="tensorrt",
        title="TensorRT",
        image=ctx.tensorrt_image,
        needs_root=False,
        prelude=[],
        setup=[
            # Under `set -e` this is an assertion: no TensorRT packages in the
            # image means the stage stops here rather than producing junk.
            "dpkg -l | grep TensorRT",
            "export TRT_SUPPRESS_DEPRECATION_WARNINGS=1",
            'pip3 install "ml_dtypes<=0.5.4"',
        ],
        steps=[
            Generate(
                "gen_qa_identity_models.py",
                ["--tensorrt-shape-io"],
                "qa_shapetensor_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--tensorrt-shape-io"],
                "qa_shapetensor_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_dyna_sequence_models.py",
                ["--tensorrt-shape-io"],
                "qa_shapetensor_model_repository",
            ),
            Generate("gen_qa_models.py", ["--tensorrt"], "qa_model_repository"),
            Generate(
                "gen_qa_models.py",
                ["--tensorrt", "--variable"],
                "qa_variable_model_repository",
            ),
            Generate(
                "gen_qa_identity_models.py",
                ["--tensorrt"],
                "qa_identity_model_repository",
                chmod=False,
            ),
            Generate(
                "gen_qa_identity_models.py",
                ["--tensorrt-compat"],
                "qa_identity_model_repository",
            ),
            Generate(
                "gen_qa_identity_models.py",
                ["--tensorrt-big"],
                "qa_identity_big_model_repository",
            ),
            Generate(
                "gen_qa_reshape_models.py",
                ["--tensorrt", "--variable"],
                "qa_reshape_model_repository",
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--tensorrt"],
                "qa_sequence_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--tensorrt"],
                "qa_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_implicit_models.py",
                ["--tensorrt", "--variable"],
                "qa_variable_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_dyna_sequence_models.py",
                ["--tensorrt"],
                "qa_dyna_sequence_model_repository",
            ),
            Generate(
                "gen_qa_sequence_models.py",
                ["--tensorrt", "--variable"],
                "qa_variable_sequence_model_repository",
            ),
            Generate(
                "gen_qa_dyna_sequence_implicit_models.py",
                ["--tensorrt"],
                "qa_dyna_sequence_implicit_model_repository",
            ),
            Generate(
                "gen_qa_ragged_models.py", ["--tensorrt"], "qa_ragged_model_repository"
            ),
            Generate(
                "gen_qa_trt_format_models.py", [], "qa_trt_format_model_repository"
            ),
            Shell(
                "nvidia-smi --query-gpu=compute_cap | grep -qzE '10\\.7|11\\.0'"
                " && echo -e '"
                + SH_COLOR_WARNING
                + "[WARNING]"
                + SH_COLOR_RESET
                + " Skipping model generation for data dependent shape"
                " (NonZero not supported on this GPU)" + SH_COLOR_RESET + "'"
                " || python3 " + ctx.source_dir + "/gen_qa_trt_data_dependent_shape.py"
                " --models_dir=" + data_dependent_root + "\n"
                "chmod -R 777 " + data_dependent_root,
                "data dependent shape models, skipped on sm 10.7 / 11.0",
            ),
            Shell(
                "# Build the custom Hardmax plugin used by L0_trt_plugin. The\n"
                "# plugin source is pulled from the public NVIDIA/TensorRT repo\n"
                "# at the release/<major.minor> branch matching the runtime TRT\n"
                "# version. If that branch is not published yet, generation is\n"
                "# skipped with a warning -- L0_trt_plugin will report missing\n"
                "# artifacts but the rest of the repository is still produced.\n"
                "TRT_BRANCH=$(echo ${TRT_VERSION} | cut -d . -f -2)\n"
                "TRTSRC=/workspace/TensorRT\n"
                "rm -rf ${TRTSRC}\n"
                "if git clone --depth 1 -b release/${TRT_BRANCH} \\\n"
                "     https://github.com/NVIDIA/TensorRT.git ${TRTSRC}; then\n"
                "  cd ${TRTSRC}/samples/python/onnx_custom_plugin && \\\n"
                "    rm -rf build && mkdir build && cd build && cmake .. &&"
                " make -j && \\\n"
                "    cp libcustomHardmaxPlugin.so " + plugin_root + "/.\n"
                "  LD_PRELOAD=" + plugin_root + "/libcustomHardmaxPlugin.so \\\n"
                "    python3 " + ctx.source_dir + "/gen_qa_trt_plugin_models.py \\\n"
                "    --models_dir=" + plugin_root + "\n"
                "  chmod -R 777 " + plugin_root + "\n"
                "else\n"
                '  echo "[WARNING] TensorRT release/${TRT_BRANCH} not available'
                " on github.com/NVIDIA/TensorRT; skipping CustomHardmax plugin"
                ' model generation. L0_trt_plugin coverage will be missing for this TRT version."\n'
                "fi",
                "clone NVIDIA/TensorRT, build the Hardmax plugin, generate",
            ),
        ],
    )

    return collections.OrderedDict(
        [
            ("openvino", openvino),
            ("onnx", onnx),
            ("pytorch", pytorch),
            ("tensorrt", tensorrt),
        ]
    )


# --------------------------------------------------------------------------
# stage script rendering
# --------------------------------------------------------------------------


def render_stage_script(stage, ctx, final=False):
    """The bash script a stage runs inside its container."""
    lines = [
        "#!/bin/bash",
        "# Generated by gen_qa_model_repository.py -- do not edit.",
        "# Make all generated files accessible outside of container",
        "umask 0000",
        "nvidia-smi --query-gpu=compute_cap,compute_mode,driver_version,name,index"
        " --format=csv || true",
        "nvidia-smi || true",
        'echo -e "{}Generating {} models{}"'.format(
            SH_COLOR_INFO, stage.title, SH_COLOR_RESET
        ),
    ]
    lines += list(stage.prelude)
    lines.append("set -e")
    lines += list(stage.setup)
    lines.append("")
    for step in stage.steps:
        lines += step.render(ctx)
    if final:
        lines += [
            "",
            "# Phase 2: sizes are only final once every stage has written its"
            " artifacts.",
            # The summary lands at the root of the tree, not inside a model, so
            # it travels with the tree and no walk of it picks the file up.
            "python3 {src}/gen_manifest.py --tree {root} --update-sizes"
            " --summary {root}/manifest-summary.json || true".format(
                src=ctx.source_dir,
                root="{}/{}".format(ctx.build_dir, ctx.triton_version),
            ),
        ]
        if ctx.archive:
            lines += [
                "",
                "# Pack each model separately, into a folder of its own beside"
                " the tree.",
                "python3 {}/gen_archive.py --tree {}/{} --dest {} || true".format(
                    ctx.source_dir,
                    ctx.build_dir,
                    ctx.triton_version,
                    ctx.container_archive_dir(),
                ),
                "chmod -R 777 {} || true".format(ctx.container_archive_dir()),
            ]
    lines += ["", "exit 0", ""]
    return "\n".join(lines)


def stage_environment(stage, ctx):
    """Variables handed to the container, including the manifest fields."""
    return collections.OrderedDict(
        [
            ("TRITON_GENSRCDIR", ctx.source_dir),
            ("TRITON_MODEL_GEN_IMAGE", stage.image),
            ("TRITON_MODEL_GEN_FRAMEWORK", stage.key),
            ("TRITON_MODEL_GEN_RUNTIME", ctx.runtime_name),
            ("TRITON_VERSION", ctx.triton_version),
        ]
    )


# --------------------------------------------------------------------------
# container runtimes
# --------------------------------------------------------------------------


def run_command(command, dry_run=False, check=True, capture=False):
    log_info("$ {}".format(" ".join(shlex.quote(part) for part in command)))
    if dry_run:
        return 0
    result = subprocess.run(command, capture_output=capture, text=capture)
    if check and result.returncode != 0:
        raise GenerationError(
            "command failed ({}): {}".format(result.returncode, command[0])
        )
    return result.returncode


class Runtime:
    name = None

    def __init__(self, ctx):
        self.ctx = ctx

    def prepare(self, stage_scripts):
        raise NotImplementedError

    def run_stage(self, stage, script_name):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError


class DockerRuntime(Runtime):
    """Models are built in a named volume mounted at /mnt."""

    name = "docker"
    mount_root = "/mnt"

    def __init__(self, ctx):
        Runtime.__init__(self, ctx)
        self.volume = ctx.docker_volume
        self.container = "{}.gen_qa_model_repository.{}".format(self.volume, ctx.job_id)

    def _labels(self):
        return [
            "--label",
            "RUNNER_ID={}".format(self.ctx.runner_id),
            "--label",
            "PROJECT_NAME={}".format(self.ctx.project_name),
        ]

    def prepare(self, stage_scripts):
        ctx = self.ctx
        inspect = subprocess.run(
            ["docker", "volume", "inspect", self.volume], capture_output=True
        )
        if inspect.returncode != 0:
            log_status(
                "docker volume: {} does not exist. Creating...".format(self.volume)
            )
            run_command(
                ["docker", "volume", "create", self.volume]
                + [
                    "--label",
                    "RUNNER_ID={}".format(ctx.runner_id),
                    "--label",
                    "PROJECT_NAME={}".format(ctx.project_name),
                ],
                ctx.dry_run,
            )
        else:
            log_status("docker volume: {} in use".format(self.volume))

        log_status("docker pull: {}".format(ctx.ubuntu_image))
        run_command(["docker", "pull", ctx.ubuntu_image], ctx.dry_run)

        log_status("docker volume: create directories on volume")
        run_command(
            ["docker", "run", "--rm"]
            + self._labels()
            + ["-v", "{}:/mnt".format(self.volume), ctx.ubuntu_image, "mkdir", "-p"]
            + [ctx.build_dir, ctx.source_dir]
            + ctx.repository_paths(),
            ctx.dry_run,
        )

        log_status("docker container: create {}".format(self.container))
        run_command(
            ["docker", "create"]
            + self._labels()
            + [
                "--name",
                self.container,
                "-v",
                "{}:/mnt".format(self.volume),
                "-w",
                "{}/{}".format(self.mount_root, ctx.job_id),
                ctx.ubuntu_image,
            ],
            ctx.dry_run,
        )

        log_status("docker container: copy generators and stage scripts")
        run_command(
            [
                "docker",
                "cp",
                str(SCRIPT_DIR) + "/.",
                "{}:{}".format(self.container, ctx.source_dir),
            ],
            ctx.dry_run,
        )
        for shared in SHARED_SOURCES:
            source = (SCRIPT_DIR / shared).resolve()
            run_command(
                [
                    "docker",
                    "cp",
                    str(source),
                    "{}:{}/".format(self.container, ctx.source_dir),
                ],
                ctx.dry_run,
            )
        for path in stage_scripts.values():
            run_command(
                [
                    "docker",
                    "cp",
                    str(path),
                    "{}:{}/".format(self.container, ctx.source_dir),
                ],
                ctx.dry_run,
            )

    def run_stage(self, stage, script_name):
        ctx = self.ctx
        log_status("docker pull: {}".format(stage.image))
        run_command(["docker", "pull", stage.image], ctx.dry_run)

        command = ["docker", "run", "--rm"]
        for key, value in stage_environment(stage, ctx).items():
            command += ["-e", "{}={}".format(key, value)]
        command += self._labels()
        command += ctx.docker_gpu_args
        command += [
            "-v",
            "{}:/mnt".format(self.volume),
            "-t",
            "-e",
            "TRT_VERBOSE",
            stage.image,
            "bash",
            "-e",
            "{}/{}".format(ctx.source_dir, script_name),
        ]
        run_command(command, ctx.dry_run)

    def collect(self):
        ctx = self.ctx
        if ctx.is_ci:
            log_status("CI set: leaving models in volume {}".format(self.volume))
            return None
        destination = pathlib.Path(ctx.output_dir)
        log_status("docker cp: copying generated models to {}".format(destination))
        # `docker cp` into a path that does not exist names the copy after that
        # path, putting the tree at <destination> instead of the
        # <destination>/<version> this returns. Create it first so the layout
        # does not depend on whether the caller's directory happens to exist.
        if not ctx.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "docker",
                "cp",
                "{}:{}/{}".format(self.container, ctx.build_dir, ctx.triton_version),
                str(destination) + "/",
            ],
            ctx.dry_run,
        )
        if ctx.archive:
            archives = pathlib.Path(ctx.archive_dir or ctx.output_dir)
            log_status("docker cp: copying archives to {}".format(archives))
            if not ctx.dry_run:
                archives.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    "docker",
                    "cp",
                    "{}:{}".format(self.container, ctx.container_archive_dir()),
                    str(archives) + "/",
                ],
                ctx.dry_run,
                check=False,
            )
        return destination / ctx.triton_version

    def cleanup(self):
        ctx = self.ctx
        if ctx.is_ci or not ctx.cleanup:
            return
        log_status("docker: removing container and volume")
        run_command(["docker", "rm", "-f", self.container], ctx.dry_run, check=False)
        run_command(["docker", "volume", "rm", self.volume], ctx.dry_run, check=False)


class EnrootRuntime(Runtime):
    """Models are built straight on the host filesystem under /tmp.

    No volume: enroot bind-mounts /tmp, so the tree is already where the caller
    wants it and there is nothing to copy out afterwards.
    """

    name = "enroot"
    mount_root = "/tmp"

    def __init__(self, ctx):
        Runtime.__init__(self, ctx)
        self.images = {}
        self.containers = []

    def prepare(self, stage_scripts):
        ctx = self.ctx
        build = pathlib.Path(ctx.build_dir)
        if build.exists() and ctx.clean_build_dir:
            # The shell driver intends to do this but cannot: it writes
            # `TRITON_MDLS_BLD_DIR=... define_model_output_directories`, and
            # bash discards a prefix assignment made to a *function* call, so
            # the variable is empty by the time its `rm -rf` runs.
            log_status("cleanup: removing existing {}".format(build))
            if not ctx.dry_run:
                shutil.rmtree(str(build), ignore_errors=True)

        log_status("create: model directory structure in {}".format(ctx.build_dir))
        if not ctx.dry_run:
            for path in [ctx.build_dir, ctx.source_dir] + ctx.repository_paths():
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        log_status("copy: generators to {}".format(ctx.source_dir))
        if not ctx.dry_run:
            for entry in sorted(SCRIPT_DIR.iterdir()):
                if entry.is_file():
                    shutil.copy2(str(entry), ctx.source_dir)
            for shared in SHARED_SOURCES:
                shutil.copy2(str((SCRIPT_DIR / shared).resolve()), ctx.source_dir)
            for path in stage_scripts.values():
                shutil.copy2(str(path), ctx.source_dir)

    def _import(self, image):
        ctx = self.ctx
        if image in self.images:
            return self.images[image]
        alias = image.replace("/", ".").replace(":", ".")
        sqsh = "/tmp/{}.{}.enroot.sqsh".format(alias, ctx.job_id)
        log_status("enroot import: {} to {}".format(image, sqsh))
        run_command(
            ["enroot", "import", "--output", sqsh, "docker://{}".format(image)],
            ctx.dry_run,
        )
        self.images[image] = sqsh
        return sqsh

    def run_stage(self, stage, script_name):
        ctx = self.ctx
        sqsh = self._import(stage.image)
        container = "{}.{}".format(stage.key, ctx.job_id)
        log_status("enroot create: {}".format(container))
        run_command(["enroot", "create", "--name", container, sqsh], ctx.dry_run)
        self.containers.append(container)

        command = ["enroot", "start"]
        if stage.needs_root:
            # The ubuntu-based stages apt-get their toolchain; the NGC images
            # already carry theirs and run unprivileged.
            command.append("--root")
        command += ["--rw", "-m", "/tmp:/tmp"]
        # enroot exposes GPUs through its 98-nvidia.sh hook, which keys entirely
        # off NVIDIA_VISIBLE_DEVICES in the container environment and bails when
        # it is unset. The shell driver never sets it on this path, so enroot
        # builds saw no GPU and recorded gpu: null -- harmless for OpenVINO, but
        # TensorRT plan files are compute-capability specific, so the field has
        # to be real. 'none' is the hook's own opt-out and is passed through.
        command += [
            "-e",
            "NVIDIA_VISIBLE_DEVICES={}".format(ctx.nvidia_visible_devices),
        ]
        for key, value in stage_environment(stage, ctx).items():
            command += ["-e", "{}={}".format(key, value)]
        command += [
            container,
            "bash",
            "-xe",
            "{}/{}".format(ctx.source_dir, script_name),
        ]
        run_command(command, ctx.dry_run)

    def collect(self):
        # Already on the host filesystem: enroot bind-mounts /tmp, so the tree
        # was written straight to where the caller wanted it.
        ctx = self.ctx
        if ctx.output_dir != DEFAULT_OUTPUT_DIR:
            log_warning(
                "--output-dir is a docker-only option; the enroot stages build "
                "in place, so the tree stays under {}".format(ctx.build_dir)
            )
        if ctx.archive and ctx.archive_dir:
            produced = pathlib.Path(ctx.container_archive_dir())
            wanted = pathlib.Path(ctx.archive_dir) / ctx.archive_basename()
            if produced != wanted and produced.is_dir() and not ctx.dry_run:
                log_status("copy: archives to {}".format(wanted))
                shutil.copytree(str(produced), str(wanted), dirs_exist_ok=True)
        return pathlib.Path(ctx.build_dir) / ctx.triton_version

    def cleanup(self):
        ctx = self.ctx
        if not ctx.cleanup:
            return
        log_status("enroot: removing containers and imported images")
        for container in self.containers:
            run_command(["enroot", "remove", "-f", container], ctx.dry_run, check=False)
        if not ctx.dry_run:
            for sqsh in self.images.values():
                try:
                    os.unlink(sqsh)
                except OSError:
                    pass


def select_runtime(ctx):
    """Pick an engine, reproducing the shell driver's precedence.

    docker first if enabled and installed, else enroot. A forced choice that is
    not installed is an error rather than a silent fallback.
    """
    have_docker = shutil.which("docker") is not None
    have_enroot = shutil.which("enroot") is not None

    if ctx.runtime == "docker":
        if not have_docker:
            raise GenerationError(
                "--runtime docker requested but docker is not installed"
            )
        return DockerRuntime(ctx)
    if ctx.runtime == "enroot":
        if not have_enroot:
            raise GenerationError(
                "--runtime enroot requested but enroot is not installed"
            )
        return EnrootRuntime(ctx)

    if ctx.use_docker and have_docker:
        log_info("Docker is installed.")
        return DockerRuntime(ctx)
    if ctx.use_enroot and have_enroot:
        log_info("NVIDIA Enroot is installed.")
        return EnrootRuntime(ctx)
    raise GenerationError(
        "Neither Docker nor NVIDIA Enroot is available. Install one of them, "
        "or pass --runtime to force a choice."
    )


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


class Context:
    """Resolved configuration: flags win, environment is the default."""

    def __init__(self, args):
        self.triton_version = args.triton_version
        self.onnx_version = args.onnx_version
        self.onnx_opset = args.onnx_opset
        self.openvino_version = args.openvino_version
        self.ubuntu_image = args.ubuntu_image
        self.pytorch_image = (
            args.pytorch_image
            or "nvcr.io/nvidia/pytorch:{}-py3".format(self.triton_version)
        )
        self.tensorrt_image = (
            args.tensorrt_image
            or "nvcr.io/nvidia/tensorrt:{}-py3".format(self.triton_version)
        )
        self.model_type = args.model_type
        self.job_id = args.job_id
        self.runner_id = args.runner_id
        self.project_name = args.project_name
        self.docker_volume = args.docker_volume or (
            "volume.gen_qa_model_repository.{}".format(self.job_id)
        )
        self.runtime = args.runtime
        self.use_docker = args.use_docker
        self.use_enroot = args.use_enroot
        self.dry_run = args.dry_run
        self.output_dir = args.output_dir
        self.is_ci = bool(os.environ.get("CI"))
        self.archive = args.archive
        self.archive_dir = args.archive_dir
        self.cleanup = args.cleanup
        self.clean_build_dir = args.clean_build_dir
        self.nvidia_visible_devices = args.nvidia_visible_devices

        # Filled in once the engine is known: the mount root differs, and the
        # build directory hangs off it.
        self.runtime_name = None
        self.build_dir = args.build_dir
        self.source_dir = None
        self.docker_gpu_args = resolve_gpu_args(args)

    def bind_runtime(self, runtime):
        self.runtime_name = runtime.name
        if self.build_dir is None:
            self.build_dir = "{}/{}".format(runtime.mount_root, self.job_id)
        self.source_dir = "{}/gen_srcdir".format(self.build_dir)

    def repository_paths(self):
        root = "{}/{}".format(self.build_dir, self.triton_version)
        return ["{}/{}".format(root, name) for name in REPOSITORY_DIRS]

    def container_archive_dir(self):
        """Archives live beside the model tree, never inside it.

        An archive written under the tree would be picked up by the next walk
        of it -- measured as part of a model, packed into the next archive --
        so the separation is structural, not tidiness.
        """
        return "{}/{}-archives".format(self.build_dir, self.triton_version)

    def archive_basename(self):
        return "{}-archives".format(self.triton_version)


def resolve_gpu_args(args):
    """The docker GPU flags, reproducing the shell driver's DOCKER_GPU_ARGS."""
    declared = os.environ.get("DOCKER_GPU_ARGS")
    if declared:
        return shlex.split(os.path.expandvars(declared))
    runner_gpus = os.environ.get("RUNNER_GPUS", "")
    if runner_gpus[:1].isdigit():
        # The shell driver `eval`s NV_DOCKER_ARGS here. Expanding variables and
        # splitting is the same thing for every value CI actually sets, without
        # handing the environment to a shell.
        return shlex.split(os.path.expandvars(os.environ.get("NV_DOCKER_ARGS", "")))
    return [
        "--runtime=nvidia",
        "-e",
        "NVIDIA_VISIBLE_DEVICES={}".format(args.nvidia_visible_devices),
    ]


def env_default(name, fallback=None):
    value = os.environ.get(name)
    return value if value else fallback


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every option defaults to the environment variable the shell "
        "driver reads, so existing CI works unchanged.",
    )

    stages = parser.add_argument_group(
        "frameworks",
        "Select what to build. With none given, all four run, in the fixed "
        "order OpenVINO -> ONNX -> PyTorch -> TensorRT.",
    )
    stages.add_argument("--openvino", action="store_true", help="build OpenVINO models")
    stages.add_argument("--onnx", action="store_true", help="build ONNX models")
    stages.add_argument(
        "--pytorch", action="store_true", help="build LibTorch/AOTI models"
    )
    stages.add_argument("--tensorrt", action="store_true", help="build TensorRT models")
    stages.add_argument("--all", action="store_true", help="build all four (default)")

    images = parser.add_argument_group("images", "Per-framework image overrides")
    images.add_argument(
        "--ubuntu-image",
        default=env_default("UBUNTU_IMAGE", DEFAULT_UBUNTU_IMAGE),
        help="image for the OpenVINO and ONNX stages [UBUNTU_IMAGE]",
    )
    images.add_argument(
        "--pytorch-image",
        default=env_default("PYTORCH_IMAGE"),
        help="image for the PyTorch stage [PYTORCH_IMAGE]",
    )
    images.add_argument(
        "--tensorrt-image",
        default=env_default("TENSORRT_IMAGE"),
        help="image for the TensorRT stage [TENSORRT_IMAGE]",
    )

    versions = parser.add_argument_group("versions")
    versions.add_argument(
        "--triton-version",
        default=env_default("TRITON_VERSION", DEFAULT_TRITON_VERSION),
        help="container train, names the output directory [TRITON_VERSION]",
    )
    versions.add_argument(
        "--onnx-version",
        default=env_default("ONNX_VERSION", DEFAULT_ONNX_VERSION),
        help="onnx package version [ONNX_VERSION]",
    )
    versions.add_argument(
        "--onnx-opset",
        default=env_default("ONNX_OPSET", DEFAULT_ONNX_OPSET),
        help="opset passed to the generators, 0 means the default [ONNX_OPSET]",
    )
    versions.add_argument(
        "--openvino-version",
        default=env_default("OPENVINO_VERSION", DEFAULT_OPENVINO_VERSION),
        help="openvino package version [OPENVINO_VERSION]",
    )

    engine = parser.add_argument_group("container engine")
    engine.add_argument(
        "--runtime",
        choices=["auto", "docker", "enroot"],
        default="auto",
        help="force an engine; auto follows the shell driver's precedence",
    )
    engine.add_argument(
        "--use-docker",
        type=int,
        default=int(env_default("TRITON_MODELS_USE_DOCKER", "1")),
        help="allow docker in auto mode [TRITON_MODELS_USE_DOCKER]",
    )
    engine.add_argument(
        "--use-enroot",
        type=int,
        default=int(env_default("TRITON_MODELS_USE_ENROOT", "1")),
        help="allow enroot in auto mode [TRITON_MODELS_USE_ENROOT]",
    )
    engine.add_argument(
        "--nvidia-visible-devices",
        default=env_default("NVIDIA_VISIBLE_DEVICES", DEFAULT_NVIDIA_VISIBLE_DEVICES),
        help="GPU to target [NVIDIA_VISIBLE_DEVICES]",
    )
    engine.add_argument(
        "--docker-volume",
        default=env_default("DOCKER_VOLUME"),
        help="named volume to build in [DOCKER_VOLUME]",
    )

    layout = parser.add_argument_group("layout")
    layout.add_argument(
        "--build-dir",
        default=env_default("TRITON_MDLS_BLD_DIR"),
        help="build root inside the container; defaults to <mount>/<job id> "
        "[TRITON_MDLS_BLD_DIR]",
    )
    layout.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="where docker copies the finished tree; docker only, the enroot "
        "stages build in place (default: {})".format(DEFAULT_OUTPUT_DIR),
    )
    layout.add_argument(
        "--model-type",
        default=env_default("MODEL_TYPE", ""),
        help="'igpu' skips TensorRT entirely and torch_tensorrt within "
        "PyTorch [MODEL_TYPE]",
    )
    layout.add_argument(
        "--job-id",
        default=env_default(
            "CI_JOB_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ),
        help="isolates concurrent runs [CI_JOB_ID]",
    )
    layout.add_argument(
        "--runner-id", default=env_default("RUNNER_ID", "0"), help="[RUNNER_ID]"
    )
    layout.add_argument(
        "--project-name",
        default=env_default("PROJECT_NAME", DEFAULT_PROJECT_NAME),
        help="[PROJECT_NAME]",
    )

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--archive",
        action="store_true",
        help="once generation finishes, pack each model into its own archive "
        "under <version>-archives, with an index.json describing them all",
    )
    behaviour.add_argument(
        "--archive-dir",
        default=None,
        help="where the archive folder is placed (default: alongside the "
        "generated tree)",
    )
    behaviour.add_argument(
        "--list", action="store_true", help="print what would run, then exit"
    )
    behaviour.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command and the stage scripts without running them",
    )
    behaviour.add_argument(
        "--cleanup",
        action="store_true",
        default=None,
        help="remove containers, volumes and imported images afterwards",
    )
    behaviour.add_argument(
        "--no-cleanup", dest="cleanup", action="store_false", help=argparse.SUPPRESS
    )
    behaviour.add_argument(
        "--clean-build-dir",
        action="store_true",
        default=True,
        help="remove an existing build directory first (enroot only)",
    )
    behaviour.add_argument(
        "--no-clean-build-dir", dest="clean_build_dir", action="store_false"
    )

    args = parser.parse_args(argv)

    selected = [key for key in STAGE_ORDER if getattr(args, key)]
    if args.all or not selected:
        selected = list(STAGE_ORDER)
    if args.model_type == "igpu" and "tensorrt" in selected:
        log_warning("MODEL_TYPE=igpu: skipping the TensorRT stage")
        selected.remove("tensorrt")
    args.selected = selected

    if args.cleanup is None:
        # Match the shell driver: docker tears its volume down outside CI,
        # enroot leaves its containers in place.
        args.cleanup = args.runtime != "enroot" and not os.environ.get("CI")

    return args


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def describe_plan(ctx, stages, selected):
    print("")
    print("runtime          : {}".format(ctx.runtime_name or ctx.runtime))
    print("triton version   : {}".format(ctx.triton_version))
    print("build dir        : {}".format(ctx.build_dir))
    print("source dir       : {}".format(ctx.source_dir))
    print("model type       : {}".format(ctx.model_type or "(unset)"))
    print("stages           : {}".format(", ".join(selected)))
    for key in selected:
        stage = stages[key]
        print("")
        print("=== {} ({}) ===".format(stage.title, stage.image))
        for step in stage.steps:
            print("    {}".format(step.describe()))
    print("")


def write_stage_scripts(ctx, stages, selected, directory):
    """Render each selected stage into `directory`, returning key -> path.

    Written outside the checkout on purpose. The shell driver writes its
    generated scripts into the source tree and deletes them as it goes, which
    leaves debris behind whenever a stage fails.
    """
    scripts = collections.OrderedDict()
    for index, key in enumerate(selected):
        stage = stages[key]
        name = "gen.{}.gen_qa_model_repository.sh".format(stage.title)
        path = pathlib.Path(directory) / name
        path.write_text(
            render_stage_script(stage, ctx, final=(index == len(selected) - 1))
        )
        path.chmod(0o755)
        scripts[key] = path
    return scripts


def main(argv=None):
    args = parse_args(argv)
    ctx = Context(args)
    selected = args.selected

    if not selected:
        log_error("no stages selected")
        return 1

    try:
        runtime = select_runtime(ctx)
    except GenerationError as error:
        log_error(str(error))
        return 1
    ctx.bind_runtime(runtime)

    stages = build_stages(ctx)

    if args.list:
        describe_plan(ctx, stages, selected)
        return 0

    log_status("stages to run: {}".format(", ".join(selected)))

    staging = pathlib.Path(
        os.environ.get("TMPDIR", "/tmp")
    ) / "gen_qa_model_repository.{}".format(ctx.job_id)
    staging.mkdir(parents=True, exist_ok=True)
    scripts = write_stage_scripts(ctx, stages, selected, staging)
    log_status("stage scripts written to {}".format(staging))

    if ctx.dry_run:
        describe_plan(ctx, stages, selected)
        for key, path in scripts.items():
            print("=" * 72)
            print("--- {} ---".format(path))
            print("=" * 72)
            print(path.read_text())

    try:
        runtime.prepare(scripts)
        for key in selected:
            stage = stages[key]
            log_status("run: {} stage in {}".format(stage.title, stage.image))
            runtime.run_stage(stage, scripts[key].name)
        tree = runtime.collect()
    except GenerationError as error:
        log_error(str(error))
        return 1
    finally:
        try:
            runtime.cleanup()
        except GenerationError as error:
            log_warning("cleanup: {}".format(error))

    if tree is not None:
        log_status("models generated in {}".format(tree))
    log_status("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
