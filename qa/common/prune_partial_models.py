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

"""Delete the half-written models a refused TensorRT build leaves behind.

Every plan generator writes its config.pbtxt first and its engine second, and
serializes with

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)

so when the builder hands back a null plan, the open() has already created the
file before the write raises. What survives is a model directory that looks
complete -- a config.pbtxt beside a zero-byte model.plan -- which is worse
than one that is simply absent: nothing downstream opens the engine, so the
manifest records it, the archive ships it, and Triton is the first thing to
notice, at load time.

Run from gen_qa_model_repository.py's step wrapper after a step is skipped
because TensorRT has no kernels for this GPU. Takes the skipped step's own
command line and reads --models_dir out of it.

Deliberately narrow. A zero-byte model.plan is never a legitimate artifact, so
that alone is the signal; a plan with any content at all is left untouched.
Only version directories under the named --models_dir are considered, and the
model directory goes only once its last version has, since a config.pbtxt with
no version left to serve is equally unloadable.
"""

import os
import shutil
import sys

COLOR_WARNING = "\033[33m"
COLOR_RESET = "\033[0m"

ENGINE_NAME = "model.plan"


def warn(message):
    print(
        "{}[WARNING] {}{}".format(COLOR_WARNING, message, COLOR_RESET),
        file=sys.stderr,
        flush=True,
    )


def models_dir_from(argv):
    """The --models_dir the skipped step was given, or None."""
    for index, arg in enumerate(argv):
        if arg.startswith("--models_dir="):
            return arg.split("=", 1)[1]
        if arg == "--models_dir" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _contained(path, root):
    """Is path inside root? Guards the removals below against a stray path."""
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def prune(models_dir):
    """Remove every model whose engine was never actually written."""
    removed = []
    if not os.path.isdir(models_dir):
        return removed

    for model_name in sorted(os.listdir(models_dir)):
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        from_this_model = []
        for version in sorted(os.listdir(model_dir)):
            version_dir = os.path.join(model_dir, version)
            engine = os.path.join(version_dir, ENGINE_NAME)
            if not os.path.isfile(engine) or os.path.getsize(engine) != 0:
                continue
            if not _contained(version_dir, models_dir):
                continue
            shutil.rmtree(version_dir, ignore_errors=True)
            from_this_model.append(version_dir)

        removed.extend(from_this_model)
        if not from_this_model:
            continue

        # A config.pbtxt whose every version has gone cannot be served, so the
        # model goes with them -- but only when this model is the one that
        # lost them. A directory that was already empty is left alone.
        remaining = [
            entry
            for entry in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, entry))
        ]
        if not remaining and _contained(model_dir, models_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
            removed.append(model_dir)

    return removed


def main(argv):
    models_dir = models_dir_from(argv)
    if models_dir is None:
        return 0

    removed = prune(models_dir)
    if removed:
        warn(
            "Removed {} partially written model path(s) under {}:".format(
                len(removed), models_dir
            )
        )
        for path in removed:
            warn("  {}".format(path))
    return 0


if __name__ == "__main__":
    # Best-effort cleanup: never turn tidying up after a skip into the thing
    # that fails the stage.
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:  # noqa: BLE001
        warn("Could not prune partial models: {}".format(error))
        sys.exit(0)
