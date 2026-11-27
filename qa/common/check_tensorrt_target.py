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

"""Does the TensorRT in this container have kernels for the GPU we are on?

A GPU newer than the TensorRT that ships beside it is the case this exists
for: every generator that calls `build_serialized_network` then gets a null
plan back, and the ones that write it unchecked die on a TypeError that says
nothing about the real cause.

Two users, both reading the same answer from TensorRT itself:

  * `UNSUPPORTED_TARGET_MARKER` is what gen_qa_model_repository.py's per-step
    wrapper greps a failed generator's output for, to tell "this GPU is out
    of TensorRT's reach" from "this generator is broken".
  * `unsupported_target_reason()` asks the question up front, which
    gen_qa_torchtrt_models.py does to skip before downloading model weights
    it would only throw away.
"""

import subprocess
import sys

# The escapes gen_qa_model_repository.py uses, so a warning raised here reads
# the same in a CI log as one raised by the driver around it.
COLOR_WARNING = "\033[33m"
COLOR_RESET = "\033[0m"

# How TensorRT names a target it has no kernels for, e.g. "Target GPU SM 107
# is not supported by this TensorRT release."
UNSUPPORTED_TARGET_MARKER = "is not supported by this TensorRT release"


def warn(message):
    print(
        "{}[WARNING] {}{}".format(COLOR_WARNING, message, COLOR_RESET),
        file=sys.stderr,
        flush=True,
    )


def compute_capability():
    """The running GPU's compute capability, in TensorRT's dotted spelling.

    Read from nvidia-smi rather than torch: this runs in the TensorRT stage's
    container too, which has no torch in it.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _probe_unsupported_target():
    """TensorRT's own words for why it cannot build here, or None if it can.

    TensorRT publishes no list of the SMs it was compiled for, so the question
    goes to the builder instead of to a table this script would have to keep
    in step with each release: serialize a one-layer engine for the current
    device and read what the builder logs. The SM guard lives in
    IBuilder::buildSerializedNetwork -- which is what the error names -- so a
    trivial network trips it exactly as a real model does, in a fraction of
    the time.
    """
    import tensorrt as trt

    class _Recorder(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []

        def log(self, severity, message):
            self.messages.append(message)

    recorder = _Recorder()
    builder = trt.Builder(recorder)
    network = builder.create_network()
    identity = network.add_identity(
        network.add_input("input", trt.float32, (1, 1, 1, 1))
    )
    network.mark_output(identity.get_output(0))
    if builder.build_serialized_network(network, builder.create_builder_config()):
        return None
    for message in recorder.messages:
        if UNSUPPORTED_TARGET_MARKER in message:
            return message.strip()
    # The probe failed for some other reason. Report nothing and let the real
    # generators fail loudly, rather than turn an unrelated regression into a
    # silent skip of every TensorRT model.
    return None


def unsupported_target_reason():
    """_probe_unsupported_target, with the probe itself never fatal."""
    try:
        return _probe_unsupported_target()
    except Exception as error:
        # A gate that cannot run is not a gate. If tensorrt is absent, or its
        # builder API moved between releases, fall through to generation and
        # let that report the truth.
        warn("Could not ask TensorRT whether it supports this GPU: {}".format(error))
        return None
