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

# Build TensorRT plans for the torchvision image models.
#
# gen_qa_image_models.py exports <name>_onnx models in the PyTorch stage. This
# script runs afterwards in the TensorRT stage and builds a <name>_plan model
# next to each exported ONNX model, so the L0 tests copy a prebuilt plan
# instead of building one at run time with trtexec (which is no longer shipped
# in the default Triton container).

import argparse
import os
import re
import shutil
import sys

import gen_manifest
import tensorrt as trt

IMAGE_MODELS = ("resnet50", "resnet152", "vgg19")

# Dynamic batch profile, matching the one the tests previously passed to
# trtexec via --minShapes/--optShapes/--maxShapes.
INPUT_NAME = "input"
MIN_SHAPE = (1, 3, 224, 224)
OPT_SHAPE = (32, 3, 224, 224)
MAX_SHAPE = (32, 3, 224, 224)


def build_plan(onnx_path, plan_path):
    logger = trt.Logger(
        trt.Logger.VERBOSE if os.environ.get("TRT_VERBOSE") == "1" else trt.Logger.INFO
    )
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"ONNX parser error: {parser.get_error(i)}", file=sys.stderr)
        raise RuntimeError(f"Failed to parse {onnx_path}")

    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, MIN_SHAPE, OPT_SHAPE, MAX_SHAPE)
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build a TensorRT engine from {onnx_path}")
    with open(plan_path, "wb") as f:
        f.write(engine_bytes)


def create_plan_config(onnx_model_dir, plan_model_dir, name):
    # Derive the plan config from the ONNX one so the I/O definition has a
    # single source of truth (gen_qa_image_models.py).
    with open(os.path.join(onnx_model_dir, "config.pbtxt")) as f:
        config = f.read()
    config = re.sub(r'^name: ".*"$', f'name: "{name}_plan"', config, flags=re.M)
    config = re.sub(
        r'^platform: ".*"$', 'platform: "tensorrt_plan"', config, flags=re.M
    )
    with open(os.path.join(plan_model_dir, "config.pbtxt"), "w") as f:
        f.write(config)
    shutil.copy(
        os.path.join(onnx_model_dir, "labels.txt"),
        os.path.join(plan_model_dir, "labels.txt"),
    )


def create_plan_model(models_dir, name):
    onnx_model_dir = os.path.join(models_dir, f"{name}_onnx")
    onnx_path = os.path.join(onnx_model_dir, "1", "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"{onnx_path} not found; run gen_qa_image_models.py --{name} first"
        )

    plan_model_dir = os.path.join(models_dir, f"{name}_plan")
    os.makedirs(os.path.join(plan_model_dir, "1"), exist_ok=True)
    plan_path = os.path.join(plan_model_dir, "1", "model.plan")
    build_plan(onnx_path, plan_path)
    create_plan_config(onnx_model_dir, plan_model_dir, name)
    print(f"{name} TensorRT plan built at: {plan_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TensorRT plans from exported image ONNX models"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory holding the <name>_onnx models; plans are added beside them",
    )
    for name in IMAGE_MODELS:
        parser.add_argument(
            f"--{name}", action="store_true", help=f"Build the {name} plan"
        )
    FLAGS, unparsed = parser.parse_known_args()

    # Fingerprint the tree first, so emit_manifests() below stamps only the
    # models this script creates rather than relabelling the ONNX ones.
    manifest_baseline = gen_manifest.snapshot_model_dirs(FLAGS.models_dir)

    for name in IMAGE_MODELS:
        if getattr(FLAGS, name):
            create_plan_model(FLAGS.models_dir, name)

    # Record what produced these models, beside each config.pbtxt.
    gen_manifest.emit_manifests(FLAGS.models_dir, manifest_baseline)
