#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import requests
import test_util as tu
import torch
import torch.onnx
import torchvision.models as models

LABELS_URL = "https://raw.githubusercontent.com/triton-inference-server/python_backend/main/examples/preprocessing/model_repository/resnet50_trt/labels.txt"


def download_labels_file(url, path, file_name="labels.txt"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(path, file_name), "wb") as file:
            file.write(response.content)
    else:
        print(
            f"Failed to download file from {url}. Status code: {response.status_code}"
        )


def create_onnx_model_config(
    name,
    batch_size,
    input_name,
    input_shape,
    output_name,
    output_shape,
    label_filename,
    config_dir,
    config_name="config.pbtxt",
):
    config = """name: "{}"
max_batch_size: {}
platform: "onnxruntime_onnx"
input [
  {{
    name: "{}"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ {} ]
  }}
]
output [
  {{
    name: "{}"
    data_type: TYPE_FP32
    dims: [ {} ]
    label_filename: "{}"
  }}
]
""".format(
        name,
        batch_size,
        input_name,
        tu.shape_to_dims_str(input_shape),
        output_name,
        tu.shape_to_dims_str(output_shape),
        label_filename,
    )
    with open(f"{config_dir}/{config_name}", "w") as cfile:
        cfile.write(config)


def export_vgg19(models_dir, model_name="model.onnx"):
    model_path = f"{models_dir}/{model_name}"
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"VGG19 model exported to: {model_path}")
    create_onnx_model_config(
        "vgg19_onnx",
        32,
        "input",
        (3, 224, 224),
        "output",
        (1000,),
        "labels.txt",
        os.path.dirname(models_dir),
    )
    download_labels_file(LABELS_URL, os.path.dirname(models_dir))


def export_resnet152(models_dir, model_name="model.onnx"):
    model_path = f"{models_dir}/{model_name}"
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ResNet-152 model exported to: {model_path}")
    create_onnx_model_config(
        "resnet152_onnx",
        32,
        "input",
        (3, 224, 224),
        "output",
        (1000,),
        "labels.txt",
        os.path.dirname(models_dir),
    )
    download_labels_file(LABELS_URL, os.path.dirname(models_dir))


def export_resnet50(models_dir, model_name="model.onnx"):
    model_path = f"{models_dir}/{model_name}"
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ResNet-50 model exported to: {model_path}")
    create_onnx_model_config(
        "resnet50_onnx",
        32,
        "input",
        (3, 224, 224),
        "output",
        (1000,),
        "labels.txt",
        os.path.dirname(models_dir),
    )
    download_labels_file(LABELS_URL, os.path.dirname(models_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pre-trained models to ONNX format"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory to save the ONNX models",
    )
    parser.add_argument(
        "--resnet152", action="store_true", help="Export ResNet-152 model"
    )
    parser.add_argument(
        "--resnet50", action="store_true", help="Export ResNet-50 model"
    )
    parser.add_argument("--vgg19", action="store_true", help="Export VGG19 model")
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.resnet152:
        models_dir = os.path.join(FLAGS.models_dir, "resnet152_onnx/1")
        os.makedirs(models_dir, exist_ok=True)
        export_resnet152(models_dir)
    if FLAGS.resnet50:
        models_dir = os.path.join(FLAGS.models_dir, "resnet50_onnx/1")
        os.makedirs(models_dir, exist_ok=True)
        export_resnet50(models_dir)
    if FLAGS.vgg19:
        models_dir = os.path.join(FLAGS.models_dir, "vgg19_onnx/1")
        os.makedirs(models_dir, exist_ok=True)
        export_vgg19(models_dir)
