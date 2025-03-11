#!/usr/bin/env python3

# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FLAGS = None

def create_moduloop_modelfile(models_dir, model_version):
    model_name = "libtorch_modulo"

    op_source = """
    #include <torch/script.h>
    torch::Tensor custom_modulo(torch::Tensor input1, torch::Tensor input2) {
      torch::Tensor output = torch::fmod(input1, input2);
      return output.clone();
    }
    static auto registry =
      torch::RegisterOperators("my_ops::custom_modulo", &custom_modulo);
    """

    torch.utils.cpp_extension.load_inline(
        name="custom_modulo",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )

    class ModuloCustomNet(nn.Module):
        def __init__(self):
            super(ModuloCustomNet, self).__init__()

        def forward(self, input0, input1):
            return torch.ops.my_ops.custom_modulo(input0, input1)

    moduloCustomModel = ModuloCustomNet()
    example_input0 = torch.arange(1, 11, dtype=torch.float32)
    example_input1 = torch.tensor([2] * 10, dtype=torch.float32)
    traced = torch.jit.trace(moduloCustomModel, (example_input0, example_input1))

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    traced.save(model_version_dir + "/model.pt")


def create_moduloop_modelconfig(models_dir, model_version):
    model_name = "libtorch_modulo"
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }},
  {{
    name: "INPUT__1"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]
""".format(
        model_name
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


# Use Torchvision ops
def create_visionop_modelfile(models_dir, model_version):
    model_name = "libtorch_visionop"

    class CustomVisionNet(nn.Module):
        def __init__(self):
            super(CustomVisionNet, self).__init__()

        def forward(self, input, boxes):
            return torchvision.ops.roi_align(input, boxes, [5, 5], 1.0, -1, False)

    visionCustomModel = CustomVisionNet()
    visionCustomModel.eval()
    scripted = torch.jit.script(visionCustomModel)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    scripted.save(model_version_dir + "/model.pt")


def create_visionop_modelconfig(models_dir, model_version):
    model_name = "libtorch_visionop"
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 1, 3, 10, 10 ]
  }},
  {{
    name: "INPUT__1"
    data_type: TYPE_FP32
    dims: [1, 5]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [1, 3, 5, 5]
  }}
]
""".format(
        model_name
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

def create_modulo_op_models(models_dir):
    model_version = 1

    if FLAGS.libtorch:
        create_moduloop_modelconfig(models_dir, model_version)
        create_moduloop_modelfile(models_dir, model_version)


def create_vision_op_models(models_dir):
    model_version = 1

    if FLAGS.libtorch:
        create_visionop_modelconfig(models_dir, model_version)
        create_visionop_modelfile(models_dir, model_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    parser.add_argument(
        "--zero_out_lib_path",
        type=str,
        required=False,
        default="./libzeroout.so",
        help="Fullpath to libzeroout.so",
    )
    parser.add_argument(
        "--cuda_op_lib_path",
        type=str,
        required=False,
        default="./libcudaop.so",
        help="Fullpath to libcudaop.so",
    )
    parser.add_argument(
        "--busy_op_lib_path",
        type=str,
        required=False,
        default="./libbusyop.so",
        help="Fullpath to libbusyop.so",
    )
    parser.add_argument(
        "--libtorch",
        required=False,
        action="store_true",
        help="Generate Pytorch LibTorch models",
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.libtorch:
        import torch
        import torch.utils.cpp_extension
        import torchvision
        from torch import nn

        create_modulo_op_models(FLAGS.models_dir)
        create_vision_op_models(FLAGS.models_dir)
