# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from torch import nn


class AddSubNet(nn.Module):
    def __init__(self):
        super(AddSubNet, self).__init__()

    def forward(self, input0, input1):
        return (input0 + input1), (input0 - input1)


def generate_model(model_dir):
    model = AddSubNet()

    traced_model = torch.jit.trace(
        model,
        (torch.rand(1, 4, dtype=torch.float), torch.rand(1, 4, dtype=torch.float)),
    )

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")

    traced_model.save(model_path)


def generate_config(config_path):
    with open(f"{config_path}/config.pbtxt", "w") as f:
        f.write(
            """
backend: "pytorch"
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
input [
  {
    name: "INPUT1"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
output [
  {
    name: "OUTPUT1"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
"""
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-directory",
        type=str,
        required=True,
        help="The path to the model repository.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="add_sub_pytorch",
        help="Model name",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default="1",
        help="Model version",
    )

    args = parser.parse_args()

    model_directory = os.path.join(args.model_directory, args.model_name)
    os.makedirs(model_directory, exist_ok=True)

    generate_model(model_dir=os.path.join(model_directory, args.version))
    generate_config(model_directory)
