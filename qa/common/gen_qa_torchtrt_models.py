# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torchvision
import torch_tensorrt


def create_resnet50_torchtrt(models_dir, max_batch):
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    example_input = torch.rand(1, 3, 224, 224, dtype=torch.float)

    resnet50_ts = torch.jit.trace(model, example_input)

    trt_ts_module = torch_tensorrt.compile(
        resnet50_ts,
        inputs=[
            torch_tensorrt.Input(min_shape=[1, 3, 224, 224],
                                 opt_shape=[1, 3, 224, 224],
                                 max_shape=[max_batch, 3, 224, 224],
                                 dtype=torch.float)
        ],
        enabled_precisions={torch.float},
    )

    model_name = "resnet50_libtorch"

    model_version = 1
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    torch.jit.save(trt_ts_module, model_version_dir + "/model.pt")


def create_resnet50_torchtrt_modelconfig(models_dir, max_batch):

    model_name = "resnet50_libtorch"
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
backend: "pytorch"
max_batch_size: {}
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    label_filename: "resnet50_labels.txt"
  }}
]
'''.format(model_name, max_batch)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    FLAGS, unparsed = parser.parse_known_args()

    create_resnet50_torchtrt(FLAGS.models_dir, 128)
    create_resnet50_torchtrt_modelconfig(FLAGS.models_dir, 128)
