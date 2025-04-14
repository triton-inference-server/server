#!/usr/bin/env python
# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def gen_pytorch_model(name, batch_size):
    class PyAddSubNet(nn.Module):
        """
        Simple AddSub network in PyTorch. This network outputs the sum and
        subtraction of the inputs.
        """

        def __init__(self):
            super(PyAddSubNet, self).__init__()

        def forward(self, input0, input1):
            return torch.sub(input0, input1, alpha=-1), torch.sub(
                input0, input1, alpha=1
            )

    model = PyAddSubNet()
    model.eval()
    batch_size = 1
    example_inputs = torch.zeros([8, 4], dtype=torch.int64), torch.zeros(
        [8, 4], dtype=torch.int64
    )
    model_neuron = torch_neuron.trace(model, example_inputs, dynamic_batch_size=True)
    model_neuron.save("{}.pt".format(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["pytorch"],
        help="""The type of the compiled model. Currently,
                        only supports \"pytorch\".""",
    )
    parser.add_argument(
        "--name", type=str, required=True, help="The name of the compiled model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for the compiled model",
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception("Unrecognized options: {}".format(unparsed))
    elif FLAGS.model_type == "pytorch":
        import torch
        import torch_neuron
        from torch import nn

        gen_pytorch_model(FLAGS.name, FLAGS.batch_size)
