#!/usr/bin/python
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.nn as nn


class SumModule(nn.Module):
    def __init__(self, device):
        super(SumModule, self).__init__()
        self.device = device

    def forward(self, INPUT0, INPUT1):
        INPUT0 = INPUT0.to(self.device)
        INPUT1 = INPUT1.to(self.device)
        print(
            "SumModule - INPUT0 device: {}, INPUT1 device: {}\n".format(
                INPUT0.device, INPUT1.device
            )
        )
        return INPUT0 + INPUT1


class DiffModule(nn.Module):
    def __init__(self, device):
        super(DiffModule, self).__init__()
        self.device = device

    def forward(self, INPUT0, INPUT1):
        INPUT0 = INPUT0.to(self.device)
        INPUT1 = INPUT1.to(self.device)
        print(
            "DiffModule - INPUT0 device: {}, INPUT1 device: {}\n".format(
                INPUT0.device, INPUT1.device
            )
        )
        return INPUT0 - INPUT1


class TestModel(nn.Module):
    def __init__(self, device0, device1):
        super(TestModel, self).__init__()
        self.device0 = device0
        self.device1 = device1

        self.layer1 = SumModule(self.device0)
        self.layer2 = DiffModule(self.device1)

    def forward(self, INPUT0, INPUT1):
        op0 = self.layer1(INPUT0, INPUT1)
        op1 = self.layer2(INPUT0, INPUT1)
        return op0, op1


if torch.cuda.device_count() < 2:
    print("Need at least 2 GPUs to run this test")
    exit(1)

devices = [("cuda:1", "cuda:0"), ("cpu", "cuda:1")]
model_names = ["libtorch_multi_gpu", "libtorch_multi_device"]

for device_pair, model_name in zip(devices, model_names):
    model = TestModel(device_pair[0], device_pair[1])
    model_path = "models/" + model_name + "/1/model.pt"
    scripted_model = torch.jit.script(model)
    scripted_model.save(model_path)
