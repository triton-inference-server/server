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

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
    def initialize(self, args):
        """
        This function initializes pre-trained ResNet50 model,
        depending on the value specified by an `instance_group` parameter
        in `config.pbtxt`.

        Depending on what `instance_group` was specified in
        the config.pbtxt file (KIND_CPU or KIND_GPU), the model instance
        will be initialised on a cpu, a gpu, or both. If `instance_group` was
        not specified in the config file, then models will be loaded onto
        the default device of the framework.
        """
        # Here we set up the device onto which our model will beloaded,
        # based on specified `model_instance_kind` and `model_instance_device_id`
        # fields.
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        # This example is configured to work with torch=1.13
        # and torchvision=0.14. Thus, we need to provide a proper tag `0.14.1`
        # to make sure loaded Resnet50 is compatible with
        # installed `torchvision`.
        # Refer to README for installation instructions.
        self.model = (
            torch.hub.load(
                "pytorch/vision:v0.14.1",
                "resnet50",
                weights="IMAGENET1K_V2",
                skip_validation=True,
            )
            .to(self.device)
            .eval()
        )

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            with torch.no_grad():
                result = self.model(
                    torch.as_tensor(input_tensor.as_numpy(), device=self.device)
                )
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
