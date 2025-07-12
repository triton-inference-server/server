# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import json
import sys
import warnings

import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import *

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="resnet50",
        help="Model name",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        required=False,
        default="http://images.cocodataset.org/test2017/000000557146.jpg",
        help="Image URL. Default is:\
                            http://images.cocodataset.org/test2017/000000557146.jpg",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        required=False,
        default="./resnet50_labels.txt",
        help="Path to the file with text representation \
                        of available labels",
    )
    args = parser.parse_args()

    utils = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_convnets_processing_utils",
        skip_validation=True,
    )

    try:
        triton_client = httpclient.InferenceServerClient(args.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(args.label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    if args.verbose:
        print(json.dumps(triton_client.get_model_config(args.model_name), indent=4))

    input_name = "INPUT"
    output_name = "OUTPUT"
    batch = np.asarray(utils.prepare_input_from_uri(args.image_url))

    input = httpclient.InferInput(input_name, batch.shape, "FP32")
    output = httpclient.InferRequestedOutput(output_name)

    input.set_data_from_numpy(batch)
    results = triton_client.infer(
        model_name=args.model_name, inputs=[input], outputs=[output]
    )

    output_data = results.as_numpy(output_name)
    max_id = np.argmax(output_data, axis=1)[0]
    print("Results is class: {}".format(labels_dict[max_id]))

    print("PASS: ResNet50 instance kind")
    sys.exit(0)
