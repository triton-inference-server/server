#!/usr/bin/env python3
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

import argparse
import json
import numpy as np


def generate_input_data(num_inputs, batch_size, output_file):
    data = {"data": []}
    for _ in range(num_inputs):
        input_data = np.random.rand(batch_size, 1024).astype(np.float32)
        entry = {"INPUT0": input_data.flatten().tolist()}
        data["data"].append(entry)

    with open(output_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random input data for perf_analyzer."
    )
    parser.add_argument(
        "--num-inputs", type=int, help="Number of unique random inputs to generate."
    )
    parser.add_argument("--batch-size", type=int, help="The batch size for each input.")
    parser.add_argument(
        "--output-file", type=str, help="The name of the output JSON file."
    )
    args = parser.parse_args()

    generate_input_data(args.num_inputs, args.batch_size, args.output_file)
    print(f"Successfully generated {args.num_inputs} inputs in '{args.output_file}'.")
