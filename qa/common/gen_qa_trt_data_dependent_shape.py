#!/usr/bin/env python3

# Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tensorrt as trt
import test_util as tu
from gen_common import np_to_model_dtype, np_to_trt_dtype


# The 'nonzero' model that we use for data dependent shape is naturally
# not support batching, because the layer output is not trivially separable
# based on the request batch size.
# input_shape is config shape
def create_data_dependent_modelfile(
    models_dir, model_name, input_shape, input_dtype=np.int32, min_dim=1, max_dim=32
):
    trt_input_dtype = np_to_trt_dtype(input_dtype)

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()

    # input
    in0 = network.add_input("INPUT", trt_input_dtype, input_shape)

    # layers
    non_zero = network.add_non_zero(in0)

    # configure output
    out0 = non_zero.get_output(0)
    out0.name = "OUTPUT"
    network.mark_output(out0)

    # optimization profile
    min_shape = []
    opt_shape = []
    max_shape = []
    for i in input_shape:
        if i == -1:
            min_shape = min_shape + [min_dim]
            opt_shape = opt_shape + [int((max_dim + min_dim) / 2)]
            max_shape = max_shape + [max_dim]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT", min_shape, opt_shape, max_shape)
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    # serialized model
    engine_bytes = builder.build_serialized_network(network, config)

    model_version_dir = models_dir + "/" + model_name + "/1"
    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_data_dependent_modelconfig(
    models_dir, model_name, input_shape, input_dtype=np.int32
):
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {{
    name: "INPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {}
    dims: [ {} ]
   }}
]
""".format(
        model_name,
        np_to_model_dtype(input_dtype),
        tu.shape_to_dims_str(input_shape),
        np_to_model_dtype(np.int32),
        tu.shape_to_dims_str((len(input_shape), -1)),
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    FLAGS, unparsed = parser.parse_known_args()

    # Fixed input shape
    create_data_dependent_modelfile(
        FLAGS.models_dir, "plan_nobatch_nonzero_fixed", (4, 4)
    )
    create_data_dependent_modelconfig(
        FLAGS.models_dir, "plan_nobatch_nonzero_fixed", (4, 4)
    )

    # Dynamic input shape
    create_data_dependent_modelfile(
        FLAGS.models_dir, "plan_nobatch_nonzero_dynamic", (-1, -1)
    )
    create_data_dependent_modelconfig(
        FLAGS.models_dir, "plan_nobatch_nonzero_dynamic", (-1, -1)
    )
