#!/usr/bin/env python3

# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import os

import numpy as np
import tensorrt as trt
from gen_common import np_to_model_dtype, np_to_trt_dtype

np_dtype_string = np.dtype(object)

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def get_trt_plugin(plugin_name):
    plugin = None
    field_collection = None
    plugin_creators = trt.get_plugin_registry().plugin_creator_list
    for plugin_creator in plugin_creators:
        if (plugin_creator.name == "CustomHardmax") and (
            plugin_name == "CustomHardmax"
        ):
            axis_attr = trt.PluginField(
                "axis", np.array([0]), type=trt.PluginFieldType.INT32
            )
            field_collection = trt.PluginFieldCollection([axis_attr])
            break

    if field_collection is None:
        raise RuntimeError("Plugin not found: " + plugin_name)
    plugin = plugin_creator.create_plugin(
        name=plugin_name, field_collection=field_collection
    )

    return plugin


def create_plan_modelfile(
    models_dir,
    max_batch,
    model_version,
    plugin_name,
    input_shape,
    output0_shape,
    input_dtype,
    output0_dtype,
):
    if not tu.validate_for_trt_model(
        input_dtype,
        output0_dtype,
        output0_dtype,
        input_shape,
        output0_shape,
        output0_shape,
    ):
        return

    trt_input_dtype = np_to_trt_dtype(input_dtype)

    model_name = (
        tu.get_model_name(
            "plan_nobatch" if max_batch == 0 else "plan",
            input_dtype,
            output0_dtype,
            output0_dtype,
        )
        + "_"
        + plugin_name
    )

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    if max_batch == 0:
        input_with_batchsize = [i for i in input_shape]
    else:
        input_with_batchsize = [-1] + [i for i in input_shape]

    input_layer = network.add_input(
        name="INPUT0", dtype=trt_input_dtype, shape=input_with_batchsize
    )
    plugin_layer = network.add_plugin_v2(
        inputs=[input_layer], plugin=get_trt_plugin(plugin_name)
    )
    plugin_layer.get_output(0).name = "OUTPUT0"
    network.mark_output(plugin_layer.get_output(0))

    min_shape = []
    opt_shape = []
    max_shape = []
    for i in input_shape:
        min_shape = min_shape + [i]
        opt_shape = opt_shape + [i]
        max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    if max_batch == 0:
        profile.set_shape("INPUT0", min_shape, opt_shape, max_shape)
    else:
        profile.set_shape(
            "INPUT0",
            [1] + min_shape,
            [max_batch] + opt_shape,
            [max_batch] + max_shape,
        )

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


def create_plan_modelconfig(
    models_dir,
    max_batch,
    model_version,
    plugin_name,
    input_shape,
    output0_shape,
    input_dtype,
    output0_dtype,
):
    if not tu.validate_for_trt_model(
        input_dtype,
        output0_dtype,
        output0_dtype,
        input_shape,
        output0_shape,
        output0_shape,
    ):
        return

    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = (
        tu.get_model_name(
            "plan_nobatch" if max_batch == 0 else "plan",
            input_dtype,
            output0_dtype,
            output0_dtype,
        )
        + "_"
        + plugin_name
    )
    config_dir = models_dir + "/" + model_name
    config = """
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
   }}
]
""".format(
        model_name,
        max_batch,
        version_policy_str,
        np_to_model_dtype(input_dtype),
        tu.shape_to_dims_str(input_shape),
        np_to_model_dtype(output0_dtype),
        tu.shape_to_dims_str(output0_shape),
    )

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_plugin_models(models_dir):
    model_version = 1

    # custom CustomHardmax
    create_plan_modelconfig(
        models_dir,
        8,
        model_version,
        "CustomHardmax",
        (2, 2),
        (2, 2),
        np.float32,
        np.float32,
    )
    create_plan_modelfile(
        models_dir,
        8,
        model_version,
        "CustomHardmax",
        (2, 2),
        (2, 2),
        np.float32,
        np.float32,
    )

    create_plan_modelconfig(
        models_dir,
        0,
        model_version,
        "CustomHardmax",
        (16, 1, 1),
        (16, 1, 1),
        np.float32,
        np.float32,
    )
    create_plan_modelfile(
        models_dir,
        0,
        model_version,
        "CustomHardmax",
        (16, 1, 1),
        (16, 1, 1),
        np.float32,
        np.float32,
    )


def windows_load_plugin_lib(win_plugin_dll):
    if os.path.isfile(win_plugin_dll):
        try:
            ctypes.CDLL(win_plugin_dll, winmode=0)
        except TypeError:
            # winmode only introduced in python 3.8
            ctypes.CDLL(win_plugin_dll)
        return

    raise IOError('Failed to load library: "{}".'.format(win_plugin_dll))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    parser.add_argument(
        "--win_plugin_dll",
        type=str,
        required=False,
        default="",
        help="Path to Windows plugin .dll",
    )
    FLAGS, unparsed = parser.parse_known_args()

    import test_util as tu

    # Linux can leverage LD_PRELOAD. We must load the Windows plugin manually
    # in order for it to be discovered in the registry.
    if os.name == "nt":
        windows_load_plugin_lib(FLAGS.win_plugin_dll)

    create_plugin_models(FLAGS.models_dir)
