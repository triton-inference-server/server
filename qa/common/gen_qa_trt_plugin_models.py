# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def np_to_model_dtype(np_dtype):
    if np_dtype == bool:
        return "TYPE_BOOL"
    elif np_dtype == np.int8:
        return "TYPE_INT8"
    elif np_dtype == np.int16:
        return "TYPE_INT16"
    elif np_dtype == np.int32:
        return "TYPE_INT32"
    elif np_dtype == np.int64:
        return "TYPE_INT64"
    elif np_dtype == np.uint8:
        return "TYPE_UINT8"
    elif np_dtype == np.uint16:
        return "TYPE_UINT16"
    elif np_dtype == np.float16:
        return "TYPE_FP16"
    elif np_dtype == np.float32:
        return "TYPE_FP32"
    elif np_dtype == np.float64:
        return "TYPE_FP64"
    elif np_dtype == np_dtype_string:
        return "TYPE_STRING"
    return None


def np_to_trt_dtype(np_dtype):
    if np_dtype == bool:
        return trt.bool
    elif np_dtype == np.int8:
        return trt.int8
    elif np_dtype == np.int32:
        return trt.int32
    elif np_dtype == np.float16:
        return trt.float16
    elif np_dtype == np.float32:
        return trt.float32
    return None


def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if (plugin_creator.name == "Normalize_TRT") and \
                (plugin_name == "Normalize_TRT"):
            nbWeights = trt.PluginField("nbWeights",
                                        np.array([1], dtype=np.int32),
                                        trt.PluginFieldType.INT32)
            eps = trt.PluginField("eps", np.array([0.00001], dtype=np.float32),
                                  trt.PluginFieldType.FLOAT32)
            weights = trt.PluginField('weights',
                                      np.array([1] * 16, dtype=np.float32),
                                      trt.PluginFieldType.FLOAT32)
            field_collection = trt.PluginFieldCollection(
                [weights, eps, nbWeights])
            plugin = plugin_creator.create_plugin(
                name=plugin_name, field_collection=field_collection)
            break
        elif (plugin_creator.name
              == "CustomGeluPluginDynamic") and (plugin_name
                                                 == "CustomGeluPluginDynamic"):
            type_id = trt.PluginField("type_id", np.array([0], np.int32),
                                      trt.PluginFieldType.INT32)
            bias = trt.PluginField("bias", np.array([[[1]]], np.float32),
                                   trt.PluginFieldType.FLOAT32)
            field_collection = trt.PluginFieldCollection([type_id, bias])
            plugin = plugin_creator.create_plugin(
                name=plugin_name, field_collection=field_collection)
            break
    return plugin


def create_plan_modelfile(models_dir, max_batch, model_version, plugin_name,
                          input_shape, output0_shape, input_dtype,
                          output0_dtype):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output0_dtype,
                                     input_shape, output0_shape, output0_shape):
        return

    trt_input_dtype = np_to_trt_dtype(input_dtype)

    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype,
                                   output0_dtype) + '_' + plugin_name

    # using explicit batch is necessary for CustomGeluPluginDynamic
    if plugin_name == "CustomGeluPluginDynamic":
        explicit_batch = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                explicit_batch) as network:
            input_layer = network.add_input(name="INPUT0",
                                            dtype=trt_input_dtype,
                                            shape=input_shape)
            plugin_layer = network.add_plugin_v2(
                inputs=[input_layer], plugin=get_trt_plugin(plugin_name))
            plugin_layer.get_output(0).name = "OUTPUT0"
            network.mark_output(plugin_layer.get_output(0))

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20

            try:
                engine_bytes = builder.build_serialized_network(network, config)
            except AttributeError:
                engine = builder.build_engine(network, config)
                engine_bytes = engine.serialize()
                del engine

            model_version_dir = models_dir + "/" + model_name + "/" + str(
                model_version)

            try:
                os.makedirs(model_version_dir)
            except OSError as ex:
                pass  # ignore existing dir

            with open(model_version_dir + "/model.plan", "wb") as f:
                f.write(engine_bytes)
    else:
        with trt.Builder(
                TRT_LOGGER) as builder, builder.create_network() as network:
            input_layer = network.add_input(name="INPUT0",
                                            dtype=trt_input_dtype,
                                            shape=input_shape)
            plugin_layer = network.add_plugin_v2(
                inputs=[input_layer], plugin=get_trt_plugin(plugin_name))
            plugin_layer.get_output(0).name = "OUTPUT0"
            network.mark_output(plugin_layer.get_output(0))

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20
            builder.max_batch_size = max(1, max_batch)

            try:
                engine_bytes = builder.build_serialized_network(network, config)
            except AttributeError:
                engine = builder.build_engine(network, config)
                engine_bytes = engine.serialize()
                del engine

            model_version_dir = models_dir + "/" + model_name + "/" + str(
                model_version)

            try:
                os.makedirs(model_version_dir)
            except OSError as ex:
                pass  # ignore existing dir

            with open(model_version_dir + "/model.plan", "wb") as f:
                f.write(engine_bytes)


def create_plan_modelconfig(models_dir, max_batch, model_version, plugin_name,
                            input_shape, output0_shape, input_dtype,
                            output0_dtype):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output0_dtype,
                                     input_shape, output0_shape, output0_shape):
        return

    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype,
                                   output0_dtype) + '_' + plugin_name
    config_dir = models_dir + "/" + model_name
    config = '''
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
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_plugin_models(models_dir):
    model_version = 1

    # default CustomGeluPluginDynamic plugin
    create_plan_modelconfig(models_dir, 0, model_version,
                            "CustomGeluPluginDynamic", (16, 1, 1), (16, 1, 1),
                            np.float32, np.float32)
    create_plan_modelfile(models_dir, 0, model_version,
                          "CustomGeluPluginDynamic", (16, 1, 1), (16, 1, 1),
                          np.float32, np.float32)

    # custom Normalize_TRT
    create_plan_modelconfig(models_dir, 8, model_version, "Normalize_TRT", (
        16,
        16,
        16,
    ), (
        16,
        16,
        16,
    ), np.float32, np.float32)
    create_plan_modelfile(models_dir, 8, model_version, "Normalize_TRT", (
        16,
        16,
        16,
    ), (
        16,
        16,
        16,
    ), np.float32, np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    FLAGS, unparsed = parser.parse_known_args()

    import test_util as tu

    create_plugin_models(FLAGS.models_dir)
