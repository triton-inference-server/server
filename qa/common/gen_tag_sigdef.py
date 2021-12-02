# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import sys

sys.path.append("../common")

import os
from builtins import range

from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow.compat.v1 as tf
import gen_ensemble_model_utils as gu
"""Create SaveModels that contains multiple tags and multiple signature defs"""


def create_savedmodel(models_dir,
                      model_version=1,
                      dims=16,
                      model_name="sig_tag",
                      tag_name="testTag",
                      signature_def_name="testSigDef"):
    """
    Creates 4 SavedModels that have different combinations of model_name and tag_name.
    The models multiplies the input tensor by a multiplier and the multiplier value is different for each model.
    Naming convention and config used:
    <model_name>0: tag: "serve",    signature_def: "serving_default",    multiplier 1
    <model_name>1: tag: "serve",    signature_def: <signature_def_name>, multiplier 2
    <model_name>2: tag: <tag_name>, signature_def: "serving_default",    multiplier 3
    <model_name>3: tag: <tag_name>, signature_def: <signature_def_name>, multiplier 4
    """
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, [dims], "TENSOR_INPUT")

        # tag:"serve", signature_def:"serving_default"
        multiplier_0 = tf.constant(1.0, name="multiplier_0")
        # tag:"serve", signature_def:signature_def_name
        multiplier_1 = tf.constant(2.0, name="multiplier_1")
        # tag:tag_name, signature_def:"serving_default"
        multiplier_2 = tf.constant(3.0, name="multiplier_2")
        # tag:tag_name, signature_def:signature_def_name
        multiplier_3 = tf.constant(4.0, name="multiplier_3")

        output_tensor_0 = tf.multiply(multiplier_0,
                                      input_tensor,
                                      name="TENSOR_OUTPUT")
        output_tensor_1 = tf.multiply(multiplier_1,
                                      input_tensor,
                                      name="TENSOR_OUTPUT")
        output_tensor_2 = tf.multiply(multiplier_2,
                                      input_tensor,
                                      name="TENSOR_OUTPUT")
        output_tensor_3 = tf.multiply(multiplier_3,
                                      input_tensor,
                                      name="TENSOR_OUTPUT")

        # build_tensor_info_op could be used if build_tensor_info is deprecated
        input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
        output_tensor_info_0 = tf.saved_model.utils.build_tensor_info(
            output_tensor_0)
        output_tensor_info_1 = tf.saved_model.utils.build_tensor_info(
            output_tensor_1)
        output_tensor_info_2 = tf.saved_model.utils.build_tensor_info(
            output_tensor_2)
        output_tensor_info_3 = tf.saved_model.utils.build_tensor_info(
            output_tensor_3)

        # Using predict method name because simple save uses it
        # tag:"serve", signature_def:"serving_default"
        signature_0 = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"INPUT": input_tensor_info},
            outputs={"OUTPUT": output_tensor_info_0},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # tag:"serve", signature_def:signature_def_name
        signature_1 = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"INPUT": input_tensor_info},
            outputs={"OUTPUT": output_tensor_info_1},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # tag:tag_name, signature_def:"serving_default"
        signature_2 = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"INPUT": input_tensor_info},
            outputs={"OUTPUT": output_tensor_info_2},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # tag:tag_name, signature_def:signature_def_name
        signature_3 = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"INPUT": input_tensor_info},
            outputs={"OUTPUT": output_tensor_info_3},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        signature_def_map_0 = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_0,
            signature_def_name: signature_1
        }
        signature_def_map_1 = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_2,
            signature_def_name: signature_3
        }

        b = builder.SavedModelBuilder(model_version_dir + "/model.savedmodel")
        b.add_meta_graph_and_variables(sess,
                                       tags=[tag_constants.SERVING],
                                       signature_def_map=signature_def_map_0,
                                       assets_collection=ops.get_collection(
                                           ops.GraphKeys.ASSET_FILEPATHS),
                                       clear_devices=True)
        b.add_meta_graph(tags=[tag_name],
                         signature_def_map=signature_def_map_1,
                         assets_collection=ops.get_collection(
                             ops.GraphKeys.ASSET_FILEPATHS),
                         clear_devices=True)
        b.save()


def create_savedmodel_modelconfig(models_dir,
                                  model_version=1,
                                  dims=16,
                                  model_name="sig_tag",
                                  tag_name="testTag",
                                  signature_def_name="testSigDef"):
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_savedmodel"
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
parameters: {{
key: "TF_GRAPH_TAG"
value: {{
string_value: "{}"
}}
}}
parameters: {{
key: "TF_SIGNATURE_DEF"
value: {{
string_value: "{}"
}}
}}
'''.format(model_name, gu.np_to_model_dtype(tf.float32), str(dims),
           gu.np_to_model_dtype(tf.float32), str(dims), tag_name,
           signature_def_name)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='getting model output dir')
    parser.add_argument('--dir', help='directory to run model in')
    args = parser.parse_args()
    base_dir = args.dir
    base_model_name = "sig_tag"
    base_tag = "serve"
    test_tag = "testTag"
    base_sig_def = "serving_default"
    test_sig_def = "testSigDef"

    for i in range(4):
        model_name = base_model_name + str(i)
        create_savedmodel(args.dir,
                          model_name=model_name,
                          tag_name=test_tag,
                          signature_def_name=test_sig_def)
    create_savedmodel_modelconfig(args.dir,
                                  model_name="sig_tag0",
                                  tag_name=base_tag,
                                  signature_def_name=base_sig_def)
    create_savedmodel_modelconfig(args.dir,
                                  model_name="sig_tag1",
                                  tag_name=base_tag,
                                  signature_def_name=test_sig_def)
    create_savedmodel_modelconfig(args.dir,
                                  model_name="sig_tag2",
                                  tag_name=test_tag,
                                  signature_def_name=base_sig_def)
    create_savedmodel_modelconfig(args.dir,
                                  model_name="sig_tag3",
                                  tag_name=test_tag,
                                  signature_def_name=test_sig_def)
