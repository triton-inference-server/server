# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorflow as tf
from tensorflow.python.framework import graph_io


def create_graphdefmodel(models_dir, model_name, model_version=1):
    """A simple tensorflow model that accumulates the INPUT with internal
    model parameter named VARIABLE and produces OUTPUT.
    """

    tf.reset_default_graph()
    input0 = tf.placeholder(tf.int32, [
        1,
    ], "INPUT")
    variable = tf.get_variable("VARIABLE", [
        1,
    ],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.int32)
    tf.add(variable, input0, name="OUTPUT")
    tf.global_variables_initializer()
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        graph_io.write_graph(sess.graph.as_graph_def(),
                             model_version_dir,
                             "model.graphdef",
                             as_text=False)


def create_graphdef_modelconfig(models_dir, model_name):
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_graphdef"
input [
  {{
    name: "INPUT"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]
'''.format(model_name)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    args = parser.parse_args()

    model_name = "graphdef_variable"
    create_graphdefmodel(args.models_dir, model_name)
    create_graphdef_modelconfig(args.models_dir, model_name=model_name)
