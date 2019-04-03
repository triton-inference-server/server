# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
import test_util as tu
import numpy as np

BASIC_ENSEMBLE_TYPES = ["simple", "sequence", "fan"]

np_dtype_string = np.dtype(object)

def np_to_model_dtype(np_dtype):
    if np_dtype == np.bool:
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

def fixed_to_variable_size(shape):
    return [-1] * len(shape)

def platform_types_and_validation():
    res = [
        ("graphdef", tu.validate_for_tf_model),
        ("savedmodel", tu.validate_for_tf_model),
        ("netdef", tu.validate_for_c2_model),
        ("plan", tu.validate_for_trt_model)]
    return res

class EnsembleSchedule:
    """Helper class to generate ensemble schedule given an ensemble type"""
    def __init__(self, ensemble_type):
        if ensemble_type == "fan":
            self._get_steps = EnsembleSchedule._get_fan_ensemble_steps
        elif ensemble_type == "sequence":
            self._get_steps = EnsembleSchedule._get_sequence_ensemble_steps
        else:
            self._get_steps = EnsembleSchedule._get_simple_ensemble_steps

    def get_schedule(self, base_model_name,
            input_dim_len, output0_dim_len, output1_dim_len,
            input_model_dtype, output0_model_dtype, output1_model_dtype):
        return self._get_steps(base_model_name,
            input_dim_len, output0_dim_len, output1_dim_len,
            input_model_dtype, output0_model_dtype, output1_model_dtype)

    @classmethod
    def _get_simple_ensemble_steps(cls, base_model_name,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype):
        # ensemble input -> addsub -> ensemble output
        steps = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT0"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT1"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT0"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "OUTPUT1"
      }}
    }}
  ]
}}
'''.format(base_model_name)
        return steps

    @classmethod
    def _get_sequence_ensemble_steps(cls, base_model_name,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype):
        # ensemble input -> nop -> addsub -> ensemble output
        nop_input_shape = fixed_to_variable_size(input_shape)
        steps = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT0"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT1"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "same_input0"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "same_input1"
      }}
    }},
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "same_input0"
      }}
      input_map {{
        key: "INPUT1"
        value: "same_input1"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT0"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "OUTPUT1"
      }}
    }}
  ]
}}
'''.format(input_dtype, tu.shape_to_dims_str(nop_input_shape), base_model_name)
        return steps

    @classmethod
    def _get_fan_ensemble_steps(cls, base_model_name,
            input_shape, output0_shape, output1_shape,
            input_dtype, output0_dtype, output1_dtype):
        # ensemble input -> nop -> addsub ->
        # nop (fan out, one output send to one nop) -> ensemble output (fan in)
        nop_input_shape = fixed_to_variable_size(input_shape)
        nop_output0_shape = fixed_to_variable_size(output0_shape)
        nop_output1_shape = fixed_to_variable_size(output1_shape)
        steps = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT0"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT1"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "same_input0"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "same_input1"
      }}
    }},
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "same_input0"
      }}
      input_map {{
        key: "INPUT1"
        value: "same_input1"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "same_output0"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "same_output1"
      }}
    }},
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "same_output0"
      }}
      input_map {{
        key: "INPUT1"
        value: "same_output0"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT0"
      }}
    }},
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "same_output1"
      }}
      input_map {{
        key: "INPUT1"
        value: "same_output1"
      }}
      output_map {{
        key: "OUTPUT1"
        value: "OUTPUT1"
      }}
    }}
  ]
}}
'''.format(input_dtype, tu.shape_to_dims_str(nop_input_shape), base_model_name,
              output0_dtype, tu.shape_to_dims_str(nop_output0_shape),
              output1_dtype, tu.shape_to_dims_str(nop_output1_shape))
        return steps

def create_ensemble_modelfile(
        base_model, models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype, swap=False):

    # No actual model file in ensemble model

    # Use a different model name for the non-batching variant
    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        ensemble_model_name = "{}_{}{}".format(ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_model_name(ensemble_model_name,
                                      input_dtype, output0_dtype, output1_dtype)
        model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

        try:
            os.makedirs(model_version_dir)
        except OSError as ex:
            pass # ignore existing dir

def create_ensemble_modelconfig(
        base_model, models_dir, max_batch, model_version,
        input_shape, output0_shape, output1_shape,
        input_dtype, output0_dtype, output1_dtype,
        output0_label_cnt, version_policy):

    # No validation as long as the base model supports the type and shape

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    input_model_dtype = np_to_model_dtype(input_dtype)
    output0_model_dtype = np_to_model_dtype(output0_dtype)
    output1_model_dtype = np_to_model_dtype(output1_dtype)

    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        # Use a different model name for the non-batching variant
        ensemble_model_name = "{}_{}{}".format(ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_model_name(ensemble_model_name,
                                    input_dtype, output0_dtype, output1_dtype)
        base_model_name = tu.get_model_name("{}{}".format(base_model, "_nobatch" if max_batch == 0 else ""),
                                    input_dtype, output0_dtype, output1_dtype)

        ensemble_schedule = EnsembleSchedule(ensemble_type).get_schedule(
                        base_model_name, input_shape, output0_shape,
                        output1_shape, input_model_dtype,
                        output0_model_dtype, output1_model_dtype)

        config_dir = models_dir + "/" + model_name
        config = '''
name: "{}"
platform: "ensemble"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
  }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
{}
'''.format(model_name, max_batch, version_policy_str,
            input_model_dtype, tu.shape_to_dims_str(input_shape),
            input_model_dtype, tu.shape_to_dims_str(input_shape),
            output0_model_dtype, tu.shape_to_dims_str(output0_shape),
            output1_model_dtype, tu.shape_to_dims_str(output1_shape),
            ensemble_schedule)

        try:
            os.makedirs(config_dir)
        except OSError as ex:
            pass # ignore existing dir

        with open(config_dir + "/config.pbtxt", "w") as cfile:
            cfile.write(config)

        with open(config_dir + "/output0_labels.txt", "w") as lfile:
            for l in range(output0_label_cnt):
                lfile.write("label" + str(l) + "\n")


def create_nop_modelconfig(models_dir, tensor_shape, tensor_dtype):
    model_name = "nop_{}_{}".format(tensor_dtype, tu.shape_to_dims_str(tensor_shape))
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{model_name}"
platform: "custom"
max_batch_size: {batch_size}
default_model_filename: "libidentity.so"
input [
  {{
    name: "INPUT0"
    data_type: {dtype}
    dims: [ {dim} ]
  }},
  {{
    name: "INPUT1"
    data_type: {dtype}
    dims: [ {dim} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {dtype}
    dims: [ {dim} ]
  }},
  {{
    name: "OUTPUT1"
    data_type: {dtype}
    dims: [ {dim} ]
  }}
]
instance_group [ {{ kind: KIND_CPU }} ]
'''.format(model_name=model_name, dtype=tensor_dtype,
            batch_size=1024, dim=tu.shape_to_dims_str(tensor_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

