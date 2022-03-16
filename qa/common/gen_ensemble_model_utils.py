# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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


def fixed_to_variable_size(shape):
    return [-1] * len(shape)


def platform_types_and_validation():
    res = [("graphdef", tu.validate_for_tf_model),
           ("savedmodel", tu.validate_for_tf_model),
           ("plan", tu.validate_for_trt_model),
           ("onnx", tu.validate_for_onnx_model),
           ("libtorch", tu.validate_for_libtorch_model)]
    return res


class AddSubEnsembleSchedule:
    """
    Helper class to generate ensemble schedule that behaves the same as
    addsub model given an ensemble type
    """

    def __init__(self, ensemble_type):
        if ensemble_type == "fan":
            self._get_schedule = AddSubEnsembleSchedule._get_fan_ensemble_schedule
        elif ensemble_type == "sequence":
            self._get_schedule = AddSubEnsembleSchedule._get_sequence_ensemble_schedule
        else:
            self._get_schedule = AddSubEnsembleSchedule._get_simple_ensemble_schedule

    def get_schedule(self, base_model_name, input_shape, output0_shape,
                     output1_shape, input_model_dtype, output0_model_dtype,
                     output1_model_dtype):
        return self._get_schedule(base_model_name, input_shape, output0_shape,
                                  output1_shape, input_model_dtype,
                                  output0_model_dtype, output1_model_dtype)

    @classmethod
    def _get_simple_ensemble_schedule(cls, base_model_name, input_shape,
                                      output0_shape, output1_shape, input_dtype,
                                      output0_dtype, output1_dtype):
        # libtorch model uses other naming convention for outputs
        output_index_delimiter = "__" if "libtorch" in base_model_name else ""
        # ensemble input -> addsub -> ensemble output
        schedule = '''
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
        key: "OUTPUT{delimiter}0"
        value: "OUTPUT0"
      }}
      output_map {{
        key: "OUTPUT{delimiter}1"
        value: "OUTPUT1"
      }}
    }}
  ]
}}
'''.format(base_model_name, delimiter=output_index_delimiter)
        return schedule

    @classmethod
    def _get_sequence_ensemble_schedule(cls, base_model_name, input_shape,
                                        output0_shape, output1_shape,
                                        input_dtype, output0_dtype,
                                        output1_dtype):
        # libtorch model uses other naming convention for outputs
        output_index_delimiter = "__" if "libtorch" in base_model_name else ""
        # ensemble input -> nop -> addsub -> ensemble output
        nop_input_shape = fixed_to_variable_size(input_shape)
        schedule = '''
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
        key: "OUTPUT{delimiter}0"
        value: "OUTPUT0"
      }}
      output_map {{
        key: "OUTPUT{delimiter}1"
        value: "OUTPUT1"
      }}
    }}
  ]
}}
'''.format(input_dtype,
           tu.shape_to_dims_str(nop_input_shape),
           base_model_name,
           delimiter=output_index_delimiter)
        return schedule

    @classmethod
    def _get_fan_ensemble_schedule(cls, base_model_name, input_shape,
                                   output0_shape, output1_shape, input_dtype,
                                   output0_dtype, output1_dtype):
        # libtorch model uses other naming convention for outputs
        output_index_delimiter = "__" if "libtorch" in base_model_name else ""

        # ensemble input -> nop -> addsub ->
        # nop (fan out, one output send to one nop) -> ensemble output (fan in)
        nop_input_shape = fixed_to_variable_size(input_shape)
        nop_output0_shape = fixed_to_variable_size(output0_shape)
        nop_output1_shape = fixed_to_variable_size(output1_shape)
        schedule = '''
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
        key: "OUTPUT{delimiter}0"
        value: "same_output0"
      }}
      output_map {{
        key: "OUTPUT{delimiter}1"
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
'''.format(input_dtype,
           tu.shape_to_dims_str(nop_input_shape),
           base_model_name,
           output0_dtype,
           tu.shape_to_dims_str(nop_output0_shape),
           output1_dtype,
           tu.shape_to_dims_str(nop_output1_shape),
           delimiter=output_index_delimiter)
        return schedule


class IdentityEnsembleSchedule:
    """
    Helper class to generate ensemble schedule that behaves the same as
    identity model given an ensemble type
    """

    def __init__(self, ensemble_type, ensemble_test_type="zero"):
        self._test_type = ensemble_test_type
        if ensemble_type == "fan":
            self._get_schedule = IdentityEnsembleSchedule._get_fan_ensemble_schedule
        elif ensemble_type == "sequence":
            self._get_schedule = IdentityEnsembleSchedule._get_sequence_ensemble_schedule
        else:
            self._get_schedule = IdentityEnsembleSchedule._get_simple_ensemble_schedule

    def get_schedule(self, dtype, input_shapes, input_model_shapes,
                     output_shapes, output_model_shapes):
        return self._get_schedule(dtype, input_shapes, input_model_shapes,
                                  output_shapes, output_model_shapes,
                                  self._test_type)

    @classmethod
    def _get_simple_ensemble_schedule(cls, dtype, input_shapes,
                                      input_model_shapes, output_shapes,
                                      output_model_shapes, test_type):
        # ensemble reshaped input -> nop with reshaped tensor shape -> ensemble
        # reshaped output (actual ensemble input/output is not visible in schedule)
        steps = []
        for idx in range(len(input_shapes)):
            steps.append('''
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT{}"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT{}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT{}"
      }}
    }}
'''.format(np_to_model_dtype(dtype),
            tu.shape_to_dims_str(input_model_shapes[idx]), idx, idx, idx))

        schedule = '''
ensemble_scheduling {{
  step [
{}
  ]
}}
'''.format(",".join(steps))

        return schedule

    @classmethod
    def _get_sequence_ensemble_schedule(cls, dtype, input_shapes,
                                        input_model_shapes, output_shapes,
                                        output_model_shapes, test_type):
        in_str = "tunnel_in_" if test_type == "reshape" else ""
        out_str = "tunnel_out_" if test_type == "reshape" else ""
        # ensemble reshaped input -> nop with another input only reshape ->
        # nop with output only reshape -> ensemble reshaped output
        steps = []
        for idx in range(len(input_shapes)):
            steps.append('''
    {{
      model_name: "nop_{in_str}{type}_{shape}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT{idx}"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT{idx}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "temp_{idx}"
      }}
    }},
    {{
      model_name: "nop_{out_str}{type}_{shape}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "temp_{idx}"
      }}
      input_map {{
        key: "INPUT1"
        value: "temp_{idx}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT{idx}"
      }}
    }}
'''.format(type=np_to_model_dtype(dtype),
            in_str=in_str,
            out_str=out_str,
            idx=idx,
            shape=tu.shape_to_dims_str(input_model_shapes[idx])))

        schedule = '''
ensemble_scheduling {{
  step [
{}
  ]
}}
'''.format(",".join(steps))

        return schedule

    @classmethod
    def _get_fan_ensemble_schedule(cls, dtype, input_shapes, input_model_shapes,
                                   output_shapes, output_model_shapes,
                                   test_type):
        # Note that the simple and sequence test already test "fan" in some
        # degree, because there is no direct match from nop input/output
        # like what is in addsub-like ensemble.
        #
        # ensemble reshaped input -> nop with another input only reshape ->
        # nop with variable size -> nop with output only reshape ->
        # ensemble reshaped output
        in_str = ""
        out_str = ""
        intermediate_shapes = input_model_shapes
        if test_type == "reshape":
            in_str = "tunnel_in_"
            out_str = "tunnel_out_"
            intermediate_shapes = [[-1]] * len(input_model_shapes)
        steps = []
        for idx in range(len(input_shapes)):
            steps.append('''
    {{
      model_name: "nop_{in_str}{type}_{shape}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT{idx}"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT{idx}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "temp_in_{idx}"
      }}
    }},
    {{
      model_name: "nop_{type}_{intermediate_shape}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "temp_in_{idx}"
      }}
      input_map {{
        key: "INPUT1"
        value: "temp_in_{idx}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "temp_out_{idx}"
      }}
    }},
    {{
      model_name: "nop_{out_str}{type}_{shape}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "temp_out_{idx}"
      }}
      input_map {{
        key: "INPUT1"
        value: "temp_out_{idx}"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT{idx}"
      }}
    }}
'''.format(type=np_to_model_dtype(dtype),
            in_str=in_str,
            out_str=out_str,
            intermediate_shape=tu.shape_to_dims_str(intermediate_shapes[idx]),
            idx=idx,
            shape=tu.shape_to_dims_str(input_model_shapes[idx])))

        schedule = '''
ensemble_scheduling {{
  step [
{}
  ]
}}
'''.format(",".join(steps))

        return schedule


class SequenceEnsembleSchedule:
    """
    Helper class to generate ensemble schedule that behaves the same as
    sequence model given an ensemble type
    """

    def __init__(self, ensemble_type):
        if ensemble_type == "fan":
            self._get_schedule = SequenceEnsembleSchedule._get_fan_ensemble_schedule
        elif ensemble_type == "sequence":
            self._get_schedule = SequenceEnsembleSchedule._get_sequence_ensemble_schedule
        else:
            self._get_schedule = SequenceEnsembleSchedule._get_simple_ensemble_schedule

    def get_schedule(self, base_model_name, shape, model_dtype):
        return self._get_schedule(base_model_name, shape, model_dtype)

    @classmethod
    def _get_simple_ensemble_schedule(cls, base_model_name, shape, model_dtype):
        # libtorch model uses other naming convention
        index_suffix = "__0" if "libtorch" in base_model_name else ""
        # ensemble input -> sequence -> ensemble output
        schedule = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT{index}"
        value: "INPUT"
      }}
      output_map {{
        key: "OUTPUT{index}"
        value: "OUTPUT"
      }}
    }}
  ]
}}
'''.format(base_model_name, index=index_suffix)
        return schedule

    @classmethod
    def _get_sequence_ensemble_schedule(cls, base_model_name, shape,
                                        model_dtype):
        # nop cannot handle STRING data type, fall back to simple
        if model_dtype == "TYPE_STRING":
            return SequenceEnsembleSchedule._get_simple_ensemble_schedule(
                base_model_name, shape, model_dtype)

        # libtorch model uses other naming convention
        index_suffix = "__0" if "libtorch" in base_model_name else ""
        # ensemble input -> nop -> sequence -> ensemble output
        nop_input_shape = fixed_to_variable_size(shape)
        schedule = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "same_input"
      }}
    }},
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT{index}"
        value: "same_input"
      }}
      output_map {{
        key: "OUTPUT{index}"
        value: "OUTPUT"
      }}
    }}
  ]
}}
'''.format(model_dtype,
           tu.shape_to_dims_str(nop_input_shape),
           base_model_name,
           index=index_suffix)
        return schedule

    @classmethod
    def _get_fan_ensemble_schedule(cls, base_model_name, shape, model_dtype):
        # nop cannot handle STRING data type, fall back to simple
        if model_dtype == "TYPE_STRING":
            return SequenceEnsembleSchedule._get_simple_ensemble_schedule(
                base_model_name, shape, model_dtype)

        # libtorch model uses other naming convention
        index_suffix = "__0" if "libtorch" in base_model_name else ""
        # Not a "fan" due to configuration of base sequence model
        # ensemble input -> nop -> sequence -> nop -> ensemble output
        nop_shape = fixed_to_variable_size(shape)
        schedule = '''
ensemble_scheduling {{
  step [
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "INPUT"
      }}
      input_map {{
        key: "INPUT1"
        value: "INPUT"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "same_input"
      }}
    }},
    {{
      model_name: "{}"
      model_version: -1
      input_map {{
        key: "INPUT{index}"
        value: "same_input"
      }}
      output_map {{
        key: "OUTPUT{index}"
        value: "same_output"
      }}
    }},
    {{
      model_name: "nop_{}_{}"
      model_version: -1
      input_map {{
        key: "INPUT0"
        value: "same_output"
      }}
      input_map {{
        key: "INPUT1"
        value: "same_output"
      }}
      output_map {{
        key: "OUTPUT0"
        value: "OUTPUT"
      }}
    }}
  ]
}}
'''.format(model_dtype,
           tu.shape_to_dims_str(nop_shape),
           base_model_name,
           model_dtype,
           tu.shape_to_dims_str(nop_shape),
           index=index_suffix)
        return schedule


def create_ensemble_modelfile(base_model,
                              models_dir,
                              max_batch,
                              model_version,
                              input_shape,
                              output0_shape,
                              output1_shape,
                              input_dtype,
                              output0_dtype,
                              output1_dtype,
                              swap=False):

    # No actual model file in ensemble model

    # Use a different model name for the non-batching variant
    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        ensemble_model_name = "{}_{}{}".format(
            ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_model_name(ensemble_model_name, input_dtype,
                                       output0_dtype, output1_dtype)
        model_version_dir = models_dir + "/" + model_name + "/" + str(
            model_version)

        try:
            os.makedirs(model_version_dir)
        except OSError as ex:
            pass  # ignore existing dir


def create_ensemble_modelconfig(base_model, models_dir, max_batch,
                                model_version, input_shape, output0_shape,
                                output1_shape, input_dtype, output0_dtype,
                                output1_dtype, output0_label_cnt,
                                version_policy):

    # No validation as long as the base model supports the type and shape

    input_model_dtype = np_to_model_dtype(input_dtype)
    output0_model_dtype = np_to_model_dtype(output0_dtype)
    output1_model_dtype = np_to_model_dtype(output1_dtype)

    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        # Only in "fan" that ensemble output is not directly from addsub. In
        # other case, ensemble should still generate proper labell without
        # label file
        labels = None if ensemble_type != "fan" else "output0_labels.txt"

        # Use a different model name for the non-batching variant
        ensemble_model_name = "{}_{}{}".format(
            ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_model_name(ensemble_model_name, input_dtype,
                                       output0_dtype, output1_dtype)
        base_model_name = tu.get_model_name(
            "{}{}".format(base_model, "_nobatch" if max_batch == 0 else ""),
            input_dtype, output0_dtype, output1_dtype)

        ensemble_schedule = AddSubEnsembleSchedule(ensemble_type).get_schedule(
            base_model_name, input_shape, output0_shape, output1_shape,
            input_model_dtype, output0_model_dtype, output1_model_dtype)

        config_dir = models_dir + "/" + model_name
        config = create_general_modelconfig(model_name,
                                            "ensemble",
                                            max_batch,
                                            repeat(input_dtype, 2),
                                            repeat(input_shape, 2),
                                            repeat(None, 2),
                                            [output0_dtype, output1_dtype],
                                            [output0_shape, output1_shape],
                                            repeat(None, 2), [labels, None],
                                            version_policy=version_policy,
                                            force_tensor_number_suffix=True)
        config += ensemble_schedule

        try:
            os.makedirs(config_dir)
        except OSError as ex:
            pass  # ignore existing dir

        with open(config_dir + "/config.pbtxt", "w") as cfile:
            cfile.write(config)

        if labels is not None:
            with open(config_dir + "/output0_labels.txt", "w") as lfile:
                for l in range(output0_label_cnt):
                    lfile.write("label" + str(l) + "\n")


def create_identity_ensemble_modelfile(ensemble_test_type, models_dir,
                                       model_version, max_batch, dtype,
                                       input_shapes, output_shapes):
    io_cnt = len(input_shapes)

    # Use a different model name for the non-batching variant
    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        ensemble_prefix = "{}_{}".format(ensemble_type, ensemble_test_type)
        model_name = tu.get_zero_model_name(
            ensemble_prefix + ("_nobatch" if max_batch == 0 else ""), io_cnt,
            dtype)
        model_version_dir = models_dir + "/" + model_name + "/" + str(
            model_version)

        try:
            os.makedirs(model_version_dir)
        except OSError as ex:
            pass  # ignore existing dir


def create_identity_ensemble_modelconfig(ensemble_test_type, models_dir,
                                         model_version, max_batch, dtype,
                                         input_shapes, input_model_shapes,
                                         output_shapes, output_model_shapes):
    io_cnt = len(input_shapes)

    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        # Use a different model name for the non-batching variant
        ensemble_prefix = "{}_{}".format(ensemble_type, ensemble_test_type)
        model_name = tu.get_zero_model_name(
            ensemble_prefix + ("_nobatch" if max_batch == 0 else ""), io_cnt,
            dtype)

        ensemble_schedule = IdentityEnsembleSchedule(
            ensemble_type,
            ensemble_test_type).get_schedule(dtype, input_shapes,
                                             input_model_shapes, output_shapes,
                                             output_model_shapes)

        config_dir = models_dir + "/" + model_name
        config = create_general_modelconfig(model_name,
                                            "ensemble",
                                            max_batch,
                                            repeat(dtype, io_cnt),
                                            input_shapes,
                                            input_model_shapes,
                                            repeat(dtype, io_cnt),
                                            output_shapes,
                                            output_model_shapes,
                                            repeat(None, io_cnt),
                                            force_tensor_number_suffix=True)
        config += ensemble_schedule

        try:
            os.makedirs(config_dir)
        except OSError as ex:
            pass  # ignore existing dir

        with open(config_dir + "/config.pbtxt", "w") as cfile:
            cfile.write(config)


def create_sequence_ensemble_modelfile(base_model, models_dir, max_batch,
                                       model_version, shape, dtype):

    # No actual model file in ensemble model

    # Use a different model name for the non-batching variant
    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        ensemble_model_name = "{}_{}{}".format(
            ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_sequence_model_name(ensemble_model_name, dtype)
        model_version_dir = models_dir + "/" + model_name + "/" + str(
            model_version)

        try:
            os.makedirs(model_version_dir)
        except OSError as ex:
            pass  # ignore existing dir


def create_sequence_ensemble_modelconfig(base_model, models_dir, max_batch,
                                         model_version, shape, dtype):

    # No validation as long as the base model supports the type and shape

    model_dtype = np_to_model_dtype(dtype)

    for ensemble_type in BASIC_ENSEMBLE_TYPES:
        # Use a different model name for the non-batching variant
        ensemble_model_name = "{}_{}{}".format(
            ensemble_type, base_model, "_nobatch" if max_batch == 0 else "")
        model_name = tu.get_sequence_model_name(ensemble_model_name, dtype)
        base_model_name = tu.get_sequence_model_name(
            "{}{}".format(base_model, "_nobatch" if max_batch == 0 else ""),
            dtype)

        ensemble_schedule = SequenceEnsembleSchedule(
            ensemble_type).get_schedule(base_model_name, shape, model_dtype)

        config_dir = models_dir + "/" + model_name
        config = create_general_modelconfig(model_name, "ensemble", max_batch,
                                            [dtype], [shape], [None], [dtype],
                                            [shape], [None], [None])
        config += ensemble_schedule

        try:
            os.makedirs(config_dir)
        except OSError as ex:
            pass  # ignore existing dir

        with open(config_dir + "/config.pbtxt", "w") as cfile:
            cfile.write(config)


def create_nop_modelconfig(models_dir,
                           tensor_shape,
                           tensor_dtype,
                           tensor_model_shape=None):
    model_name = "nop_{}_{}".format(dtype_str(tensor_dtype),
                                    tu.shape_to_dims_str(tensor_shape))
    # Make [] to [1].
    # Note that this doesn't affect the naming ("nop_{}_" instead of "nop_{}_1")
    if len(tensor_shape) == 0:
        tensor_shape = [1]

    config_dir = models_dir + "/" + model_name
    config = create_general_modelconfig(
        model_name,
        "",
        1024,
        repeat(tensor_dtype, 2),
        repeat(tensor_shape, 2),
        repeat(tensor_model_shape, 2),
        repeat(tensor_dtype, 2),
        repeat(tensor_shape, 2),
        repeat(tensor_model_shape, 2),
        repeat(None, 2),
        backend="identity",
        instance_group_str="instance_group [ { kind: KIND_CPU } ]")

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_nop_tunnel_modelconfig(models_dir, tensor_shape, tensor_dtype):
    # Must be fixed size
    in_model_name = "nop_tunnel_in_{}_{}".format(
        dtype_str(tensor_dtype), tu.shape_to_dims_str(tensor_shape))
    out_model_name = "nop_tunnel_out_{}_{}".format(
        dtype_str(tensor_dtype), tu.shape_to_dims_str(tensor_shape))
    # Make [] to [1].
    # Note that this doesn't affect the naming ("nop_{}_" instead of "nop_{}_1")
    if len(tensor_shape) == 0:
        tensor_shape = [1]
    internal_shape = 1
    for dim in tensor_shape:
        if dim < 0:
            raise Exception(
                "Must specify fixed size input / output for nop tunnel")
        internal_shape *= dim

    # Tunnel in nop (reshape to one dimension)
    config_dir = models_dir + "/" + in_model_name
    config = create_general_modelconfig(
        in_model_name,
        "",
        1024,
        repeat(tensor_dtype, 2),
        repeat(tensor_shape, 2),
        repeat([internal_shape], 2),
        repeat(tensor_dtype, 2),
        repeat([internal_shape], 2),
        repeat(None, 2),
        repeat(None, 2),
        backend="identity",
        instance_group_str="instance_group [ { kind: KIND_CPU } ]")

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    # Tunnel out nop (reshape back to original shape)
    config_dir = models_dir + "/" + out_model_name
    config = create_general_modelconfig(
        out_model_name,
        "",
        1024,
        repeat(tensor_dtype, 2),
        repeat([internal_shape], 2),
        repeat(tensor_shape, 2),
        repeat(tensor_dtype, 2),
        repeat(tensor_shape, 2),
        repeat(None, 2),
        repeat(None, 2),
        backend="identity",
        instance_group_str="instance_group [ { kind: KIND_CPU } ]")

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_general_modelconfig(model_name,
                               platform,
                               max_batch,
                               input_dtypes,
                               input_shapes,
                               input_model_shapes,
                               output_dtypes,
                               output_shapes,
                               output_model_shapes,
                               label_filenames,
                               backend=None,
                               version_policy=None,
                               default_model_filename=None,
                               instance_group_str="",
                               force_tensor_number_suffix=False):
    assert len(input_dtypes) == len(input_shapes)
    assert len(input_model_shapes) == len(input_shapes)
    assert len(output_dtypes) == len(output_shapes)
    assert len(output_model_shapes) == len(output_shapes)
    assert len(label_filenames) == len(output_shapes)

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    default_model_filename_str = ""
    if default_model_filename is not None:
        default_model_filename_str = 'default_model_filename: "{}"'.format(
            default_model_filename)

    # If backend is specified use backend instead of platform
    if backend is not None:
        key = "backend"
        val = backend
    else:
        key = "platform"
        val = platform

    config = '''
name: "{}"
{}: "{}"
max_batch_size: {}
version_policy: {}
{}
{}
'''.format(model_name, key, val, max_batch, version_policy_str,
           default_model_filename_str, instance_group_str)

    for idx in range(len(input_dtypes)):
        idx_str = ""
        if len(input_dtypes) != 1 or force_tensor_number_suffix:
            idx_str = str(idx)
        config += '''
input [
  {{
    name: "INPUT{}"
    data_type: {}
    dims: [ {} ]
    {}
  }}
]'''.format(idx_str, dtype_str(input_dtypes[idx]),
            tu.shape_to_dims_str(input_shapes[idx]),
            reshape_str(input_shapes[idx], input_model_shapes[idx]))

    for idx in range(len(output_dtypes)):
        idx_str = ""
        if len(input_dtypes) != 1 or force_tensor_number_suffix:
            idx_str = str(idx)
        config += '''
output [
  {{
    name: "OUTPUT{}"
    data_type: {}
    dims: [ {} ]
    {}
    {}
  }}
]'''.format(idx_str, dtype_str(output_dtypes[idx]),
            tu.shape_to_dims_str(output_shapes[idx]),
            reshape_str(output_shapes[idx], output_model_shapes[idx]),
            label_str(label_filenames[idx]))
    return config


def repeat(obj, cnt):
    return [obj] * cnt


def dtype_str(dtype):
    return dtype if isinstance(dtype, str) else np_to_model_dtype(dtype)


def reshape_str(shape, model_shape):
    if model_shape is None or shape == model_shape:
        return ""
    return "reshape: {{ shape: [ {} ] }}".format(
        tu.shape_to_dims_str(model_shape))


def label_str(label):
    if label is None:
        return ""
    return 'label_filename: "{}"'.format(label)
