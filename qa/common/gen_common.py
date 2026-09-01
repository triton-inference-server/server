# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List

# Common utilities for model generation scripts
import ml_dtypes
import numpy as np
import test_util as tu

np_dtype_string = np.dtype(object)

# Numpy does not support the BF16 datatype natively.
# We use ml_dtypes.bfloat16 for BF16.
np_dtype_bfloat16 = ml_dtypes.bfloat16


def np_to_onnx_dtype(np_dtype):
    import onnx

    if np_dtype == bool:
        return onnx.TensorProto.BOOL
    elif np_dtype == np.int8:
        return onnx.TensorProto.INT8
    elif np_dtype == np.int16:
        return onnx.TensorProto.INT16
    elif np_dtype == np.int32:
        return onnx.TensorProto.INT32
    elif np_dtype == np.int64:
        return onnx.TensorProto.INT64
    elif np_dtype == np.uint8:
        return onnx.TensorProto.UINT8
    elif np_dtype == np.uint16:
        return onnx.TensorProto.UINT16
    elif np_dtype == np.float16:
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.float32:
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.float64:
        return onnx.TensorProto.DOUBLE
    elif np_dtype == np_dtype_string:
        return onnx.TensorProto.STRING
    elif np_dtype == np_dtype_bfloat16:
        return onnx.TensorProto.BFLOAT16
    return None


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
    elif np_dtype == np_dtype_bfloat16:
        return "TYPE_BF16"
    return None


def np_to_trt_dtype(np_dtype):
    import tensorrt as trt

    if np_dtype == bool:
        return trt.bool
    elif np_dtype == np.int8:
        return trt.int8
    elif np_dtype == np.int32:
        return trt.int32
    elif np_dtype == np.int64:
        return trt.int64
    elif np_dtype == np.uint8:
        return trt.uint8
    elif np_dtype == np.float16:
        return trt.float16
    elif np_dtype == np.float32:
        return trt.float32
    elif np_dtype == np_dtype_bfloat16:
        return trt.bfloat16
    return None


def np_to_torch_dtype(np_dtype):
    import torch

    if np_dtype == bool:
        return torch.bool
    elif np_dtype == np.int8:
        return torch.int8
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int
    elif np_dtype == np.int64:
        return torch.long
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.uint16:
        return None  # Not supported in Torch
    elif np_dtype == np.float16:
        return None
    elif np_dtype == np.float32:
        return torch.float
    elif np_dtype == np.float64:
        return torch.double
    elif np_dtype == np_dtype_string:
        return List[str]
    return None


def trt_set_dynamic_range(tensor, lo, hi):
    """Set ITensor.dynamic_range on TRT versions that support it.

    Removed in TensorRT 11+ (strongly-typed networks). Silently skip on
    versions where the attribute is gone so the QA model-gen scripts stay
    compatible with both old and new TRT."""
    try:
        tensor.dynamic_range = (lo, hi)
    except AttributeError:
        pass  # ITensor.dynamic_range removed in TensorRT 11+ (strongly-typed)


def trt_cast_tensor(network, tensor, target_dtype):
    """Insert an explicit dtype cast that works on both TRT 8.5+ (add_cast)
    and older TRT (add_identity + set_output_type). Returns the cast layer."""
    if hasattr(network, "add_cast"):
        return network.add_cast(tensor, target_dtype)
    layer = network.add_identity(tensor)
    layer.set_output_type(0, target_dtype)
    return layer


def openvino_save_model(model_version_dir, model):
    import openvino as ov

    # W/A for error moving to OpenVINO new APIs "Attempt to get a name for a Tensor without names".
    # For more details, check https://github.com/triton-inference-server/openvino_backend/issues/89
    if len(model.outputs) == 0:
        model.outputs[0].get_tensor().set_names({"OUTPUT"})
    else:
        for idx, out in enumerate(model.outputs):
            out.get_tensor().set_names({f"OUTPUT{idx}"})

    os.makedirs(model_version_dir, exist_ok=True)
    ov.serialize(
        model, model_version_dir + "/model.xml", model_version_dir + "/model.bin"
    )


# Moved here from gen_ensemble_model_utils: five generators build a plain
# model config with it and only one of them has anything to do with
# ensembles, so reaching through `emu` for it described the file it happened
# to live in rather than what it does.


def dtype_str(dtype):
    return dtype if isinstance(dtype, str) else np_to_model_dtype(dtype)


def reshape_str(shape, model_shape):
    if model_shape is None or shape == model_shape:
        return ""
    return "reshape: {{ shape: [ {} ] }}".format(tu.shape_to_dims_str(model_shape))


def label_str(label):
    if label is None:
        return ""
    return 'label_filename: "{}"'.format(label)


def create_general_modelconfig(
    model_name,
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
    force_tensor_number_suffix=False,
):
    assert len(input_dtypes) == len(input_shapes)
    assert len(input_model_shapes) == len(input_shapes)
    assert len(output_dtypes) == len(output_shapes)
    assert len(output_model_shapes) == len(output_shapes)
    assert len(label_filenames) == len(output_shapes)

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == "latest":
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(val)
        elif type == "specific":
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    default_model_filename_str = ""
    if default_model_filename is not None:
        default_model_filename_str = 'default_model_filename: "{}"'.format(
            default_model_filename
        )

    # If backend is specified use backend instead of platform
    if backend is not None:
        key = "backend"
        val = backend
    else:
        key = "platform"
        val = platform

    config = """
name: "{}"
{}: "{}"
max_batch_size: {}
version_policy: {}
{}
{}
""".format(
        model_name,
        key,
        val,
        max_batch,
        version_policy_str,
        default_model_filename_str,
        instance_group_str,
    )

    for idx in range(len(input_dtypes)):
        idx_str = ""
        if len(input_dtypes) != 1 or force_tensor_number_suffix:
            idx_str = str(idx)
        config += """
input [
  {{
    name: "INPUT{}"
    data_type: {}
    dims: [ {} ]
    {}
  }}
]""".format(
            idx_str,
            dtype_str(input_dtypes[idx]),
            tu.shape_to_dims_str(input_shapes[idx]),
            reshape_str(input_shapes[idx], input_model_shapes[idx]),
        )

    for idx in range(len(output_dtypes)):
        idx_str = ""
        if len(input_dtypes) != 1 or force_tensor_number_suffix:
            idx_str = str(idx)
        config += """
output [
  {{
    name: "OUTPUT{}"
    data_type: {}
    dims: [ {} ]
    {}
    {}
  }}
]""".format(
            idx_str,
            dtype_str(output_dtypes[idx]),
            tu.shape_to_dims_str(output_shapes[idx]),
            reshape_str(output_shapes[idx], output_model_shapes[idx]),
            label_str(label_filenames[idx]),
        )
    return config
