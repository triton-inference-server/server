#!/usr/bin/env python3

# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import unittest

import numpy as np

_last_request_id = 0


def shape_element_count(shape):
    cnt = 0
    for d in shape:
        if d == -1:
            return -1
        if cnt == 0:
            cnt = d
        else:
            cnt = cnt * d
    return cnt


def shape_is_fixed(shape):
    return shape_element_count(shape) != -1


def shape_to_tf_shape(shape):
    return [None if i == -1 else i for i in shape]


def shape_to_onnx_shape(shape, idx=0, increment_index=True):
    # Onnx use string for variable size dimension, and the same string
    # will be inferred to have same value for the model run.
    # So there is an extra "idx" parameter to make sure the string is
    # unique
    res = []
    for dim in shape:
        if dim == -1:
            res.append("var_" + str(idx))
            if increment_index:
                idx += 1
        else:
            res.append(dim)
    return res, idx


def shape_to_dims_str(shape):
    return ",".join(str(i) for i in shape)


def validate_for_tf_model(
    input_dtype, output0_dtype, output1_dtype, input_shape, output0_shape, output1_shape
):
    """Return True if input and output dtypes are supported by a TF model."""

    # Not extending test to uint8 yet
    if (
        input_dtype == np.uint8
        or output0_dtype == np.uint8
        or output1_dtype == np.uint8
    ):
        return False

    # If the input type is string the output type must be string or
    # int32. This is because the QA models we generate convert strings
    # internally to int32 for compute.
    if (input_dtype == np.object_) and (
        ((output0_dtype != np.object_) and (output0_dtype != np.int32))
        or ((output1_dtype != np.object_) and (output1_dtype != np.int32))
    ):
        return False

    return True


def validate_for_trt_model(
    input_dtype, output0_dtype, output1_dtype, input_shape, output0_shape, output1_shape
):
    """Return True if input and output dtypes are supported by a TRT model."""
    supported_datatypes = [bool, np.int8, np.int32, np.uint8, np.float16, np.float32]
    # FIXME: Remove this check when jetson supports TRT 8.5 (DLIS-4256)
    if not support_trt_uint8():
        supported_datatypes.remove(np.uint8)
    if not input_dtype in supported_datatypes:
        return False
    if not output0_dtype in supported_datatypes:
        return False
    if not output1_dtype in supported_datatypes:
        return False

    datatype_set = set([input_dtype, output0_dtype, output1_dtype])

    # Incompatible datatype conversions
    if (np.int32 in datatype_set) and (np.int8 in datatype_set):
        return False
    if (np.float32 in datatype_set) and (np.int32 in datatype_set):
        return False

    return True


def validate_for_ensemble_model(
    ensemble_type,
    input_dtype,
    output0_dtype,
    output1_dtype,
    input_shape,
    output0_shape,
    output1_shape,
):
    """Return True if input and output dtypes are supported by the ensemble type."""

    # Not extending test to uint8 yet
    if (
        input_dtype == np.uint8
        or output0_dtype == np.uint8
        or output1_dtype == np.uint8
    ):
        return False

    # Those ensemble types contains "identity" model which doesn't allow STRING
    # data type
    # Test types that use identity for both input and output
    test_type_involved = ["reshape", "zero", "fan"]
    if (
        input_dtype == np.object_
        or output0_dtype == np.object_
        or output1_dtype == np.object_
    ):
        for type_str in test_type_involved:
            if type_str in ensemble_type:
                return False

    # Otherwise, check input / output separately
    if input_dtype == np.object_ and "sequence" in ensemble_type:
        return False

    return True


def validate_for_onnx_model(
    input_dtype, output0_dtype, output1_dtype, input_shape, output0_shape, output1_shape
):
    """Return True if input and output dtypes are supported by a Onnx model."""

    # Not extending test to uint8 yet
    if (
        input_dtype == np.uint8
        or output0_dtype == np.uint8
        or output1_dtype == np.uint8
    ):
        return False

    # If the input type is string the output type must be string or
    # int32. This is because the QA models we generate convert strings
    # internally to int32 for compute.
    if (input_dtype == np.object_) and (
        ((output0_dtype != np.object_) and (output0_dtype != np.int32))
        or ((output1_dtype != np.object_) and (output1_dtype != np.int32))
    ):
        return False

    return True


def validate_for_libtorch_model(
    input_dtype,
    output0_dtype,
    output1_dtype,
    input_shape,
    output0_shape,
    output1_shape,
    max_batch=0,
    reshape=False,
):
    """Return True if input and output dtypes are supported by a libtorch model."""

    # Not extending test to uint8 yet
    if (
        input_dtype == np.uint8
        or output0_dtype == np.uint8
        or output1_dtype == np.uint8
    ):
        return False

    # STRING data type does not support I/O with more than 1 dims. It supports
    # batching when 'reshape' field is set properly to empty shape.
    has_string_type = (
        (input_dtype == np.object_)
        or (output0_dtype == np.object_)
        or (output1_dtype == np.object_)
    )
    is_more_than_one_dimensional = (
        (len(input_shape) > 1)
        or (len(output0_shape) > 1)
        or (len(output1_shape) > 1)
        or (max_batch != 0)
    )

    if has_string_type and is_more_than_one_dimensional and not reshape:
        return False

    # FLOAT16 and UINT16 data types are not supported currently
    if (
        (input_dtype == np.uint16)
        or (output0_dtype == np.uint16)
        or (output1_dtype == np.uint16)
    ):
        return False
    if (
        (input_dtype == np.float16)
        or (output0_dtype == np.float16)
        or (output1_dtype == np.float16)
    ):
        return False

    return True


def validate_for_openvino_model(
    input_dtype, output0_dtype, output1_dtype, input_shape, output0_shape, output1_shape
):
    """Return True if input and output dtypes are supported by an OpenVino model."""

    # Not extending test to uint8 yet
    if (
        input_dtype == np.uint8
        or output0_dtype == np.uint8
        or output1_dtype == np.uint8
    ):
        return False

    # float16 is not supported on CPU by OpenVino
    supported_datatypes = [np.int8, np.int32, np.float32]
    if not input_dtype in supported_datatypes:
        return False
    if not output0_dtype in supported_datatypes:
        return False
    if not output1_dtype in supported_datatypes:
        return False

    # Return false if input dtype != output dtype and shape > 1 dims
    # https://github.com/openvinotoolkit/openvino/issues/7173
    if ((output1_dtype != input_dtype) or (output0_dtype != input_dtype)) and len(
        input_shape
    ) > 1:
        return False

    return True


def get_model_name(pf, input_dtype, output0_dtype, output1_dtype):
    return "{}_{}_{}_{}".format(
        pf,
        np.dtype(input_dtype).name,
        np.dtype(output0_dtype).name,
        np.dtype(output1_dtype).name,
    )


def get_sequence_model_name(pf, dtype):
    return "{}_sequence_{}".format(pf, np.dtype(dtype).name)


def get_dyna_sequence_model_name(pf, dtype):
    return "{}_dyna_sequence_{}".format(pf, np.dtype(dtype).name)


def get_zero_model_name(pf, io_cnt, dtype):
    return "{}_zero_{}_{}".format(pf, io_cnt, np.dtype(dtype).name)


# FIXME: Remove this def when jetson supports TRT 8.5 (DLIS-4256)
def support_trt_uint8():
    try:
        import tensorrt as trt
    except:
        # tensorrt library is not found, detect from environment
        import os

        return not bool(int(os.environ.get("TEST_JETSON", 0)))
    # tensorrt library is found, return if uint8 is defined
    return hasattr(trt, "uint8")


class TestResultCollector(unittest.TestCase):
    # TestResultCollector stores test result and prints it to stdout. In order
    # to use this class, unit tests must inherit this class. Use
    # `check_test_results` bash function from `common/util.sh` to verify the
    # expected number of tests produced by this class

    @classmethod
    def setResult(cls, total, errors, failures):
        cls.total, cls.errors, cls.failures = total, errors, failures

    @classmethod
    def tearDownClass(cls):
        # this method is called when all the unit tests in a class are
        # finished.
        json_res = {"total": cls.total, "errors": cls.errors, "failures": cls.failures}
        with open("test_results.txt", "w+") as f:
            f.write(json.dumps(json_res))

    def run(self, result=None):
        # result argument stores the accumulative test results
        test_result = super().run(result)
        total = test_result.testsRun
        errors = len(test_result.errors)
        failures = len(test_result.failures)
        self.setResult(total, errors, failures)
