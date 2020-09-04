# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np

TRITION_TO_NUMPY_TYPE = {
    # TRITONSERVER_TYPE_BOOL
    1: np.bool,
    # TRITONSERVER_TYPE_UINT8
    2: np.uint8,
    # TRITONSERVER_TYPE_UINT16
    3: np.uint16,
    # TRITONSERVER_TYPE_UINT32
    4: np.uint32,
    # TRITONSERVER_TYPE_UINT64
    5: np.uint64,
    # TRITONSERVER_TYPE_INT8
    6: np.int8,
    # TRITONSERVER_TYPE_INT16
    7: np.int16,
    # TRITONSERVER_TYPE_INT32
    8: np.int32,
    # TRITONSERVER_TYPE_INT64
    9: np.int64,
    # TRITONSERVER_TYPE_FP16
    10: np.float16,
    # TRITONSERVER_TYPE_FP32
    11: np.float32,
    # TRITONSERVER_TYPE_FP64
    12: np.float64,
    # TRITONSERVER_TYPE_STRING
    13: np.str_
}

NUMPY_TO_TRITION_TYPE = {v: k for k, v in TRITION_TO_NUMPY_TYPE.items()}


class InferenceRequest:
    """InferenceRequest represents a request for inference for a model that
    executes using this backend.

    Parameters
    ----------
    inputs : list
        A list of Tensor objects, each describing data for an input tensor
        required by the model
    request_id : str
        ID assoiciated with this request, or empty string if no ID is
        associated with the request.
    correlation_id : str
        Correlation ID associated with this request, or empty string if no
        correlation ID is associated with the request.
    requested_output_name : list
        The names of the output tensors that should be calculated and
        returned for this request.
    """

    def __init__(self, inputs, request_id, correlation_id,
                 requested_output_names):
        self._inputs = inputs
        self._request_id = request_id
        self._correlation_id = correlation_id
        self._requested_output_names = requested_output_names

    def inputs(self):
        """Get input tensors
        Returns
        ----
        list
            A list of input Tensor objects
        """
        return self._inputs

    def request_id(self):
        """Get request ID
        Returns
        -------
        str
            Request ID
        """
        return self._request_id

    def correlation_id(self):
        """Get correlation ID
        Returns
        -------
        int
            Request correlation ID
        """
        return self._correlation_id

    def requested_output_names(self):
        """Get requested output names
        Returns
        -------
        list
            A list of strings, each describing the requested output name
        """
        return self._requested_output_names


class InferenceResponse:
    """An InfrenceResponse object is used to represent the response to an
    inference request.

    Parameters
    ----------
    output_tensors : list
        A list of Tensor objects, each describing data for an output tensor
        required the InferenceRequest
    error : TritonError
        A TritonError object describing any errror encountered while creating
        resposne
    """

    def __init__(self, output_tensors, error=None):
        self._output_tensors = output_tensors
        self._err = error

    def output_tensors(self):
        """Get output tensors
        Returns
        -------
        list
            A list of Tensor objects
        """
        return self._output_tensors

    def has_error(self):
        """True if response has error
        Returns
        -------
        boolean
            A boolean indicating whether response has an error
        """
        return self._err != None

    def error(self):
        """Get TritonError for this inference response
        Returns
        -------
        TritonError
            A TritonError containing the error
        """
        return self._err


class Tensor:
    """A Tensor object is used to represent inputs and output data for an
    InferenceRequest or InferenceResponse.

    Parameters
    ----------
    name : str
        Tensor name
    numpy_array : numpy.ndarray
        A numpy array containing input/output data
    """

    def __init__(self, name, numpy_array):
        if not isinstance(numpy_array, (np.ndarray,)):
            raise TritonModelException("numpy_array must be a numpy array")

        self._name = name
        self._numpy_array = numpy_array

    def name(self):
        """Get the name of tensor
        Returns
        -------
        str
            The name of tensor
        """
        return self._name

    def as_numpy(self):
        """Get the underlying numpy array
        Returns
        -------
        numpy.ndarray
            The numpy array
        """
        return self._numpy_array


class TritonError:
    """Error indicating non-Success status.

    Parameters
    ----------
    msg : str
        A brief description of error
    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg

    def message(self):
        """Get the error message.

        Returns
        -------
        str
            The message associated with this error, or None if no message.

        """
        return self._msg


class TritonModelException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    msg : str
        A brief description of error
    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg

    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.
        """
        return self._msg


def get_input_tensor_by_name(inference_request, name):
    """Find an input Tensor in the inference_request that has the given
    name

    Parameters
    ----------
    inference_request : InferenceRequest
        InferenceRequest object
    name : str
        name of the input Tensor object

    Returns
    -------
    Tensor
        The input Tensor with the specified name, or None if no
        input Tensor with this name exists
    """
    input_tensors = inference_request.inputs()
    for input_tensor in input_tensors:
        if input_tensor.name() == name:
            return input_tensor

    return None


def get_input_config_by_name(model_config, name):
    """Get input properties corresponding to the input
    with given `name`

    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the input object

    Returns
    -------
    dict
        A dictionary containing all the properties for a given input
        name, or None if no input with this name exists
    """
    if 'input' in model_config:
        inputs = model_config['input']
        for input_properties in inputs:
            if input_properties['name'] == name:
                return input_properties

    return None


def get_output_config_by_name(model_config, name):
    """Get output properties corresponding to the output
    with given `name`

    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the output object

    Returns
    -------
    dict
        A dictionary containing all the properties for a given output
        name, or None if no output with this name exists
    """
    if 'output' in model_config:
        outputs = model_config['output']
        for output_properties in outputs:
            if output_properties['name'] == name:
                return output_properties

    return None


def triton_to_numpy_type(data_type):
    return TRITION_TO_NUMPY_TYPE[data_type]


def numpy_to_triton_type(data_type):
    return NUMPY_TO_TRITION_TYPE[data_type]
