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
    """

    def __init__(self, output_tensors):
        self._output_tensors = output_tensors

    def output_tensors(self):
        """Get output tensors
        Returns
        -------
        list
            A list of Tensor objects
        """
        return self._output_tensors


class Tensor:
    """A Tensor object is used to represent inputs and output data for an
    InferenceRequest or InferenceResponse. The object is intended to be used
    by a single thread and simultaneously calling different methods with
    different threads is not supported and will cause undefined behavior.

    Parameters
    ----------
    name : str
        Tensor name
    numpy_array : numpy.ndarray
        A numpy array containing input/output data
    """

    def __init__(self, name, numpy_array):
        if not isinstance(numpy_array, (np.ndarray,)):
            raise_error("numpy_array must be a numpy array")

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

    def numpy_array(self):
        """Get the underlying numpy array
        Returns
        -------
        numpy.ndarray
            The numpy array
        """
        return self._numpy_array
