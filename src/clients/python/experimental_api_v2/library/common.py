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

__all__ = ['raise_error',
        'np_to_trtis_dtype',
        'trtis_to_np_dtype',
        'InferenceServerException']

def raise_error(msg):
    """
    Raise error with the provided message
    """
    raise InferenceServerException(msg=msg) from None


class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : RequestStatus Protobuf
        The protobuf message describing the error

    """

    def __init__(self, msg, status=None, debug_details=None):
        self._msg = msg
        self._status = status
        self._debug_details = debug_details

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        if self._status is not None:
            msg = '[' + self._status + '] ' + msg
        return msg

    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        return self._msg

    def status(self):
        """Get the status of the exception.

        Returns
        -------
        str
            Returns the status of the exception

        """
        return self._status

    def debug_details(self):
        """Get the detailed information about the exception for debugging purposes

        Returns
        -------
        str
            Returns string in JSON format containing the exception details            

        """
        return self._debug_details


def np_to_trtis_dtype(np_dtype):
    if np_dtype == np.bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    # FIXMEPV2 String support
    # elif np_dtype == np_dtype_string:
    #    return "STRING"
    return None

def trtis_to_np_dtype(dtype):
    if dtype == "BOOL":
        return np.bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    # FIXMEPV2 String support
    # elif dtype == "STRING":
    #    return np_dtype_string
    return None
