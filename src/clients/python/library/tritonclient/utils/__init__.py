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
import struct


def raise_error(msg):
    """
    Raise error with the provided message
    """
    raise InferenceServerException(msg=msg) from None


def serialized_byte_size(tensor_value):
    """
    Get the underlying number of bytes for a numpy ndarray.

    Parameters
    ----------
    tensor_value : numpy.ndarray
        Numpy array to calculate the number of bytes for.

    Returns
    -------
    int
        Number of bytes present in this tensor
    """

    if tensor_value.dtype != np.object_:
        raise_error('The tensor_value dtype must be np.object_')

    if tensor_value.size > 0:
        total_bytes = 0
        for obj in np.nditer(tensor_value, flags=["refs_ok"], order='C'):
            total_bytes += len(obj.item())
        return total_bytes
    else:
        return 0


class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    msg : str
        A brief description of error

    status : str
        The error code

    debug_details : str
        The additional details on the error

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
        """Get the detailed information about the exception
        for debugging purposes

        Returns
        -------
        str
            Returns the exception details

        """
        return self._debug_details


def np_to_triton_dtype(np_dtype):
    if np_dtype == bool:
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
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def triton_to_np_dtype(dtype):
    if dtype == "BOOL":
        return bool
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
    elif dtype == "BYTES":
        return np.object_
    return None


def serialize_byte_tensor(input_tensor):
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.

    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in 'C' order.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """

    if input_tensor.size == 0:
        return np.empty([0], dtype=np.object_)

    # If the input is a tensor of string/bytes objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C"
    # order.
    if (input_tensor.dtype == np.object_) or (input_tensor.dtype.type
                                              == np.bytes_):
        flattened_ls = []
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order='C'):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if input_tensor.dtype == np.object_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = str(obj.item()).encode('utf-8')
            else:
                s = obj.item()
            flattened_ls.append(struct.pack("<I", len(s)))
            flattened_ls.append(s)
        flattened = b''.join(flattened_ls)
        flattened_array = np.asarray(flattened, dtype=np.object_)
        if not flattened_array.flags['C_CONTIGUOUS']:
            flattened_array = np.ascontiguousarray(flattened_array,
                                                   dtype=np.object_)
        return flattened_array
    else:
        raise_error("cannot serialize bytes tensor: invalid datatype")
    return None


def deserialize_bytes_tensor(encoded_tensor):
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects

    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in 'C' order.
   
    """
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))
