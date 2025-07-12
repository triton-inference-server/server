# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import struct

import numpy as np

TRITON_STRING_TO_NUMPY = {
    "TYPE_BOOL": bool,
    "TYPE_UINT8": np.uint8,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.int8,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.object_,
}


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
        return ()

    # If the input is a tensor of string/bytes objects, then must flatten those
    # into a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C" order.
    if (input_tensor.dtype == np.object_) or (input_tensor.dtype.type == np.bytes_):
        flattened_ls = []
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order="C"):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if input_tensor.dtype == np.object_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = str(obj.item()).encode("utf-8")
            else:
                s = obj.item()
            flattened_ls.append(struct.pack("<I", len(s)))
            flattened_ls.append(s)
        flattened = b"".join(flattened_ls)
        return flattened
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
    return np.array(strs, dtype=np.object_)


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


def get_output_tensor_by_name(inference_response, name):
    """Find an output Tensor in the inference_response that has the given
    name
    Parameters
    ----------
    inference_response : InferenceResponse
        InferenceResponse object
    name : str
        name of the output Tensor object
    Returns
    -------
    Tensor
        The output Tensor with the specified name, or None if no
        output Tensor with this name exists
    """
    output_tensors = inference_response.output_tensors()
    for output_tensor in output_tensors:
        if output_tensor.name() == name:
            return output_tensor

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
    if "input" in model_config:
        inputs = model_config["input"]
        for input_properties in inputs:
            if input_properties["name"] == name:
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
    if "output" in model_config:
        outputs = model_config["output"]
        for output_properties in outputs:
            if output_properties["name"] == name:
                return output_properties

    return None


def using_decoupled_model_transaction_policy(model_config):
    """Whether or not the model is configured with decoupled
    transaction policy.
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration

    Returns
    -------
    bool
        True if the model is configured with decoupled transaction
        policy.
    """
    if "model_transaction_policy" in model_config:
        return model_config["model_transaction_policy"]["decoupled"]

    return False


def triton_to_numpy_type(data_type):
    if data_type == 1:
        return np.bool_
    elif data_type == 2:
        return np.uint8
    elif data_type == 3:
        return np.uint16
    elif data_type == 4:
        return np.uint32
    elif data_type == 5:
        return np.uint64
    elif data_type == 6:
        return np.int8
    elif data_type == 7:
        return np.int16
    elif data_type == 8:
        return np.int32
    elif data_type == 9:
        return np.int64
    elif data_type == 10:
        return np.float16
    elif data_type == 11:
        return np.float32
    elif data_type == 12:
        return np.float64
    elif data_type == 13:
        return np.object_


def numpy_to_triton_type(data_type):
    if data_type == np.bool_:
        return 1
    elif data_type == np.uint8:
        return 2
    elif data_type == np.uint16:
        return 3
    elif data_type == np.uint32:
        return 4
    elif data_type == np.uint64:
        return 5
    elif data_type == np.int8:
        return 6
    elif data_type == np.int16:
        return 7
    elif data_type == np.int32:
        return 8
    elif data_type == np.int64:
        return 9
    elif data_type == np.float16:
        return 10
    elif data_type == np.float32:
        return 11
    elif data_type == np.float64:
        return 12
    elif data_type == np.object_ or data_type == np.bytes_:
        return 13


def triton_string_to_numpy(triton_type_string):
    return TRITON_STRING_TO_NUMPY[triton_type_string]


class ModelConfig:
    """An object of ModelConfig class is used to describe
    the model configuration for autocomplete.
    Parameters
    ----------
    model_config : ModelConfig Object
        Object containing the model configuration. Only the max_batch_size, inputs
        and outputs properties can be modified for auto-complete model configuration.
    """

    def __init__(self, model_config):
        self._model_config = json.loads(model_config)

    def as_dict(self):
        """Provide the read-only access to the model configuration
        Returns
        -------
        dict
            dictionary type of the model configuration contained in
            the ModelConfig object
        """
        return self._model_config

    def set_max_batch_size(self, max_batch_size):
        """Set the max batch size for the model.
        Parameters
        ----------
        max_batch_size : int
            The max_batch_size to be set.
        Raises
        ------
        ValueError
            If configuration has specified max_batch_size non-zero value which
            is larger than the max_batch_size to be set for the model.
        """
        if self._model_config["max_batch_size"] > max_batch_size:
            raise ValueError(
                "configuration specified max_batch_size "
                + str(self._model_config["max_batch_size"])
                + ", but in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' specified max_batch_size "
                + str(max_batch_size)
            )
        else:
            self._model_config["max_batch_size"] = max_batch_size

    def set_dynamic_batching(self):
        """Set dynamic_batching as the scheduler for the model if no scheduler
        is set. If dynamic_batching is set in the model configuration, then no
        action is taken and return success.
        Raises
        ------
        ValueError
            If the 'sequence_batching' or 'ensemble_scheduling' scheduler is
            set for this model configuration.
        """
        found_scheduler = None
        if "sequence_batching" in self._model_config:
            found_scheduler = "sequence_batching"
        elif "ensemble_scheduling" in self._model_config:
            found_scheduler = "ensemble_scheduling"

        if found_scheduler != None:
            raise ValueError(
                "Configuration specified scheduling_choice as '"
                + found_scheduler
                + "', but auto-complete-config "
                "function for model '"
                + self._model_config["name"]
                + "' tries to set scheduling_choice as 'dynamic_batching'"
            )

        if "dynamic_batching" not in self._model_config:
            self._model_config["dynamic_batching"] = {}

    def add_input(self, input):
        """Add the input for the model.
        Parameters
        ----------
        input : dict
            The input to be added.
        Raises
        ------
        ValueError
            If input contains property other than 'name', 'data_type',
            'dims', 'optional' or any of the non-optional properties
            are not set, or if an input with the same name already exists
            in the configuration but has different data_type or dims property
        """
        valid_properties = ["name", "data_type", "dims", "optional"]
        for current_property in input:
            if current_property not in valid_properties:
                raise ValueError(
                    "input '"
                    + input["name"]
                    + "' in auto-complete-config function for model '"
                    + self._model_config["name"]
                    + "' contains property other than 'name', 'data_type', 'dims' and 'optional'."
                )

        if "name" not in input:
            raise ValueError(
                "input in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'name' property."
            )
        elif "data_type" not in input:
            raise ValueError(
                "input '"
                + input["name"]
                + "' in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'data_type' property."
            )
        elif "dims" not in input:
            raise ValueError(
                "input '"
                + input["name"]
                + "' in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'dims' property."
            )

        for current_input in self._model_config["input"]:
            if input["name"] == current_input["name"]:
                if (
                    current_input["data_type"] != "TYPE_INVALID"
                    and current_input["data_type"] != input["data_type"]
                ):
                    raise ValueError(
                        "unable to load model '"
                        + self._model_config["name"]
                        + "', configuration expects datatype "
                        + current_input["data_type"]
                        + " for input '"
                        + input["name"]
                        + "', model provides "
                        + input["data_type"]
                    )
                elif current_input["dims"] and current_input["dims"] != input["dims"]:
                    raise ValueError(
                        "model '"
                        + self._model_config["name"]
                        + "', tensor '"
                        + input["name"]
                        + "': the model expects dims "
                        + str(input["dims"])
                        + " but the model configuration specifies dims "
                        + str(current_input["dims"])
                    )
                elif (
                    "optional" in current_input
                    and "optional" in input
                    and current_input["optional"] != input["optional"]
                ):
                    raise ValueError(
                        "model '"
                        + self._model_config["name"]
                        + "', tensor '"
                        + input["name"]
                        + "': the model expects optional "
                        + str(input["optional"])
                        + " but the model configuration specifies optional "
                        + str(current_input["optional"])
                    )
                else:
                    current_input["data_type"] = input["data_type"]
                    current_input["dims"] = input["dims"]
                    if "optional" in input:
                        current_input["optional"] = input["optional"]
                    return

        self._model_config["input"].append(input)

    def add_output(self, output):
        """Add the output for the model.
        Parameters
        ----------
        output : dict
            The output to be added.
        Raises
        ------
        ValueError
            If output contains property other than 'name', 'data_type'
            and 'dims' or any of the properties are not set, or if an
            output with the same name already exists in the configuration
            but has different data_type or dims property
        """
        valid_properties = ["name", "data_type", "dims"]
        for current_property in output:
            if current_property not in valid_properties:
                raise ValueError(
                    "output '"
                    + output["name"]
                    + "' in auto-complete-config function for model '"
                    + self._model_config["name"]
                    + "' contains property other than 'name', 'data_type' and 'dims'."
                )

        if "name" not in output:
            raise ValueError(
                "output in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'name' property."
            )
        elif "data_type" not in output:
            raise ValueError(
                "output '"
                + output["name"]
                + "' in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'data_type' property."
            )
        elif "dims" not in output:
            raise ValueError(
                "output '"
                + output["name"]
                + "' in auto-complete-config function for model '"
                + self._model_config["name"]
                + "' is missing 'dims' property."
            )

        for current_output in self._model_config["output"]:
            if output["name"] == current_output["name"]:
                if (
                    current_output["data_type"] != "TYPE_INVALID"
                    and current_output["data_type"] != output["data_type"]
                ):
                    raise ValueError(
                        "unable to load model '"
                        + self._model_config["name"]
                        + "', configuration expects datatype "
                        + current_output["data_type"]
                        + " for output '"
                        + output["name"]
                        + "', model provides "
                        + output["data_type"]
                    )
                elif (
                    current_output["dims"] and current_output["dims"] != output["dims"]
                ):
                    raise ValueError(
                        "model '"
                        + self._model_config["name"]
                        + "', tensor '"
                        + output["name"]
                        + "': the model expects dims "
                        + str(output["dims"])
                        + " but the model configuration specifies dims "
                        + str(current_output["dims"])
                    )
                else:
                    current_output["data_type"] = output["data_type"]
                    current_output["dims"] = output["dims"]
                    return

        self._model_config["output"].append(output)

    def set_model_transaction_policy(self, transaction_policy_dict):
        """
        Set model transaction policy for the model.
        Parameters
        ----------
        transaction_policy_dict : dict
            The dict, containing all properties to be set as a part
            of `model_transaction_policy` field.
        Raises
        ------
        ValueError
            If transaction_policy_dict contains property other
            than 'decoupled', or if `model_transaction_policy` already exists
            in the configuration, but has different `decoupled` property.
        """
        valid_properties = ["decoupled"]
        for current_property in transaction_policy_dict.keys():
            if current_property not in valid_properties:
                raise ValueError(
                    "model transaction property in auto-complete-config "
                    + "function for model '"
                    + self._model_config["name"]
                    + "' contains property other than 'decoupled'."
                )

        if "model_transaction_policy" not in self._model_config:
            self._model_config["model_transaction_policy"] = {}

        if "decoupled" in transaction_policy_dict.keys():
            if (
                "decoupled" in self._model_config["model_transaction_policy"]
                and self._model_config["model_transaction_policy"]["decoupled"]
                != transaction_policy_dict["decoupled"]
            ):
                raise ValueError(
                    "trying to change decoupled property in auto-complete-config "
                    + "for model '"
                    + self._model_config["name"]
                    + "', which is already set to '"
                    + str(self._model_config["model_transaction_policy"]["decoupled"])
                    + "'."
                )

            self._model_config["model_transaction_policy"][
                "decoupled"
            ] = transaction_policy_dict["decoupled"]


TRITONSERVER_REQUEST_FLAG_SEQUENCE_START = 1
TRITONSERVER_REQUEST_FLAG_SEQUENCE_END = 2
TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1
TRITONSERVER_REQUEST_RELEASE_ALL = 1
TRITONSERVER_REQUEST_RELEASE_RESCHEDULE = 2
