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

import utils

import argparse
import concurrent.futures as futures
import importlib.util
import sys
import threading
import signal

import numpy as np

from python_host_pb2 import *
from python_host_pb2_grpc import PythonInterpreterServicer, add_PythonInterpreterServicer_to_server
import grpc

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
    # TRITONSERVER_TYPE_BYTES
    13: np.bytes_
}

NUMPY_TO_TRITION_TYPE = {v: k for k, v in TRITION_TO_NUMPY_TYPE.items()}


def protobuf_to_numpy_type(data_type):
    return TRITION_TO_NUMPY_TYPE[data_type]


def numpy_to_protobuf_type(data_type):
    return NUMPY_TO_TRITION_TYPE[data_type]


def parse_startup_arguments():
    parser = argparse.ArgumentParser(description="Triton Python Host")
    parser.add_argument("--socket",
                        default=None,
                        required=True,
                        type=str,
                        help="Socket to comunicate with server")
    parser.add_argument("--model_path",
                        default=None,
                        required=True,
                        type=str,
                        help="Path to model code")
    parser.add_argument("--instance_name",
                        default=None,
                        required=True,
                        type=str,
                        help="Triton instance name")
    return parser.parse_args()


class PythonHost(PythonInterpreterServicer):
    """This class handles inference request for python script.
    """

    def __init__(self, module_path, *args, **kwargs):
        super(PythonInterpreterServicer, self).__init__(*args, **kwargs)
        spec = importlib.util.spec_from_file_location("TritonPythonBackend",
                                                      module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "TritonPythonBackend"):
            self.initializer_func = module.TritonPythonBackend
        else:
            self.initializer_func = None
        self.model = None

    def Init(self, request, context):
        """Init is called on TRITONBACKEND_ModelInstanceInitialize. `request`
        object contains an args which includes a `model_config` key
        containing the model configuration. This paramter is passed by
        default to every ModelInstance.
        """
        # TODO: Add error handling and returning correct status codes
        args = {x.key: x.value for x in request.args}
        self.model = self.initializer_func(args)

    def Fini(self, request, context):
        """Fini is called on TRITONBACKEND_ModelInstanceFinalize. Model
        can perform any necessary clean up in the `shutdown` function.
        """
        # TODO: Add error handling and returning correct status codes
        if hasattr(self.model, "shutdown"):
            self.model.shutdown()

        del self.model

    def Execute(self, request, context):
        """Execute is called on TRITONBACKEND_ModelInstanceExecute. Inference
        happens in this function. This function mainly converts gRPC protobufs
        to the utils.InferenceRequest and utils.InferenceResponse.

        Parameters
        ----------
        request : python_host_pb2.ExecuteRequest
            Contains a `requests` attribute which is a list of python_host_pb2.InferenceRequest
        """

        requests = request.requests
        inference_requests = []
        for request in requests:
            # This object contains a list of utils.Tensor
            input_tensors = []
            for request_input in request.inputs:
                x = request_input
                tensor = utils.Tensor(
                    x.name,
                    np.frombuffer(x.raw_data,
                                  dtype=protobuf_to_numpy_type(
                                      x.dtype)).reshape(x.dims))
                input_tensors.append(tensor)

            request_id = request.id
            correlation_id = request.correlation_id
            requested_output_names = request.requested_output_names
            inference_request = utils.InferenceRequest(input_tensors,
                                                       request_id,
                                                       correlation_id,
                                                       requested_output_names)
            inference_requests.append(inference_request)

        # Execute inference on the Python model
        # responses contains a list of utils.InferenceResponse
        responses = self.model(inference_requests)

        exec_responses = []
        for response in responses:
            output_tensors = response.output_tensors()
            response_tensors = []

            for output_tensor in output_tensors:
                output_np_array = output_tensor.numpy_array()
                tensor = Tensor(name=output_tensor.name(),
                                dtype=numpy_to_protobuf_type(
                                    output_np_array.dtype.type),
                                dims=output_np_array.shape,
                                raw_data=output_np_array.tobytes())
                response_tensors.append(tensor)
            exec_responses.append(InferenceResponse(inputs=response_tensors))
        execute_response = ExecuteResponse(responses=exec_responses)

        return execute_response


if __name__ == "__main__":
    FLAGS = parse_startup_arguments()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PythonInterpreterServicer_to_server(
        PythonHost(module_path=FLAGS.model_path), server)

    server.add_insecure_port(FLAGS.socket)
    server.start()

    def signal_handler(signal, frame):
        server.stop(5)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

    server.start()
    server.wait_for_termination()
