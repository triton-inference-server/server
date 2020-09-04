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

import triton_python_backend_utils as tpb_utils

import argparse
import concurrent.futures as futures
import importlib.util
import sys
import threading
import signal
import time

import numpy as np

from python_host_pb2 import *
from python_host_pb2_grpc import PythonInterpreterServicer, add_PythonInterpreterServicer_to_server
import grpc


def parse_startup_arguments():
    parser = argparse.ArgumentParser(description="Triton Python Host")
    parser.add_argument("--socket",
                        default=None,
                        required=True,
                        type=str,
                        help="Socket to comunicate with server")
    parser.add_argument("--model-path",
                        default=None,
                        required=True,
                        type=str,
                        help="Path to model code")
    parser.add_argument("--instance-name",
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
        spec = importlib.util.spec_from_file_location('TritonPythonModel',
                                                      module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module_path = module_path

        if hasattr(module, 'TritonPythonModel'):
            self.backend = module.TritonPythonModel()
        else:
            raise NotImplementedError(
                'TritonPythonModel class doesn\'t exist in ' + module_path)

    def Init(self, request, context):
        """Init is called on TRITONBACKEND_ModelInstanceInitialize. `request`
        object contains an args key which includes a `model_config` key
        containing the model configuration. This paramter is passed by
        default to every ModelInstance.
        """

        backend = self.backend

        if not hasattr(request, 'args'):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('request objects does\'nt have args attribute')
            return Empty()

        if hasattr(backend, 'initialize'):
            args = {x.key: x.value for x in request.args}
            try:
                self.backend.initialize(args)
            except tpb_utils.TritonModelException as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(e.message())

        return Empty()

    def Fini(self, request, context):
        """Fini is called on TRITONBACKEND_ModelInstanceFinalize. Model
        can perform any necessary clean up in the `finalize` function.
        """
        if hasattr(self.backend, 'finalize'):
            try:
                self.backend.finalize()
            except tpb_utils.TritonModelException as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(e.message())

        return Empty()

    def Execute(self, request, context):
        """Execute is called on TRITONBACKEND_ModelInstanceExecute. Inference
        happens in this function. This function mainly converts gRPC
        protobufs to the triton_python_backend_utils.InferenceRequest and
        triton_python_backend_utils.InferenceResponse.

        Parameters
        ----------
        request : python_host_pb2.ExecuteRequest
            Contains a `requests` attribute which is a list of python_host_pb2.InferenceRequest
        """

        requests = request.requests
        inference_requests = []
        for request in requests:
            # This object contains a list of tpb_utils.Tensor
            input_tensors = []
            for request_input in request.inputs:
                x = request_input
                tensor = tpb_utils.Tensor(
                    x.name,
                    np.frombuffer(x.raw_data,
                                  dtype=tpb_utils.triton_to_numpy_type(
                                      x.dtype)).reshape(x.dims))
                input_tensors.append(tensor)

            request_id = request.id
            correlation_id = request.correlation_id
            requested_output_names = request.requested_output_names
            inference_request = tpb_utils.InferenceRequest(
                input_tensors, request_id, correlation_id,
                requested_output_names)
            inference_requests.append(inference_request)

        # Execute inference on the Python backend responses contains a list of
        # triton_python_backend_utils.InferenceResponse. Each backend must
        # implement an execute method
        if not hasattr(self.backend, 'execute'):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Backend does not implement `execute` method')
            return ExecuteResponse()

        responses = self.backend.execute(inference_requests)

        # Make sure that number of InferenceResponse and InferenceRequest
        # objects match
        if len(inference_requests) != len(responses):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                'Number of inference responses and requests don\'t match ( requests='
                + len(inference_requests) + ' != responses=' + len(responses) +
                ')')
            return ExecuteResponse()

        exec_responses = []
        for response in responses:
            # If there is an error do not look into output_tensors
            if response.has_error():
                error = Error(message=response.error().message())
                inference_response = InferenceResponse(outputs=[],
                                                       error=error,
                                                       failed=True)
                exec_responses.append(inference_response)
                continue

            output_tensors = response.output_tensors()
            response_tensors = []

            for output_tensor in output_tensors:
                output_np_array = output_tensor.as_numpy()
                tensor = Tensor(name=output_tensor.name(),
                                dtype=tpb_utils.numpy_to_triton_type(
                                    output_np_array.dtype.type),
                                dims=output_np_array.shape,
                                raw_data=output_np_array.tobytes())
                response_tensors.append(tensor)
            exec_responses.append(InferenceResponse(outputs=response_tensors))
        execute_response = ExecuteResponse(responses=exec_responses)

        return execute_response


if __name__ == "__main__":
    signal_received = False
    FLAGS = parse_startup_arguments()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    # Create an Event to keep the GRPC server running
    event = threading.Event()
    python_host = PythonHost(module_path=FLAGS.model_path)
    add_PythonInterpreterServicer_to_server(python_host, server)

    def interrupt_handler(signum, frame):
        pass

    def sigterm_handler(signum, frame):
        global signal_received
        if not signal_received:
            signal_received = True
        else:
            return

        event.set()

    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    server.add_insecure_port(FLAGS.socket)
    server.start()
    event.wait()
    server.stop(grace=5)
