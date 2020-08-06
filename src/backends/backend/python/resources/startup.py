import argparse
import concurrent.futures as futures
import importlib.util
import sys
import threading

import numpy as np

from python_host_pb2 import *
from python_host_pb2_grpc import PythonInterpreterServicer, add_PythonInterpreterServicer_to_server
from model_config_pb2 import DataType
import grpc

_TYPE_MAPPING_DICTIONARY = {
    DataType.TYPE_BOOL: np.bool,
    DataType.TYPE_UINT8: np.uint8,
    DataType.TYPE_UINT16: np.uint16,
    DataType.TYPE_UINT32: np.uint32,
    DataType.TYPE_UINT64: np.uint64,
    DataType.TYPE_INT8: np.int8,
    DataType.TYPE_INT16: np.int16,
    DataType.TYPE_INT32: np.int32,
    DataType.TYPE_INT64: np.int64,
    DataType.TYPE_FP16: np.float16,
    DataType.TYPE_FP32: np.float32,
    DataType.TYPE_FP64: np.float64,
    DataType.TYPE_STRING: np.bytes_
}

def protobuf_to_numpy_type(data_type):
    return _TYPE_MAPPING_DICTIONARY[data_type]

def numpy_to_protobuf_type(data_type):
    return {v: k for k, v in _TYPE_MAPPING_DICTIONARY.items()}[data_type]

def parse_startup_arguments():
    parser = argparse.ArgumentParser(description="TensorRT-Inference-Server Python Host")
    parser.add_argument("--socket", default=None, required=True, type=str, 
                        help="Socket to comunicate with server")
    parser.add_argument("--model_path", default=None, required=True, type=str,
                        help="Path to model code")
    parser.add_argument("--instance_name", default=None, required=True, type=str,
                        help="TRTIS instance name")
    return parser.parse_args()

lock = threading.Lock()
cv = threading.Condition(lock)

class PythonHost(PythonInterpreterServicer):
    r"""
    This class handles inference request for python script.
    """
    def __init__(self, module_path, *args, **kwargs):
        super(PythonInterpreterServicer, self).__init__(*args, **kwargs)
        spec = importlib.util.spec_from_file_location("trtis", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "initialize_model"):
            self.initializer_func = module.initialize_model
        elif hasattr(module, "trtis"):
            self.initializer_func = module.trtis
        else:
            self.initializer_func = None
        self.model = None

    def InterpreterInit(self, request, context):
        if self.model:
            return StatusCode(code=0)

        if not self.initializer_func:
            return StatusCode(code=1)

        args = argparse.Namespace(**{x.key: x.value for x in request.model_command})
        self.model = self.initializer_func(args)

        if not self.model:
            return StatusCode(code=2)

        return StatusCode(code=0)

    def InterpreterShutdown(self, request, context):
        if hasattr(self.model, "shutdown"):
            self.model.shutdown()

        del self.model
        with cv:
            cv.notify()
        return StatusCode(code=0)

    def InferenceRequest(self, request, context):
        input_dictionary = {x.name: np.frombuffer(x.raw_data, 
            dtype=protobuf_to_numpy_type(x.dtype)).reshape(x.dims) for x in request.tensors}
        if not all([(request.batch_size == x.shape[0]) for x in input_dictionary.values()]):
            return InferenceBatch(batch_size=-1, tensors=[])

        output_dictionary = self.model(input_dictionary)
        output_tensors = [InferenceData(name=name, dims=array.shape,
                                        dtype=numpy_to_protobuf_type(array.dtype.type),
                                        raw_data=array.tobytes()) 
                          for name, array in output_dictionary.items()]

        return InferenceBatch(batch_size=request.batch_size,
                              tensors=output_tensors)

if __name__ == "__main__":
    FLAGS = parse_startup_arguments()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PythonInterpreterServicer_to_server(PythonHost(module_path=FLAGS.model_path), 
                                            server)

    server.add_insecure_port(FLAGS.socket)
    server.start()

    with cv:
        cv.wait()
    server.stop(grace=5)

