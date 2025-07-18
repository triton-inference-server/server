"""
Test model implementations for Triton Python Backend testing
"""

import numpy as np
import json
import time
import sys
import os

# Mock triton_python_backend_utils for standalone testing
class MockPbUtils:
    @staticmethod
    def get_input_tensor_by_name(request, name):
        class MockTensor:
            def __init__(self, data):
                self._data = data
            
            def as_numpy(self):
                return self._data
        
        # Return mock tensor with test data
        return MockTensor(request.get('inputs', {}).get(name, np.array([1.0])))
    
    @staticmethod
    def Tensor(name, data):
        return {'name': name, 'data': data}
    
    @staticmethod
    def InferenceResponse(output_tensors=None, error=None):
        if error:
            return {'error': error}
        return {'outputs': output_tensors}
    
    @staticmethod
    def InferenceError(message):
        return {'error': message}

# Try to import real pb_utils, fall back to mock
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = MockPbUtils


class SimpleAddModel:
    """Simple model that adds a constant to input"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.add_value = float(args.get('model_instance_device_id', 1))
        
    def execute(self, requests):
        responses = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_data = in_0.as_numpy()
            
            # Add constant
            output_data = input_data + self.add_value
            
            out_tensor = pb_utils.Tensor("OUTPUT0", output_data)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        
        return responses
    
    def finalize(self):
        pass


class MatrixMultiplyModel:
    """Model that performs matrix multiplication"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        # Initialize weight matrix
        self.weight = np.random.randn(10, 10).astype(np.float32)
        
    def execute(self, requests):
        responses = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_matrix = in_0.as_numpy()
            
            # Ensure input is 2D
            if len(input_matrix.shape) == 1:
                input_matrix = input_matrix.reshape(1, -1)
            
            # Matrix multiplication
            try:
                output_matrix = np.matmul(input_matrix, self.weight)
                out_tensor = pb_utils.Tensor("OUTPUT0", output_matrix)
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            except Exception as e:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.InferenceError(str(e)))
            
            responses.append(response)
        
        return responses
    
    def finalize(self):
        self.weight = None


class ErrorTestModel:
    """Model for testing error handling"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.error_probability = 0.2
        
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Randomly decide to error
            if np.random.random() < self.error_probability:
                error_msg = "Simulated inference error"
                response = pb_utils.InferenceResponse(
                    error=pb_utils.InferenceError(error_msg))
            else:
                in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
                data = in_0.as_numpy()
                
                # Simple pass-through
                out_tensor = pb_utils.Tensor("OUTPUT0", data)
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            
            responses.append(response)
        
        return responses


class MemoryIntensiveModel:
    """Model for testing memory management"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        # Allocate large buffers
        self.buffer_size = 10 * 1024 * 1024  # 10MB
        self.buffers = []
        
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Allocate temporary buffer
            temp_buffer = np.zeros(self.buffer_size // 4, dtype=np.float32)
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            data = in_0.as_numpy()
            
            # Simulate memory-intensive operation
            result = np.zeros_like(data)
            for i in range(100):
                result += data * 0.01
            
            out_tensor = pb_utils.Tensor("OUTPUT0", result)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
            
            # Clean up temp buffer
            del temp_buffer
        
        return responses
    
    def finalize(self):
        self.buffers.clear()


class ConcurrencyTestModel:
    """Model for testing concurrent execution"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.instance_id = args.get('model_instance_name', 'unknown')
        self.request_count = 0
        
    def execute(self, requests):
        import threading
        
        responses = []
        thread_id = threading.current_thread().ident
        
        for request in requests:
            self.request_count += 1
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            data = in_0.as_numpy()
            
            # Add metadata to output
            output = np.array([
                float(self.request_count),
                float(thread_id % 1000),
                float(len(data))
            ])
            
            # Simulate some processing time
            time.sleep(0.01)
            
            out_tensor = pb_utils.Tensor("OUTPUT0", output)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        
        return responses


class PlatformSpecificModel:
    """Model that tests platform-specific features"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.platform = sys.platform
        
    def execute(self, requests):
        responses = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            data = in_0.as_numpy()
            
            # Platform-specific operations
            if self.platform == 'darwin':  # macOS
                # Test macOS-specific features
                import subprocess
                try:
                    # Get system info using macOS commands
                    result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], 
                                          capture_output=True, text=True)
                    cpu_count = int(result.stdout.strip())
                except:
                    cpu_count = os.cpu_count() or 1
            else:
                cpu_count = os.cpu_count() or 1
            
            # Return platform info as output
            output = np.array([float(cpu_count)])
            
            out_tensor = pb_utils.Tensor("OUTPUT0", output)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        
        return responses


class BatchProcessingModel:
    """Model that efficiently processes batched requests"""
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
    def execute(self, requests):
        # Collect all inputs into a batch
        batch_inputs = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            batch_inputs.append(in_0.as_numpy())
        
        # Process as batch
        if batch_inputs:
            batch_array = np.vstack(batch_inputs)
            # Batch normalization
            mean = np.mean(batch_array, axis=0, keepdims=True)
            std = np.std(batch_array, axis=0, keepdims=True) + 1e-7
            normalized = (batch_array - mean) / std
            
            # Create individual responses
            responses = []
            for i in range(len(requests)):
                out_tensor = pb_utils.Tensor("OUTPUT0", normalized[i:i+1])
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)
        else:
            responses = []
        
        return responses


# Model registry for easy access
MODEL_REGISTRY = {
    'simple_add': SimpleAddModel,
    'matrix_multiply': MatrixMultiplyModel,
    'error_test': ErrorTestModel,
    'memory_intensive': MemoryIntensiveModel,
    'concurrency_test': ConcurrencyTestModel,
    'platform_specific': PlatformSpecificModel,
    'batch_processing': BatchProcessingModel
}


def get_model_class(model_name):
    """Get model class by name"""
    return MODEL_REGISTRY.get(model_name, SimpleAddModel)


# For Triton, export as TritonPythonModel
TritonPythonModel = SimpleAddModel  # Default model