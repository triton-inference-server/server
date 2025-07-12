# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os


def tf_to_triton_dtype(dtype):
    import tensorflow as tf

    if dtype == tf.float16:
        return "FP16"
    elif dtype == tf.float32:
        return "FP32"
    elif dtype == tf.float64:
        return "FP64"
    elif dtype == tf.int8:
        return "INT8"
    elif dtype == tf.uint8:
        return "UINT8"
    elif dtype == tf.uint16:
        return "UINT16"
    elif dtype == tf.uint32:
        return "UINT32"
    elif dtype == tf.uint64:
        return "UINT64"
    elif dtype == tf.int16:
        return "INT16"
    elif dtype == tf.int32:
        return "INT32"
    elif dtype == tf.int64:
        return "INT64"
    elif dtype == tf.bool:
        return "BOOL"
    elif dtype == tf.string:
        return "STRING"

    raise Exception("The data type in the TF model is not supported")


def parse_tf_tensors(saved_model_dir, tag_set, signature_def_key):
    from tensorflow.python.tools import saved_model_utils

    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)

    input_dict = {}
    input_signatures = list(
        meta_graph_def.signature_def[signature_def_key].inputs.values()
    )
    for input_signature in input_signatures:
        datatype = tf_to_triton_dtype(input_signature.dtype)
        shape = []
        for dim in input_signature.tensor_shape.dim:
            shape.append(dim.size)
        input_dict[input_signature.name] = [datatype, shape]

    output_dict = {}
    output_signatures = list(
        meta_graph_def.signature_def[signature_def_key].outputs.values()
    )
    for output_signature in output_signatures:
        datatype = tf_to_triton_dtype(output_signature.dtype)
        shape = []
        for dim in output_signature.tensor_shape.dim:
            shape.append(dim.size)
        output_dict[output_signature.name] = [datatype, shape]
    return input_dict, output_dict


def parse_io_tensors(tensors):
    tensors_dict = {}
    for t in [t for tensor in tensors for t in tensor]:
        name, datatype, shape_str = t.split(",")
        shape = [int(i) for i in shape_str.split("x")]
        tensors_dict[name] = [datatype, shape]

    return tensors_dict


def get_parameter_spec(key1, value):
    param_spec = 'parameters: {{key: "{}", value: {{string_value: "{}"}}}} \n'.format(
        key1, value
    )

    return param_spec


def create_modelconfig(
    model_name,
    max_batch_size,
    inputs,
    outputs,
    compiled_model_path,
    nc_start_idx,
    nc_end_idx,
    threads_per_core,
    instance_count,
    enable_dynamic_batching,
    preferred_batch_size,
    max_queue_delay_microseconds,
):
    config = 'name: "{}"\n'.format(model_name)
    config += 'backend: "python"\n'
    config += "max_batch_size: {}\n".format(max_batch_size)
    if enable_dynamic_batching:
        config += """
dynamic_batching {
"""
        if preferred_batch_size is not None:
            config += """
    preferred_batch_size: {}
""".format(
                preferred_batch_size
            )
        if max_queue_delay_microseconds is not None:
            config += """
    max_queue_delay_microseconds: {}
""".format(
                max_queue_delay_microseconds
            )
        config += """
}\n"""
    for input_name in inputs.keys():
        data_type, shape = inputs[input_name]
        config += """
input [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n""".format(
            input_name, "TYPE_" + data_type, shape
        )
    for output_name in outputs.keys():
        data_type, shape = outputs[output_name]
        config += """
output [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n""".format(
            output_name, "TYPE_" + data_type, shape
        )
    config += """
instance_group [
    {{
        kind: KIND_MODEL
        count: {}
    }}
]\n""".format(
        instance_count
    )
    config += get_parameter_spec("COMPILED_MODEL", compiled_model_path)
    config += get_parameter_spec("NEURON_CORE_START_INDEX", nc_start_idx)
    config += get_parameter_spec("NEURON_CORE_END_INDEX", nc_end_idx)
    config += get_parameter_spec("NUM_THREADS_PER_CORE", threads_per_core)
    return config


def get_model_license():
    lic = """# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """
    return lic


def get_common_initialize_impl():
    init_impl = '''
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        if (len(model_config['instance_group']) != 1):
            raise pb_utils.TritonModelException(
                "this model supports only a single instance group, got {}".
                format(len(model_config['instance_group'])))

        instance_group_config = model_config['instance_group'][0]
        instance_count = instance_group_config['count']

        instance_idx = 0
        if instance_count > 1:
            instance_name_parts = args['model_instance_name'].split("_")
            if not instance_name_parts[-1].isnumeric():
                raise pb_utils.TritonModelException(
                    "internal error: the model instance name should end with '_<instance_idx>', got {}"
                    .format(args['model_instance_name']))
            instance_idx = int(instance_name_parts[-1])

        params = model_config['parameters']
        compiled_model = params['COMPILED_MODEL']['string_value']

        nc_start_idx = int(params['NEURON_CORE_START_INDEX']['string_value'])
        nc_end_idx = int(params['NEURON_CORE_END_INDEX']['string_value'])
        if nc_end_idx < nc_start_idx:
            raise pb_utils.TritonModelException(
                "the neuron core end index should be greater than or equal to the start index"
            )

        threads_per_core = int(params['NUM_THREADS_PER_CORE']['string_value'])
        if threads_per_core < 1:
            raise pb_utils.TritonModelException(
                "the number of threads per core should be greater than or equal to 1"
            )
        num_threads = (nc_end_idx - nc_start_idx + 1) * threads_per_core

        total_core_count = nc_end_idx - nc_start_idx + 1
        if (instance_count > total_core_count):
            raise pb_utils.TritonModelException(
                "can not distribute {} triton model instances to {} neuron cores"
                .format(instance_count, total_core_count))
        cores_per_instance = total_core_count // instance_count
'''
    return init_impl


def get_tensorflow_initialize_impl(is_inf2=False):
    init_impl = get_common_initialize_impl()
    init_impl += """
        self.input_list = []
        for config_input in model_config['input']:
            self.input_list.append(
                (config_input['name'], config_input['data_type'],
                 config_input['dims']))

        self.output_list = []
        for config_output in model_config['output']:
            self.output_list.append(
                (config_output['name'], config_output['data_type'],
                 config_output['dims']))

        os.environ["NEURON_RT_NUM_CORES"] = str(cores_per_instance)
"""
    if is_inf2:
        init_impl += """
        compiled_model = os.path.join(args['model_repository'], compiled_model)
        self.pred_list = [
            tf.keras.models.load_model(compiled_model)
            for _ in range(cores_per_instance)
        ] * threads_per_core
"""
    else:
        init_impl += """
        self.pred_list = [
            tf.contrib.predictor.from_saved_model(compiled_model)
            for _ in range(cores_per_instance)
        ] * threads_per_core
"""
    return init_impl


def get_pytorch_initialize_impl(is_inf2=False):
    init_impl = """
    def _validate_and_get_index(self, name):
        parts = name.split('__')
        if len(parts) != 2:
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index>, got {}"
                .format(name))

        if not parts[1].isnumeric():
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index> where <index> should be numeric, got {}"
                .format(name))

        return int(parts[1])

    def _validate_input_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.input_dict:
                raise pb_utils.TritonModelException(
                    "input corresponding to index {} not found".format(i))

    def _validate_output_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.output_dict:
                raise pb_utils.TritonModelException(
                    "output corresponding to index {} not found".format(i))
"""
    init_impl += get_common_initialize_impl()
    init_impl += """
        self.input_dict = {}
        expected_input_count = 0
        for config_input in model_config['input']:
            index = self._validate_and_get_index(config_input['name'])
            self.input_dict[index] = [
                config_input['name'], config_input['data_type'],
                config_input['dims']
            ]
            expected_input_count += 1
        self._validate_input_dict(expected_input_count)

        self.output_dict = {}
        for config_output in model_config['output']:
            index = self._validate_and_get_index(config_output['name'])
            self.output_dict[index] = [
                config_output['name'], config_output['data_type'],
                config_output['dims']
            ]

        adjusted_nc_start_idx = (instance_idx *
                                 cores_per_instance) + nc_start_idx
        cores_range = '{}-{}'.format(
            adjusted_nc_start_idx,
            (adjusted_nc_start_idx + cores_per_instance - 1))
        os.environ["NEURON_RT_VISIBLE_CORES"] = cores_range

        consumed_cores_list = [i for i in range(cores_per_instance)]
"""
    if is_inf2:
        init_impl += """
        compiled_model = os.path.join(args['model_repository'], compiled_model)
        self.model_neuron = torch.jit.load(compiled_model)
"""
    else:
        init_impl += """
        self.model_neuron = torch.neuron.DataParallel(
        torch.jit.load(compiled_model), device_ids=consumed_cores_list)
"""
    init_impl += """
        self.model_neuron.num_workers = num_threads
"""
    return init_impl


def get_tensorflow_execute_impl(disable_batch_requests_to_neuron):
    exec_impl = '''
    def _one_thread(self, pred, model_feed_dict):
        result = pred(model_feed_dict)
        return result

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
'''
    if disable_batch_requests_to_neuron:
        exec_impl += """
        responses = []
        num_threads = len(self.pred_list)
        model_feed_dict_list = [{} for _ in range(num_threads)]
        for request in requests:
            inputs = []
            for i in range(len(self.input_list)):
                name, dt, shape = self.input_list[i]
                tensor = pb_utils.get_input_tensor_by_name(request,
                                                           name).as_numpy()
                split_tensor = [None] * num_threads
                for split_index in range(num_threads):
                    model_feed_dict_list[split_index][name] = np.array_split(
                        tensor, num_threads, axis=0)[split_index]
            executor = futures.ThreadPoolExecutor(max_workers=num_threads)
            running = {
                executor.submit(self._one_thread, self.pred_list[idx],
                                model_feed_dict_list[idx]): idx
                for idx in range(num_threads)
            }
            results = [None] * num_threads
            for future in futures.as_completed(running):
                idx = running[future]
                results[idx] = future.result()
            output_tensors = []
            for i in range(len(self.output_list)):
                name, dt, shape = self.output_list[i]
                out_list = [None] * num_threads
                for idx in range(num_threads):
                    out_list[idx] = results[idx][name]
                full_tensor = out_list[0]
                for idx in range(num_threads - 1):
                    full_tensor = np.concatenate(
                        (full_tensor, out_list[idx + 1]), axis=0)
                output_tensor = pb_utils.Tensor(
                    name,
                    full_tensor.astype(pb_utils.triton_string_to_numpy(dt)))
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)
        return responses
"""
    else:
        exec_impl += """
        responses = []
        num_threads = len(self.pred_list)
        model_feed_dict_list = [{} for _ in range(num_threads)]
        num_requests = len(requests)
        request_batch_sizes = []
        inputs = []
        for i in range(len(self.input_list)):
            name, dt, shape = self.input_list[i]
            first_tensor = pb_utils.get_input_tensor_by_name(requests[0], name).as_numpy()
            request_batch_sizes.append(np.size(first_tensor, axis=0))
            batched_tensor = first_tensor
            for j in range(1, num_requests):
                tensor = pb_utils.get_input_tensor_by_name(requests[j],
                                                            name).as_numpy()
                request_batch_sizes.append(request_batch_sizes[-1] + np.size(tensor, axis=0))
                batched_tensor = np.concatenate((batched_tensor, tensor), axis=0)
            split_tensor = [None] * num_threads
            for split_index in range(num_threads):
                model_feed_dict_list[split_index][name] = np.array_split(
                    batched_tensor, num_threads, axis=0)[split_index]

        executor = futures.ThreadPoolExecutor(max_workers=num_threads)
        running = {
            executor.submit(self._one_thread, self.pred_list[idx],
                            model_feed_dict_list[idx]): idx
            for idx in range(num_threads)
        }

        results = [None] * num_threads
        for future in futures.as_completed(running):
            idx = running[future]
            results[idx] = future.result()

        chuncky_tensors = []
        for i in range(len(self.output_list)):
            name, dt, shape = self.output_list[i]
            out_list = [None] * num_threads
            for idx in range(num_threads):
                out_list[idx] = results[idx][name]
            full_tensor = out_list[0]
            for idx in range(num_threads - 1):
                full_tensor = np.concatenate(
                    (full_tensor, out_list[idx + 1]), axis=0)
            chuncky_tensors.append(np.split(full_tensor, request_batch_sizes, axis=0))

        for i in range(num_requests):
            output_tensors = []
            for j in range(len(self.output_list)):
                name, dt, shape = self.output_list[j]
                tensor = chuncky_tensors[j][i]
                output_tensor = pb_utils.Tensor(
                    name,
                    tensor.astype(pb_utils.triton_string_to_numpy(dt)))
                output_tensors.append(output_tensor)

            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)

        return responses
"""
    return exec_impl


def get_pytorch_execute_impl(disable_batch_requests_to_neuron):
    exec_impl = '''
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
'''
    if disable_batch_requests_to_neuron:
        exec_impl += """
        responses = []
        for request in requests:
            inputs = []
            for i in range(len(self.input_dict)):
                name, dt, shape = self.input_dict[i]
                tensor = torch.as_tensor(pb_utils.get_input_tensor_by_name(request,
                                                           name).as_numpy())
                inputs.append(tensor)
            results = self.model_neuron(*inputs)
            output_tensors = []
            for i in self.output_dict.keys():
                name, dt, shape = self.output_dict[i]
                result = results[i] if isinstance(results, tuple) else results
                output_tensor = pb_utils.Tensor(
                    name, result.numpy().astype(
                        pb_utils.triton_string_to_numpy(dt)))
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)
        return responses
"""
    else:
        exec_impl += """
        responses = []
        inputs = []
        num_requests = len(requests)
        request_batch_sizes = []
        for i in self.input_dict.keys():
            name, dt, shape = self.input_dict[i]
            first_tensor = torch.as_tensor(pb_utils.get_input_tensor_by_name(requests[0],
                                                            name).as_numpy())
            request_batch_sizes.append(first_tensor.size(dim=0))
            batched_tensor = first_tensor
            for j in range(1, num_requests):
                tensor = torch.as_tensor(pb_utils.get_input_tensor_by_name(requests[j],
                                                            name).as_numpy())
                request_batch_sizes.append(request_batch_sizes[-1] + tensor.size(dim=0))
                batched_tensor = torch.cat((batched_tensor, tensor), dim=0)
            inputs.append(batched_tensor)

        batched_results = self.model_neuron(*inputs)
        chunky_batched_results = []
        for i in self.output_dict.keys():
            batch = batched_results[i] if isinstance(batched_results, tuple) else batched_results
            chunky_batched_results.append(torch.tensor_split(batch, request_batch_sizes, dim=0))
        for i in range(num_requests):
            output_tensors = []
            for j in self.output_dict.keys():
                name, dt, shape = self.output_dict[j]
                result = chunky_batched_results[j][i]
                output_tensor = pb_utils.Tensor(
                    name, result.numpy().astype(
                        pb_utils.triton_string_to_numpy(dt)))
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)

        return responses
"""
    return exec_impl


def get_finalize_impl():
    finalize_impl = '''
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

'''
    return finalize_impl


def get_triton_python_model_impl(
    using_tensorflow_model, disable_batch_requests_to_neuron, is_inf2=False
):
    triton_pmi = '''
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    '''

    if using_tensorflow_model:
        triton_pmi += get_tensorflow_initialize_impl(is_inf2)
        triton_pmi += get_tensorflow_execute_impl(disable_batch_requests_to_neuron)
    else:
        triton_pmi += get_pytorch_initialize_impl(is_inf2)
        triton_pmi += get_pytorch_execute_impl(disable_batch_requests_to_neuron)

    triton_pmi += get_finalize_impl()

    return triton_pmi


def create_model_file(
    using_tensorflow_model, disable_batch_requests_to_neuron, is_inf2=False
):
    triton_model = get_model_license()
    triton_model += """
import json
import numpy as np
import os
import sys
import triton_python_backend_utils as pb_utils
"""

    if using_tensorflow_model:
        triton_model += """
import tensorflow as tf
from concurrent import futures
"""
    else:
        triton_model += """
import torch
    """
        if not is_inf2:
            triton_model += """
import torch.neuron
        """
        else:
            triton_model += """
import torch_neuronx
"""
    triton_model += get_triton_python_model_impl(
        using_tensorflow_model, disable_batch_requests_to_neuron, is_inf2
    )
    return triton_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inf2",
        required=False,
        default=False,
        action="store_true",
        help="Specify whether the model should be generate for inf2 or inf1, default is inf1",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow"],
        help="""The type of the compiled model. Currently,
                    only supports \"pytorch\" and \"tensorflow\".""",
    )
    parser.add_argument(
        "--model_version", type=int, default=1, help="The version of the model"
    )
    parser.add_argument(
        "--enable_dynamic_batching",
        action="store_true",
        help="""Enable dynamic batching. Please see model configuration
        documentation for details:
        https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher""",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=0,
        help="""The maximum batch size for the model being generated.
        Please see model configuration documentation for details:
        https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#maximum-batch-size""",
    )
    parser.add_argument(
        "--preferred_batch_size",
        type=int,
        help="""The preferred batch size. Should be multiples
        of cores available to ensure proper utilization of
        neuron cores.
        This flag is ignored if --enable_dynamic_batching is
        not specified. Please see model configuration
        documentation for details:
        https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#preferred-batch-sizes""",
    )
    parser.add_argument(
        "--max_queue_delay_microseconds",
        type=int,
        help="""Max queue delay time(ms) for dynamic batching.
        This flag is ignored if --enable_dynamic_batching is not specified.
        Please see model configuration documentation for details:
        https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#delayed-batching""",
    )
    parser.add_argument(
        "--disable_batch_requests_to_neuron",
        action="store_true",
        help="""Send each request separately to neuron if enabled.
                         If not specified, then requests are combined and sent to
                         neuron as a single batch""",
    )
    parser.add_argument(
        "--tag_set",
        type=str,
        default="serve",
        help="""The tag set to use for the TF model.
                        This option is ignored if `--model_type` is
                        not \"tensorflow\". Default value is \'serve\'.""",
    )
    parser.add_argument(
        "--signature_def_key",
        type=str,
        default="serving_default",
        help="""The signature def key to use for the TF
                        model. This option is ignored if `--model_type`
                        is not \"tensorflow\". Default value
                        is \'serving_default\'.""",
    )
    parser.add_argument(
        "--compiled_model",
        type=str,
        required=True,
        help="Fullpath to the compiled model",
    )
    parser.add_argument(
        "--triton_input",
        type=str,
        action="append",
        nargs="*",
        help="""The name, datatype and shape of the model input in
        format <input_name>,<triton_datatype>,<shape>. This
        option can be provided multiple times for multiple
        inputs. For example, to provide a FP16 input with
        shape [1,384] specify the following: INPUT0,FP16,1x384.
        This option is not required when using tensorflow model""",
    )
    parser.add_argument(
        "--triton_output",
        type=str,
        action="append",
        nargs="*",
        help="""The name, datatype and shape of the model output in
        format <output_name>,<triton_datatype>,<shape>. This
        option can be provided multiple times for multiple
        outputs. For example, to provide a FP16 output with
        shape [1,384] specify the following: OUTPUT0,FP16,1x384.
        This option is not required when using tensorflow model""",
    )
    parser.add_argument(
        "--neuron_core_range",
        type=str,
        required=True,
        help="""The range of neuron core indices
                        where the model needs to be loaded. The
                        range should be specified in format
                        <start_idx>:<end_idx>. For example to
                        load model on neuron cores (0-7), specify
                        the following: 0:7. NOTE: when using
                        multiple triton model instances the neuron
                        cores will get equally distributed. Assuming
                        the instance count is 4, Instance0 will get
                        loaded on cores 0:1, Instance1 will get loaded
                        on cores 2:3, Instance2 will get loaded on
                        cores 4:5 and Instance 3 will get loaded on
                        cores 6:7""",
    )
    parser.add_argument(
        "--threads_per_core",
        type=int,
        default=1,
        help="The number of threads per neuron core.",
    )
    parser.add_argument(
        "--triton_model_instance_count",
        type=int,
        default=1,
        help="The number of triton model instances.",
    )
    parser.add_argument(
        "--triton_model_dir",
        type=str,
        required=True,
        help="""Path to the triton model
                        directory where script will generate
                        config.pbtxt and model.py""",
    )
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception("Unrecognized options: {}".format(unparsed))

    if FLAGS.model_type == "tensorflow":
        is_tensorflow_model = True
    elif FLAGS.model_type == "pytorch":
        is_tensorflow_model = False

    print(
        """Triton Dynamic Batching is enabled: {},
        preferred_batch_size: {} and max_batch_size: {}
        with max_queue_delay_microseconds: {}.
        Batch requests to neruon are disabled: {}""".format(
            FLAGS.enable_dynamic_batching,
            FLAGS.preferred_batch_size,
            FLAGS.max_batch_size,
            FLAGS.max_queue_delay_microseconds,
            FLAGS.disable_batch_requests_to_neuron,
        )
    )

    if not is_tensorflow_model or (
        FLAGS.triton_input != None and FLAGS.triton_output != None
    ):
        inputs = parse_io_tensors(FLAGS.triton_input)
        outputs = parse_io_tensors(FLAGS.triton_output)
    else:
        inputs, outputs = parse_tf_tensors(
            FLAGS.compiled_model, FLAGS.tag_set, FLAGS.signature_def_key
        )

    nc_start_idx, nc_end_idx = [int(i) for i in FLAGS.neuron_core_range.split(":")]

    model_version_dir = FLAGS.triton_model_dir + "/" + str(FLAGS.model_version)
    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    model_name = os.path.basename(FLAGS.triton_model_dir)
    mc = create_modelconfig(
        model_name,
        FLAGS.max_batch_size,
        inputs,
        outputs,
        FLAGS.compiled_model,
        nc_start_idx,
        nc_end_idx,
        FLAGS.threads_per_core,
        FLAGS.triton_model_instance_count,
        FLAGS.enable_dynamic_batching,
        FLAGS.preferred_batch_size,
        FLAGS.max_queue_delay_microseconds,
    )
    with open(FLAGS.triton_model_dir + "/config.pbtxt", "w") as config_file:
        config_file.write(mc)

    is_inf2 = FLAGS.inf2

    mf = create_model_file(
        is_tensorflow_model, FLAGS.disable_batch_requests_to_neuron, is_inf2
    )
    with open(FLAGS.triton_model_dir + "/1/model.py", "w") as model_file:
        model_file.write(mf)
