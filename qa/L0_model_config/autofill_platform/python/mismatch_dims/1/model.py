# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(model_config):
        """`auto_complete_config` is called only once when the server is started
        with `--strict-model-config=false`. This function allows us to set
        `max_batch_size`, `input` and `output` properties of the model using
        `pb_utils.set_max_batch_size`, `pb_utils.set_input`, and
        `pb_utils.set_output`.
        Must return a `pb_utils.ModelConfig` object which contains the updated
        model configuration.
        Note: All the objects in this function will go out of scope after exiting.
        Should not store any objects in this function.

        Parameters
        ----------
        model_config : string
          A JSON string contains the model configuration.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the updated model configuration
        """
        auto_complete_model_config = pb_utils.ModelConfig()

        input0 = {
          'name' : 'INPUT0',
          'data_type' : 'TYPE_FP32',
          'dims' : [ 16 ]
        }
        input1 = {
          'name' : 'INPUT1',
          'data_type' : 'TYPE_FP32',
          'dims' : [ 16 ],
        }
        output0 = {
          'name' : 'OUTPUT0',
          'data_type' : 'TYPE_FP32',
          'dims' : [ 16 ]
        }
        output1 = {
          'name' : 'OUTPUT1',
          'data_type' : 'TYPE_FP32',
          'dims' : [ 16 ]
        }

        pb_utils.set_max_batch_size(auto_complete_model_config, 0)
        pb_utils.set_input(auto_complete_model_config, input0)
        pb_utils.set_input(auto_complete_model_config, input1)
        pb_utils.set_output(auto_complete_model_config, output0)
        pb_utils.set_output(auto_complete_model_config, output1)

        return auto_complete_model_config

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

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
        # self.model_config = model_config = json.loads(args['model_config'])
        self.model_config = model_config = json.loads('{"name":"add_sub","platform":"","backend":"python","version_policy":{"latest":{"num_versions":1}},"max_batch_size":0,"input":[],"output":[{"name":"OUTPUT0","data_type":"TYPE_FP32","dims":[4],"label_filename":"","is_shape_tensor":false},{"name":"OUTPUT1","data_type":"TYPE_FP32","dims":[4],"label_filename":"","is_shape_tensor":false}],"batch_input":[],"batch_output":[],"optimization":{"priority":"PRIORITY_DEFAULT","input_pinned_memory":{"enable":true},"output_pinned_memory":{"enable":true},"gather_kernel_buffer_threshold":0,"eager_batching":false},"instance_group":[{"name":"add_sub_0","kind":"KIND_CPU","count":1,"gpus":[],"secondary_devices":[],"profile":[],"passive":false,"host_policy":""}],"default_model_filename":"","cc_model_filenames":{},"metric_tags":{},"parameters":{},"model_warmup":[]}')

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT1")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

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

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            # Get INPUT1
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            out_0, out_1 = (in_0.as_numpy() + in_1.as_numpy(),
                            in_0.as_numpy() - in_1.as_numpy())

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1",
                                           out_1.astype(output1_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
