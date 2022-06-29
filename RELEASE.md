<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Release Notes for 2.23.0

## New Freatures and Improvements

* Auto-generated model configuration enables 
  [dynamic batching](https://github.com/triton-inference-server/server/blob/r22.06/docs/model_configuration.md#default-max-batch-size-and-dynamic-batcher) 
  in supported models by default.

* Python backend models now support 
  [auto-generated model configuration](https://github.com/triton-inference-server/python_backend/tree/r22.06#auto_complete_config).

* [Decoupled API](https://github.com/triton-inference-server/server/blob/r22.06/docs/decoupled_models.md#python-model-using-python-backend) 
  support in Python Backend model is out of beta.

* Updated I/O tensors 
  [naming convention](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#special-conventions-for-pytorch-backend) 
  for serving TorchScript models via PyTorch backend.

* Improvements to Perf Analyzer stability and profiling logic.

* Refer to the 22.06 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 22.06 inference server container is based.


## Known Issues

* Perf Analyzer stability criteria has been changed which may result in 
  reporting instability for scenarios that were previously considered stable. 
  This change has been made to improve the accuracy of Perf Analyzer results. 
  If you observe this message, it can be resolved by increasing the 
  `--measurement-interval` in the time windows mode or 
  `--measurement-request-count` in the count windows mode.

* 22.06 is the last release that defaults to 
  [TensorFlow version 1](https://github.com/triton-inference-server/tensorflow_backend/tree/r22.06#--backend-configtensorflowversionint). 
  From 22.07 onwards Triton will change the default TensorFlow version to 2.X.

* Triton PIP wheels for ARM SBSA are not available from PyPI and pip will 
  install an incorrect Jetson version of Triton for Arm SBSA. 
  
  The correct wheel file can be pulled directly from the Arm SBSA SDK image and 
  manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. 
  
  Refer to issue [pytorch#66930](https://github.com/pytorch/pytorch/issues/66930) 
  for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.

* Running a PyTorch TorchScript model using the PyTorch backend, where multiple 
  instances of a model are configured can lead to a slowdown in model execution 
  due to the following PyTorch issue: 
  [pytorch#27902](https://github.com/pytorch/pytorch/issues/27902).

* Starting from 22.02, the Triton container, which uses the 22.02 or above 
  PyTorch container, will report an error during model loading in the PyTorch 
  backend when using scripted models that were exported in the legacy format 
  (using our 19.09 or previous PyTorch NGC containers corresponding to 
  PyTorch 1.2.0 or previous releases). 
  
  To load the model successfully in Triton, you need to export the model again 
  by using a recent version of PyTorch.
