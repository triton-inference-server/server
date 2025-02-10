..
.. Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions
.. are met:
..  * Redistributions of source code must retain the above copyright
..    notice, this list of conditions and the following disclaimer.
..  * Redistributions in binary form must reproduce the above copyright
..    notice, this list of conditions and the following disclaimer in the
..    documentation and/or other materials provided with the distribution.
..  * Neither the name of NVIDIA CORPORATION nor the names of its
..    contributors may be used to endorse or promote products derived
..    from this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
.. EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
.. PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
.. OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
.. (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
.. OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

####
Extensions
####

To fully enable all capabilities
Triton also implements `HTTP/REST and GRPC
extensions <https://github.com/triton-inference-server/server/tree/main/docs/protocol>`__
to the KServe inference protocol.

.. toctree::
   :maxdepth: 1
   :hidden:

   Binary tensor data extension <../protocol/extension_binary_data.md>
   Classification extension <../protocol/extension_classification.md>
   Schedule policy extension <../protocol/extension_schedule_policy.md>
   Sequence extension <../protocol/extension_sequence.md>
   Shared-memory extension <../protocol/extension_shared_memory.md>
   Model configuration extension <../protocol/extension_model_configuration.md>
   Model repository extension <../protocol/extension_model_repository.md>
   Statistics extension <../protocol/extension_statistics.md>
   Trace extension <../protocol/extension_trace.md>
   Logging extension <../protocol/extension_logging.md>
   Parameters extension <../protocol/extension_parameters.md>