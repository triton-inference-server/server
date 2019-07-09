# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required (VERSION 3.5)

#
# TensorFlow backends
#
# tensorflow_backend_tf is copied into and built along with TensorFlow
# to ensure that it uses the same protobuf as TensorFlow, so not
# included in these sources.
#
set(
  TENSORFLOW_SRCS
  autofill.cc
  base_backend.cc
  graphdef_backend.cc
  graphdef_backend_factory.cc
  savedmodel_backend.cc
  savedmodel_backend_factory.cc
  tf_utils.cc
  tf_virtual_device.cc
)

set(
  TENSORFLOW_HDRS
  autofill.h
  base_backend.h
  graphdef_backend.h
  graphdef_backend_factory.h
  savedmodel_backend.h
  savedmodel_backend_factory.h
  tensorflow_backend_tf.h
  tf_utils.h
  tf_virtual_device.h
)

add_library(
  tensorflow-backend-library EXCLUDE_FROM_ALL OBJECT
  ${TENSORFLOW_SRCS} ${TENSORFLOW_HDRS}
)
add_dependencies(tensorflow-backend-library proto-library)
if(${TRTIS_ENABLE_GPU})
  target_include_directories(tensorflow-backend-library PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRTIS_ENABLE_GPU
