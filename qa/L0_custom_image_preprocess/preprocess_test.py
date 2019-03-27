#!/usr/bin/python

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

import argparse
import numpy as np
import os
from builtins import range
from tensorrtserver.api import *

FLAGS = None

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                       help='Enable verbose output')
   parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                       help='Inference server URL. Default is localhost:8000.')
   parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                       help='Protocol ("http"/"grpc") used to ' +
                       'communicate with inference service. Default is "http".')
   parser.add_argument('-p', '--preprocessed_filename', type=str, required=True, default=None,
                        help='Preprocessed image.')
   parser.add_argument('image_filename', type=str, nargs='?', default=None,
                        help='Input image.')

   FLAGS = parser.parse_args()
   protocol = ProtocolType.from_str(FLAGS.protocol)

   model_name = "image_preprocess_nhwc_224x224x3"
   model_version = -1
   batch_size = 1

   # Create the inference context for the model.
   ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

   # Input tensor will be raw content from image file
   image_path = FLAGS.image_filename
   with open(image_path, "rb") as fd:
      input_data = np.array([fd.read()], dtype=bytes)

   expected_res_path = FLAGS.preprocessed_filename
   with open(expected_res_path, "r") as fd:
      expected_data = np.fromfile(fd, np.float32)

   result = ctx.run({ 'INPUT' : (input_data,) },
                    { 'OUTPUT' : InferContext.ResultFormat.RAW },
                    batch_size)

   if "OUTPUT" not in result:
      print("error: expected 'OUTPUT'")
      sys.exit(1)

   if len(result["OUTPUT"]) != 1:
      print("error: expected 1 output result, got {}".format(len(result["OUTPUT"])))
      sys.exit(1)

   res_data = result["OUTPUT"][0].reshape([-1])
   if not np.array_equal(res_data, expected_data):
      print("error: result does not match expected data")
      sys.exit(1)
