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

   FLAGS = parser.parse_args()
   protocol = ProtocolType.from_str(FLAGS.protocol)

   model_name = "param"
   model_version = -1
   batch_size = 1

   # Create the inference context for the model.
   ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

   # Input tensor can be any size int32 vector...
   input_data = np.zeros(shape=1, dtype=np.int32)

   result = ctx.run({ 'INPUT' : (input_data,) },
                    { 'OUTPUT' : InferContext.ResultFormat.RAW },
                    batch_size)
   print(result)

   if "OUTPUT" not in result:
      print("error: expected 'OUTPUT'");
      sys.exit(1);

   if len(result["OUTPUT"]) != 1:
      print("error: expected 1 output result, got {}".format(len(result["OUTPUT"])));
      sys.exit(1);

   params = result["OUTPUT"][0]
   if params.size != 5:
      print("error: expected 5 output strings, got {}".format(params.size));
      sys.exit(1);

   p0 = params[0].decode("utf-8")
   if not p0.startswith("INPUT=0"):
      print("error: expected INPUT=0 string, got {}".format(p0));
      sys.exit(1);

   p1 = params[1].decode("utf-8")
   if not p1.startswith("server_0="):
      print("error: expected server_0 parameter, got {}".format(p1));
      sys.exit(1);

   p2 = params[2].decode("utf-8")
   if not p2.startswith("server_1="):
      print("error: expected server_1 parameter, got {}".format(p2));
      sys.exit(1);
   if not p2.endswith("L0_custom_param/models"):
      print("error: expected model-repository to end with L0_custom_param/models, got {}".format(p2));
      sys.exit(1);

   # configuration param values can be returned in any order.
   p3 = params[3].decode("utf-8")
   p4 = params[4].decode("utf-8")
   if p3.startswith("param1"):
      p3, p4 = p4, p3

   if p3 != "param0=value0":
      print("error: expected param0=value0, got {}".format(p3));
      sys.exit(1);

   if p4 != "param1=value1":
      print("error: expected param1=value1, got {}".format(p4));
      sys.exit(1);
