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
import os
import os.path
import sys
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status
import tensorrtserver.api.model_config_pb2 as model_config
from google.protobuf import text_format

FLAGS = None

def _get_server_status(url="localhost:8000", protocol=ProtocolType.HTTP, model_name=None):
   ctx = ServerStatusContext(url, protocol, model_name, False)
   return ctx.get_server_status()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected_dir', type=str, required=True,
                        help='Directory containing expected output files')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')
    FLAGS, unparsed = parser.parse_known_args()

    status = _get_server_status(model_name=FLAGS.model)
    status_config = status.model_status[FLAGS.model].config

    nonmatch = list()

    expected_files = [f for f in os.listdir(FLAGS.expected_dir)
                      if (os.path.isfile(os.path.join(FLAGS.expected_dir, f)) and
                          (f.startswith("expected")))]
    for efile in expected_files:
       with open(os.path.join(FLAGS.expected_dir, efile)) as f:
          config = text_format.Parse(f.read(), model_config.ModelConfig())

       if config == status_config:
          sys.exit(0)

       nonmatch.append(config)

    print("Model config doesn't match any expected output:")
    print("Model config:")
    print(status_config)
    for nm in nonmatch:
       print("Non-matching:")
       print(nm)

    sys.exit(1)
