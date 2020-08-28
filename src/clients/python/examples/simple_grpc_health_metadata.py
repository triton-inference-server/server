#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = 'simple'

    # Health
    if not triton_client.is_server_live(headers={'test': '1', 'dummy': '2'}):
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(model_name):
        print("FAILED : is_model_ready")
        sys.exit(1)

    # Metadata
    metadata = triton_client.get_server_metadata()
    if not (metadata.name == 'triton'):
        print("FAILED : get_server_metadata")
        sys.exit(1)
    print(metadata)

    metadata = triton_client.get_model_metadata(model_name,
                                                headers={
                                                    'test': '1',
                                                    'dummy': '2'
                                                })
    if not (metadata.name == model_name):
        print("FAILED : get_model_metadata")
        sys.exit(1)
    print(metadata)

    # Passing incorrect model name
    try:
        metadata = triton_client.get_model_metadata("wrong_model_name")
    except InferenceServerException as ex:
        if "Request for unknown model" not in ex.message():
            print("FAILED : get_model_metadata wrong_model_name")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
    else:
        print("FAILED : get_model_metadata wrong_model_name")
        sys.exit(1)

    # Configuration
    config = triton_client.get_model_config(model_name)
    if not (config.config.name == model_name):
        print("FAILED: get_model_config")
        sys.exit(1)
