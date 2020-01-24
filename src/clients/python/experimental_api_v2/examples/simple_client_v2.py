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

import sys
import argparse
from tensorrtserverV2.api import *

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
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')

    FLAGS = parser.parse_args()
    try:
        TRTISClient = InferenceServerHTTPClient(FLAGS.url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = 'savedmodel_float32_float32_float32'
    model_name_2 = 'savedmodel_object_object_object'

    ### Model Control API ###
    TRTISClient.load(model_name)
    if TRTISClient.is_model_ready(model_name, 1):
        print("PASS: load")

    TRTISClient.load(model_name_2)
    if TRTISClient.is_model_ready(model_name_2, 1):
        print("PASS: load_2")

    ### Health API ###
    if TRTISClient.is_server_live():
        print("PASS: is_server_live")

    if TRTISClient.is_server_ready():
        print("PASS: is_server_ready")

    if TRTISClient.is_model_ready(model_name, 1):
        print("PASS: is_model_ready")

    ### Status API ###
    status = TRTISClient.get_server_status()
    if 'id' in status.keys() and status['id'] == 'inference:0':
        print("PASS: get_server_status")

    status = TRTISClient.get_model_status(model_name)
    if 'modelStatus' in status.keys() and len(status['modelStatus']) == 1:
        print("PASS: get_model_status")

    # Passing incorrect model name
    try:
        status = TRTISClient.get_model_status("wrong_model_name")
    except InferenceServerException as ex:
        if "inference:0" == ex.server_id() and \
            "no status available for unknown model" in ex.message():
            print("PASS: detected wrong model")

    TRTISClient.unload(model_name_2)
    if not TRTISClient.is_model_ready(model_name_2, 1):
        print("PASS: unload")

    TRTISClient.close()
