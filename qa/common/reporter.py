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
import json
import os
import requests
import socket
import sys

FLAGS = None

ENVS = [ "CUDA_DRIVER_VERSION", "CUDA_VERSION",
         "TENSORRT_SERVER_VERSION", "NVIDIA_TENSORRT_SERVER_VERSION",
         "TRT_VERSION", "CUDNN_VERSION", "CUBLAS_VERSION",
         "BENCHMARK_REPO_BRANCH", "BENCHMARK_REPO_COMMIT",
         "BENCHMARK_CLUSTER", "BENCHMARK_GPU_COUNT"]

def annotate(datas):
    # Add all interesting envvar values
    for data in datas:
        for env in ENVS:
            if env in os.environ:
                val = os.environ[env]
                data['s_' + env.lower()] = val

    # Add this system's name
    data['s_benchmark_system'] = socket.gethostname()

def post_to_url(url, data):
    headers = {'Content-Type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=data, headers=headers)
    r.raise_for_status()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='Output filename')
    parser.add_argument('-u', '--url', type=str, required=False,
                        help='Post results to a URL')
    parser.add_argument('file', type=argparse.FileType('r'))
    FLAGS = parser.parse_args()

    data = json.loads(FLAGS.file.read())
    annotate(data)
    if FLAGS.verbose:
        print(json.dumps(data, sort_keys=True, indent=2))

    if FLAGS.output is not None:
        with open(FLAGS.output, "w") as f:
            f.write(json.dumps(data))
            f.write("\n")

    if FLAGS.url is not None:
        post_to_url(FLAGS.url, json.dumps(data))
