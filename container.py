#!/usr/bin/env python3
# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import sys
import docker

### Global variables
EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['tensorrt', 'ensemble']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali', 'pytorch',
    'openvino', 'fil'
]
EXAMPLE_REPOAGENTS = ['checksum']
FLAGS = None

DEPENDENCY_MAP = {
    'python': 
    ('python3-pip'),
    'pytorch':
    ('python3-dev')
}
#### helper functions
def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print('<failed to log>', file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    fail_if(True, msg)


def fail_if(p, msg):
    if p:
        print('error: {}'.format(msg), file=sys.stderr)
        sys.exit(1)


##### create base image
def make_dockerfile(ddir, dockerfile_name, container_ver):
    df = '''
FROM nvcr.io/nvidia/tritonserver:{}-py3 as full
FROM nvcr.io/nvidia/tritonserver:{}-py3-min
COPY --from=full /opt/tritonserver/bin /opt/tritonserver/bin
COPY --from=full /opt/tritonserver/lib /opt/tritonserver/lib  
'''.format(container_ver, container_ver)

    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


### add additional backends needed
def add_requested_backends(ddir, dockerfile_name, backends):
    df = ""
    for backend in backends:
        if (backend.lower()
                in (EXAMPLE_BACKENDS + CORE_BACKENDS + NONCORE_BACKENDS)):
            df += '''COPY --from=full /opt/tritonserver/backends/{} /opt/tritonserver/backends/{}    
'''.format(backend, backend)
        else:
            log("Cannot create container from unsupported backend: " + backend)

    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def add_requested_repoagents(ddir, dockerfile_name, repoagents):
    df = ""
    for ra in repoagents:
        if (ra in EXAMPLE_REPOAGENTS):
            df += '''COPY --from=full /opt/tritonserver/repoagents/{} /opt/tritonserver/repoagents/{}    
'''.format(ra, ra)
        else:
            log("Cannot create container from unsupported repoagent: " + ra)

    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)

def append_workdir(ddir, dockerfile_name, workdir_path):
    df = '''WORKDIR {}'''.format(workdir_path)
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)

def install_dependencies(ddir, dockerfile_name):
    df = '''
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libre2-dev \
            libb64-dev \
            libnvidia-ml-dev
    rm -rf /var/lib/apt/lists/*
'''
# TODO: remove libnvidia-ml-dev after 21.06 is launched
    # Add dependencies needed for python backend
    if 'python' in FLAGS.backend:
        df += '''
# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         python3 \
         python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade grpcio-tools grpcio-channelz numpy && \
    rm -rf /var/lib/apt/lists/*
'''
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)

### create build container
def build_docker_container(ddir, dockerfile_name, container_name):
    # Before attempting to run the new image, make sure any
    # previous <container_name> container is removed.
    client = docker.from_env(timeout=3600)
    try:
        existing = client.containers.get(container_name)
        existing.remove(force=True)
    except docker.errors.NotFound:
        pass  # ignore

    p = subprocess.Popen(['docker', 'build', '-t', container_name, '-f', \
        os.path.join(ddir, dockerfile_name), '.'])
    p.wait()
    fail_if(p.returncode != 0, 'docker build {} failed'.format(container_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument('-q',
                          '--quiet',
                          action="store_true",
                          required=False,
                          help='Disable console output.')
    group_qv.add_argument('-v',
                          '--verbose',
                          action="store_true",
                          required=False,
                          help='Enable verbose output.')
    parser.add_argument(
        '--build-dir',
        type=str,
        required=True,
        help=
        'Build directory. All repo clones and builds will be performed in this directory.'
    )
    parser.add_argument(
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>. To have multiple backends built, specify --backend <foo> --backend <bar>'
    )
    parser.add_argument(
        '--repoagent',
        action='append',
        required=False,
        help='Include specified repo agent in build as <repoagent-name>')
    FLAGS = parser.parse_args()
    dockerfile_name = 'Dockerfile.buildbase'
    container_name = "tritonserver_build"
    workdir_path="/opt/tritonserver"
    server_version = 21.05
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []
    print(FLAGS.backend)

    make_dockerfile(FLAGS.build_dir, dockerfile_name, server_version)
    add_requested_backends(FLAGS.build_dir, dockerfile_name, FLAGS.backend)
    add_requested_repoagents(FLAGS.build_dir, dockerfile_name, FLAGS.repoagent)
    append_workdir(FLAGS.build_dir, dockerfile_name, workdir_path)
    install_dependencies(FLAGS.build_dir, dockerfile_name)
    build_docker_container(FLAGS.build_dir, dockerfile_name, container_name)
