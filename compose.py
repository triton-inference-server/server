#!/usr/bin/env python3
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
TRITON_VERSION_MAP = {
    '2.11.0dev': (
        '21.06dev',  # triton container
        '21.05')  # upstream container     
}

EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['tensorrt', 'ensemble']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali', 'pytorch',
    'openvino', 'fil'
]
EXAMPLE_REPOAGENTS = ['checksum']
FLAGS = None

DEPENDENCY_MAP = {'python': ('python3-pip'), 'pytorch': ('python3-dev')}


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
    print('error: {}'.format(msg), file=sys.stderr)
    sys.exit(1)
    

def fail_if(p, msg):
    if p:
        fail_if(True, msg)
        

##### create base image for gpu
def start_gpu_dockerfile(ddir, dockerfile_name):
    # Set enviroment variables
    df = '''
#
# Multistage build.
#
FROM nvcr.io/nvidia/tritonserver:{}-py3 as full
FROM nvcr.io/nvidia/tritonserver:{}-py3-min
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
LABEL com.nvidia.tritonserver.version="${{TRITON_SERVER_VERSION}}"
ENV PATH /opt/tritonserver/bin:${{PATH}}
'''.format(FLAGS.upstream_container_version, FLAGS.upstream_container_version,
           FLAGS.version, FLAGS.container_version)

    # Copy over files
    df += '''
# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true && \
    if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
        useradd $TRITON_SERVER_USER; \
    fi && \
    [ `id -u $TRITON_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
COPY --chown=1000:1000 --from=full /opt/tritonserver/LICENSE .
COPY --chown=1000:1000 --from=full /opt/tritonserver/TRITON_VERSION .
COPY --chown=1000:1000 --from=full /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf .
COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include/triton/core include/triton/core
# Top-level include/core not copied so --chown does not set it correctly,
# so explicit set on all of include
RUN chown -R triton-server:triton-server include
'''
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


### add additional backends needed
def add_requested_backends(ddir, dockerfile_name, backends):
    df = "# Copying over backends \n"
    for backend in backends:
        if (backend.lower()
                in (EXAMPLE_BACKENDS + CORE_BACKENDS + NONCORE_BACKENDS)):
            df += '''COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/{} /opt/tritonserver/backends/{}    
'''.format(backend, backend)
        else:
            log("Cannot create container from unsupported backend: " + backend)

    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def add_requested_repoagents(ddir, dockerfile_name, repoagents):
    df = "#  Copying over repoagents \n"
    for ra in repoagents:
        if (ra in EXAMPLE_REPOAGENTS):
            df += '''COPY --chown=1000:1000 --from=full /opt/tritonserver/repoagents/{} /opt/tritonserver/repoagents/{}    
'''.format(ra, ra)
        else:
            log("Cannot create container from unsupported repoagent: " + ra)

    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)



# Install dependencies and run entrypoint script
def end_gpu_dockerfile(ddir, dockerfile_name):
    # TODO: remove libnvidia-ml-dev after 21.06 is launched
    df = '''
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libnvidia-ml-dev \
            libre2-dev \
            libb64-dev \
            pciutils && \
    rm -rf /var/lib/apt/lists/*
'''

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
    df += '''
# Add entrypoint script
WORKDIR /opt/tritonserver
COPY --chown=1000:1000 --from=full /opt/tritonserver/nvidia_entrypoint.sh .
ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]
'''
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


### Create container with docker build
def build_docker_container(ddir, dockerfile_name, container_name):
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
    parser.add_argument('--build-name',
                        type=str,
                        required=False,
                        help='Build name. Default is "tritonserver".')
    parser.add_argument(
        '--work-dir',
        type=str,
        required=False,
        help=
        'Generated dockerfiles are placed here. Default to current directory.'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=False,
        help=
        'The Triton version. If not specified defaults to the value in the TRITON_VERSION file.'
    )
    parser.add_argument(
        '--container-version',
        type=str,
        required=False,
        help=
        'The Triton container version to build. If not specified the container version will be chosen automatically based on --version value.'
    )
    parser.add_argument(
        '--upstream-container-version',
        type=str,
        required=False,
        help=
        'The upstream container version to use for the build. If not specified the upstream container version will be chosen automatically based on --version value.'
    )
    parser.add_argument(
        '--enable-gpu',
        type=str,
        required=True,
        help=
        'Flag to enable gpus for the build. Must be specified since only-cpu is not supported'
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
    
    if FLAGS.work_dir is None:
        FLAGS.work_dir = "$(PWD)"
    if FLAGS.build_name is None:
        FLAGS.build_name = "tritonserver"
    # Determine the versions. Start with Triton version, if --version
    # is not explicitly specified read from TRITON_VERSION file.
    if FLAGS.version is None:
        with open('TRITON_VERSION', "r") as vfile:
            FLAGS.version = vfile.readline().strip()
    if FLAGS.container_version is None:
        if FLAGS.version not in TRITON_VERSION_MAP:
            fail('container version not known for {}'.format(FLAGS.version))
        FLAGS.container_version = TRITON_VERSION_MAP[FLAGS.version][0]
    if FLAGS.upstream_container_version is None:
        if FLAGS.version not in TRITON_VERSION_MAP:
            fail('upstream container version not known for {}'.format(
                FLAGS.version))
        FLAGS.upstream_container_version = TRITON_VERSION_MAP[FLAGS.version][1]

    dockerfile_name = 'Dockerfile.compose'

    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []

    start_gpu_dockerfile(FLAGS.work_dir, dockerfile_name)
    add_requested_backends(FLAGS.work_dir, dockerfile_name, FLAGS.backend)
    add_requested_repoagents(FLAGS.work_dir, dockerfile_name, FLAGS.repoagent)
    end_gpu_dockerfile(FLAGS.work_dir, dockerfile_name)
    build_docker_container(FLAGS.work_dir, dockerfile_name, FLAGS.build_name)
