#!/usr/bin/env python3
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
import logging
import docker
import os.path
import multiprocessing
import pathlib
import shutil
import subprocess
import sys
import traceback
from distutils.dir_util import copy_tree

#
# Build Triton Inference Server.
#
EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['pytorch', 'tensorrt', 'custom', 'ensemble', 'caffe2']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali'
]
FLAGS = None

# Map from container version to corresponding component versions
# container-version -> (ort version, ort openvino version)
CONTAINER_VERSION_MAP = {
    '20.08': ('1.4.0', '2020.2'),
    '20.09': ('1.4.0', '2020.2')
}


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


def mkdir(path):
    log_verbose('mkdir: {}'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    log_verbose('rmdir: {}'.format(path))
    shutil.rmtree(path, ignore_errors=True)


def cpdir(src, dest):
    log_verbose('cpdir: {} -> {}'.format(src, dest))
    copy_tree(src, dest, preserve_symlinks=1)


def untar(targetdir, tarfile):
    log_verbose('untar {} into {}'.format(tarfile, targetdir))
    p = subprocess.Popen(['tar', '--strip-components=1', '-xf', tarfile],
                         cwd=targetdir)
    p.wait()
    fail_if(p.returncode != 0,
            'untar {} into {} failed'.format(tarfile, targetdir))


def gitclone(cwd, repo, tag, subdir):
    log_verbose('git clone of repo "{}" at tag "{}"'.format(repo, tag))
    p = subprocess.Popen([
        'git', 'clone', '--recursive', '--single-branch', '--depth=1', '-b', tag,
        'https://github.com/triton-inference-server/{}.git'.format(repo), subdir
    ],
                         cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0,
            'git clone of repo "{}" at tag "{}" failed'.format(repo, tag))


def cmake(cwd, args):
    log_verbose('cmake {}'.format(args))
    p = subprocess.Popen([
        'cmake',
    ] + args, cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'cmake failed')


def makeinstall(cwd, target='install'):
    log_verbose('make {}'.format(target))
    verbose_flag = 'VERBOSE=1' if FLAGS.verbose else 'VERBOSE=0'
    p = subprocess.Popen(
        ['make', '-j',
         str(FLAGS.build_parallel), verbose_flag, target],
        cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'make {} failed'.format(target))


def cmake_enable(flag):
    return 'ON' if flag else 'OFF'


def core_cmake_args(components, backends, install_dir):
    cargs = [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    cargs.append('-DTRITON_ENABLE_LOGGING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_logging)))
    cargs.append('-DTRITON_ENABLE_STATS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_stats)))
    cargs.append('-DTRITON_ENABLE_METRICS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_metrics)))
    cargs.append('-DTRITON_ENABLE_METRICS_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu_metrics)))
    cargs.append('-DTRITON_ENABLE_TRACING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_tracing)))

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))
    cargs.append('-DTRITON_MIN_COMPUTE_CAPABILITY={}'.format(
        FLAGS.min_compute_capability))

    cargs.append('-DTRITON_ENABLE_GRPC:BOOL={}'.format(
        cmake_enable('grpc' in FLAGS.endpoint)))
    cargs.append('-DTRITON_ENABLE_HTTP:BOOL={}'.format(
        cmake_enable('http' in FLAGS.endpoint)))

    cargs.append('-DTRITON_ENABLE_GCS:BOOL={}'.format(
        cmake_enable('gcs' in FLAGS.filesystem)))
    cargs.append('-DTRITON_ENABLE_S3:BOOL={}'.format(
        cmake_enable('s3' in FLAGS.filesystem)))

    cargs.append('-DTRITON_ENABLE_TENSORFLOW={}'.format(
        cmake_enable(('tensorflow1' in backends) or
                     ('tensorflow2' in backends))))

    for be in (CORE_BACKENDS + NONCORE_BACKENDS):
        if not be.startswith('tensorflow'):
            cargs.append('-DTRITON_ENABLE_{}={}'.format(
                be.upper(), cmake_enable(be in backends)))
        if (be in CORE_BACKENDS) and (be in backends):
            if be == 'pytorch':
                cargs += pytorch_cmake_args()
            elif be == 'tensorrt':
                pass
            elif be == 'custom':
                pass
            elif be == 'ensemble':
                pass
            elif be == 'caffe2':
                pass
            else:
                fail('unknown core backend {}'.format(be))

    cargs.append(
        '-DTRITON_EXTRA_LIB_PATHS=/opt/tritonserver/lib;/opt/tritonserver/lib/pytorch'
    )
    cargs.append('/workspace/build')
    return cargs


def backend_repo(be):
    if (be == 'tensorflow1') or (be == 'tensorflow2'):
        return 'tensorflow_backend'
    return '{}_backend'.format(be)


def backend_cmake_args(components, be, install_dir):
    if be == 'onnxruntime':
        args = onnxruntime_cmake_args()
    elif be == 'tensorflow1':
        args = tensorflow_cmake_args(1)
    elif be == 'tensorflow2':
        args = tensorflow_cmake_args(2)
    elif be == 'python':
        args = []
    elif be == 'dali':
        args = dali_cmake_args()
    elif be in EXAMPLE_BACKENDS:
        args = []
    else:
        fail('unknown backend {}'.format(be))

    cargs = args + [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))

    cargs.append('..')
    return cargs


def pytorch_cmake_args():
    return [
        '-DTRITON_PYTORCH_INCLUDE_PATHS=/opt/tritonserver/include/torch;/opt/tritonserver/include/torch/torch/csrc/api/include;/opt/tritonserver/include/torchvision;/usr/include/python3.6',
    ]


def onnxruntime_cmake_args():
    return [
        '-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON',
        '-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON',
        '-DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/opt/tritonserver/include/onnxruntime',
        '-DTRITON_ONNXRUNTIME_LIB_PATHS=/opt/tritonserver/backends/onnxruntime'
    ]


def tensorflow_cmake_args(ver):
    image = 'nvcr.io/nvidia/tensorflow:{}-tf{}-py3'.format(
        FLAGS.upstream_container_version, ver)
    return [
        '-DTRITON_TENSORFLOW_VERSION={}'.format(ver),
        '-DTRITON_TENSORFLOW_DOCKER_IMAGE={}'.format(image)
    ]


def dali_cmake_args():
    return [
        '-DTRITON_DALI_SKIP_DOWNLOAD=OFF',
    ]


def container_build(container_version):
    # Set the docker build-args based on 'container_version'
    if container_version in CONTAINER_VERSION_MAP:
        onnx_runtime_version = CONTAINER_VERSION_MAP[container_version][0]
        onnx_runtime_openvino_version = CONTAINER_VERSION_MAP[
            container_version][1]
    else:
        fail('unsupported container version {}'.format(container_version))

    # We can't use docker module for building container because it
    # doesn't stream output and it also seems to handle cache-from
    # incorrectly which leads to excessive rebuilds in the multistage
    # build.
    buildargmap = {
        'TRITON_VERSION':
            FLAGS.version,
        'TRITON_CONTAINER_VERSION':
            container_version,
        'BASE_IMAGE':
            'nvcr.io/nvidia/tritonserver:{}-py3'.format(container_version),
        'PYTORCH_IMAGE':
            'nvcr.io/nvidia/pytorch:{}-py3'.format(container_version),
        'ONNX_RUNTIME_VERSION':
            onnx_runtime_version,
        'ONNX_RUNTIME_OPENVINO_VERSION':
            onnx_runtime_openvino_version
    }

    cachefrommap = [
        'tritonserver_pytorch', 'tritonserver_pytorch_cache0',
        'tritonserver_pytorch_cache1', 'tritonserver_onnx',
        'tritonserver_onnx_cache0', 'tritonserver_onnx_cache1',
        'tritonserver_buildbase', 'tritonserver_buildbase_cache0',
        'tritonserver_buildbase_cache1'
    ]

    buildargs = [
        '--build-arg="{}={}"'.format(k, buildargmap[k]) for k in buildargmap
    ]
    cachefromargs = ['--cache-from={}'.format(k) for k in cachefrommap]
    commonargs = ['docker', 'build', '--pull', '-f', 'Dockerfile.buildbase']

    log_verbose('buildbase container {}'.format(commonargs + cachefromargs +
                                                buildargs))
    try:
        # First build Dockerfile.buildbase. Because of the way Docker
        # does caching with multi-stage images, we must build each
        # stage separately to make sure it is cached (specifically
        # this is needed for CI builds where the build starts with a
        # clean docker cache each time).

        # PyTorch
        p = subprocess.Popen(commonargs + cachefromargs + buildargs + [
            '-t', 'tritonserver_pytorch', '--target', 'tritonserver_pytorch',
            '.'
        ])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_pytorch failed')

        # ONNX Runtime
        p = subprocess.Popen(
            commonargs + cachefromargs + buildargs +
            ['-t', 'tritonserver_onnx', '--target', 'tritonserver_onnx', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_onnx failed')

        # Final buildbase image
        p = subprocess.Popen(commonargs + cachefromargs + buildargs +
                             ['-t', 'tritonserver_buildbase', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_buildbase failed')

        # Before attempting to run the new image, make sure any
        # previous 'tritonserver_builder' container is removed.
        client = docker.from_env(timeout=3600)

        try:
            existing = client.containers.get('tritonserver_builder')
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # ignore

        # Next run build.py inside the container with the same flags
        # as was used to run this instance, except:
        #
        # --container-version changes to --upstream-container-version
        #
        # --build-dir is added/overridden to /tmp/tritonbuild
        #
        # --install-dir is added/overridden to
        # --/tmp/tritonbuild/install
        runargs = [
            a.replace('--container-version', '--upstream-container-version')
            for a in sys.argv[1:]
        ]
        runargs += ['--build-dir', os.path.join(os.sep, 'tmp', 'tritonbuild')]
        runargs += [
            '--install-dir',
            os.path.join(os.sep, 'tmp', 'tritonbuild', 'install')
        ]

        log_verbose('run {}'.format(runargs))
        container = client.containers.run(
            'tritonserver_buildbase',
            './build.py {}'.format(' '.join(runargs)),
            detach=True,
            name='tritonserver_builder',
            volumes={
                '/var/run/docker.sock': {
                    'bind': '/var/run/docker.sock',
                    'mode': 'rw'
                }
            },
            working_dir='/workspace')
        if FLAGS.verbose:
            for ln in container.logs(stream=True):
                log_verbose(ln)
        ret = container.wait()
        fail_if(ret['StatusCode'] != 0, 'tritonserver_builder failed: {}'.format(ret))

        # It is possible to copy the install artifacts from the
        # container at this point (and, for example put them in the
        # specified install directory on the host). But for container
        # build we just want to use the artifacts in the server base
        # container which is created below.
        #mkdir(FLAGS.install_dir)
        #tarfilename = os.path.join(FLAGS.install_dir, 'triton.tar')
        #install_tar, stat_tar = container.get_archive(
        #    os.path.join(os.sep, 'tmp', 'tritonbuild', 'install'))
        #with open(tarfilename, 'wb') as taroutfile:
        #    for d in install_tar:
        #        taroutfile.write(d)
        #untar(FLAGS.install_dir, tarfilename)

        # Build is complete, save the container as the
        # tritonserver_build image.
        try:
            client.images.remove('tritonserver_build', force=True)
        except docker.errors.ImageNotFound:
            pass  # ignore

        container.commit('tritonserver_build', 'latest')
        container.remove(force=True)

        # Final base image... this is a multi-stage build that uses
        # the install artifacts from the tritonserver_build container.
        baseargs = ['docker', 'build', '-f', 'Dockerfile']
        p = subprocess.Popen(baseargs + buildargs + ['-t', 'tritonserver', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver failed')

    except Exception as e:
        logging.error(traceback.format_exc())
        fail('container build failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Used internally for docker build, not intended for direct use
    parser.add_argument('--upstream-container-version',
                        type=str,
                        required=False,
                        help=argparse.SUPPRESS)

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
        '--install-dir',
        required=False,
        default=None,
        help='Install directory, default is <builddir>/opt/tritonserver.')
    parser.add_argument(
        '--build-type',
        required=False,
        default='Release',
        help=
        'Build type, one of "Release", "Debug", "RelWithDebInfo" or "MinSizeRel". Default is "Release".'
    )
    parser.add_argument(
        '-j',
        '--build-parallel',
        type=int,
        required=False,
        default=None,
        help='Build parallelism. Defaults to 2 * number-of-cores.')

    parser.add_argument('--version',
                        type=str,
                        required=False,
                        default='0.0.0',
                        help='The Triton version.')
    parser.add_argument(
        '--container-version',
        type=str,
        required=False,
        help=
        'The Triton container version. If specified, Docker will be used for the build and component versions will be set automatically.'
    )

    parser.add_argument('--enable-logging',
                        action="store_true",
                        required=False,
                        help='Enable logging.')
    parser.add_argument('--enable-stats',
                        action="store_true",
                        required=False,
                        help='Enable statistics collection.')
    parser.add_argument('--enable-metrics',
                        action="store_true",
                        required=False,
                        help='Enable metrics reporting.')
    parser.add_argument('--enable-gpu-metrics',
                        action="store_true",
                        required=False,
                        help='Include GPU metrics in reported metrics.')
    parser.add_argument('--enable-tracing',
                        action="store_true",
                        required=False,
                        help='Enable tracing.')
    parser.add_argument('--enable-nvtx',
                        action="store_true",
                        required=False,
                        help='Enable NVTX.')
    parser.add_argument('--enable-gpu',
                        action="store_true",
                        required=False,
                        help='Enable GPU support.')
    parser.add_argument(
        '--min-compute-capability',
        type=str,
        required=False,
        default='6.0',
        help='Minimum CUDA compute capability supported by server.')

    parser.add_argument(
        '-e',
        '--endpoint',
        action='append',
        required=False,
        help=
        'Include specified endpoint in build. Allowed values are "grpc" and "http".'
    )
    parser.add_argument(
        '--filesystem',
        action='append',
        required=False,
        help=
        'Include specified filesystem in build. Allowed values are "gcs" and "s3".'
    )
    parser.add_argument(
        '-b',
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>[:<repo-tag>]. <repo-tag> indicates the git tag/branch to use for the build, default is "main".'
    )
    parser.add_argument(
        '-r',
        '--repo-tag',
        action='append',
        required=False,
        help=
        'The git tag/branch to use for a component of the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", or "backend".'
    )

    FLAGS = parser.parse_args()

    if FLAGS.repo_tag is None:
        FLAGS.repo_tag = []
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.endpoint is None:
        FLAGS.endpoint = []
    if FLAGS.filesystem is None:
        FLAGS.filesystem = []

    # If --container-version is specified then we use
    # Dockerfile.buildbase to create the appropriate base build
    # container and then perform the actual build within that
    # container.
    if FLAGS.container_version is not None:
        container_build(FLAGS.container_version)
        sys.exit(0)

    log('Building Triton Inference Server')

    if FLAGS.install_dir is None:
        FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")
    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2
    if FLAGS.version is None:
        FLAGS.version = '0.0.0'

    # Initialize map of common components and repo-tag for each.
    components = {'common': 'main', 'core': 'main', 'backend': 'main'}
    for be in FLAGS.repo_tag:
        parts = be.split(':')
        fail_if(
            len(parts) != 2,
            '--repo-tag must specific <component-name>:<repo-tag>')
        fail_if(
            parts[0] not in components,
            '--repo-tag <component-name> must be "common", "core", or "backend"'
        )
        components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    for be in FLAGS.backend:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append('main')
        log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        backends[parts[0]] = parts[1]

    # Build the core server. For now the core is contained in this
    # repo so we just build in place
    if True:
        repo_build_dir = os.path.join(FLAGS.build_dir, 'tritonserver', 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, 'tritonserver',
                                        'install')

        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              core_cmake_args(components, backends, repo_install_dir))
        makeinstall(repo_build_dir, target='server')

        core_install_dir = FLAGS.install_dir
        mkdir(core_install_dir)
        cpdir(repo_install_dir, core_install_dir)

    # Build each backend...
    for be in backends:
        # Core backends are not built separately from core so skip...
        if (be in CORE_BACKENDS):
            continue

        repo_build_dir = os.path.join(FLAGS.build_dir, be, 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, be, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, backend_repo(be), backends[be], be)
        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              backend_cmake_args(components, be, repo_install_dir))
        makeinstall(repo_build_dir)

        backend_install_dir = os.path.join(FLAGS.install_dir, 'backends', be)
        rmdir(backend_install_dir)
        mkdir(backend_install_dir)
        cpdir(os.path.join(repo_install_dir, 'backends', be),
              backend_install_dir)
