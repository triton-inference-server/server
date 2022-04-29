#!/usr/bin/env python3
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os.path
import multiprocessing
import pathlib
import platform
import shutil
import subprocess
import sys
import traceback
from distutils.dir_util import copy_tree
from inspect import getsourcefile

#
# Build Triton Inference Server.
#

# By default build.py builds the Triton container. The TRITON_VERSION
# file indicates the Triton version and TRITON_VERSION_MAP is used to
# determine the corresponding container version and upstream container
# version (upstream containers are dependencies required by
# Triton). These versions may be overridden. See docs/build.md for
# more information.

# Map from Triton version to corresponding container and component versions.
#
#   triton version ->
#     (triton container version,
#      upstream container version,
#      ORT version,
#      ORT OpenVINO version (use None to disable OpenVINO in ORT),
#      Standalone OpenVINO version,
#      DCGM version
#     )
#
# Currently the OpenVINO versions used in ORT and standalone must
# match because of the way dlopen works with loading the backends. If
# different versions are used then one backend or the other will
# incorrectly load the other version of the openvino libraries.
#
# The standalone openVINO describes multiple versions where each version
# is a pair of openVINO version and openVINO package version. When openVINO
# package version is specified, then backend will be built with pre-built
# openVINO release from Intel. If the package version is specified as None,
# then openVINO for the backend is built from source with openMP support.
# By default, only the first version is built. To build the all the versions
# in list use --build-multiple-openvino. Triton will use the first version
# for inference by default. In order to use different version, Triton should
# be invoked with appropriate backend configuration:
# (--backend-config=openvino,version=<version_str>)
# The version string can be obtained as follows:
# <major_version>_<minor_version>[_pre]
# Append '_pre' only if the openVINO backend was built with prebuilt openVINO
# library. In other words, when the second element of the pair is not None.
# To use ('2021.4', None) version_str should be `2021_4'.
# To use ('2021.4', '2021.4.582') version_str should be `2021_4_pre'.
# User can also build openvino backend from specific commit sha of openVINO
# repository. The pair should be (`SPECIFIC`, <commit_sha_of_ov_repo>).
# Note: Not all sha ids would successfuly compile and work.
#
TRITON_VERSION_MAP = {
    '2.21.0dev': (
        '22.04dev',  # triton container
        '22.02',  # upstream container
        '1.10.0',  # ORT
        '2021.4.582',  # ORT OpenVINO
        (('2021.4', None), ('2021.4', '2021.4.582'),
         ('SPECIFIC', 'f2f281e6')),  # Standalone OpenVINO
        '2.2.9')  # DCGM version
}

EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['ensemble']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali', 'pytorch',
    'openvino', 'fil', 'fastertransformer', 'tensorrt', 'armnn_tflite'
]
EXAMPLE_REPOAGENTS = ['checksum']
FLAGS = None
EXTRA_CORE_CMAKE_FLAGS = {}
OVERRIDE_CORE_CMAKE_FLAGS = {}
EXTRA_BACKEND_CMAKE_FLAGS = {}
OVERRIDE_BACKEND_CMAKE_FLAGS = {}

SCRIPT_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


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


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def target_machine():
    if FLAGS.target_machine is not None:
        return FLAGS.target_machine
    return platform.machine().lower()


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


def gitclone(cwd, repo, tag, subdir, org):
    # If 'tag' starts with "pull/" then it must be of form
    # "pull/<pr>/head". We just clone at "main" and then fetch the
    # reference onto a new branch we name "tritonbuildref".
    clone_dir = cwd + '/' + subdir
    if tag.startswith("pull/"):
        log_verbose('git clone of repo "{}" at ref "{}"'.format(repo, tag))

        if os.path.exists(clone_dir) and not FLAGS.no_force_clone:
            rmdir(clone_dir)

        if not os.path.exists(clone_dir):
            p = subprocess.Popen([
                'git', 'clone', '--recursive', '--depth=1', '{}/{}.git'.format(
                    org, repo), subdir
            ],
                                 cwd=cwd)
            p.wait()
            fail_if(
                p.returncode != 0,
                'git clone of repo "{}" at branch "main" failed'.format(repo))

            log_verbose('git fetch of ref "{}"'.format(tag))
            p = subprocess.Popen(
                ['git', 'fetch', 'origin', '{}:tritonbuildref'.format(tag)],
                cwd=os.path.join(cwd, subdir))
            p.wait()
            fail_if(p.returncode != 0,
                    'git fetch of ref "{}" failed'.format(tag))

            log_verbose('git checkout of tritonbuildref')
            p = subprocess.Popen(['git', 'checkout', 'tritonbuildref'],
                                 cwd=os.path.join(cwd, subdir))
            p.wait()
            fail_if(p.returncode != 0,
                    'git checkout of branch "tritonbuildref" failed')

    else:
        log_verbose('git clone of repo "{}" at tag "{}"'.format(repo, tag))

        if os.path.exists(clone_dir) and not FLAGS.no_force_clone:
            rmdir(clone_dir)

        if not os.path.exists(clone_dir):
            p = subprocess.Popen([
                'git', 'clone', '--recursive', '--single-branch', '--depth=1',
                '-b', tag, '{}/{}.git'.format(org, repo), subdir
            ],
                                 cwd=cwd)
            p.wait()
            fail_if(
                p.returncode != 0,
                'git clone of repo "{}" at tag "{}" failed'.format(repo, tag))


def prebuild_command():
    p = subprocess.Popen(FLAGS.container_prebuild_command.split())
    p.wait()
    fail_if(p.returncode != 0, 'container prebuild cmd failed')


def cmake(cwd, args):
    log_verbose('cmake {}'.format(args))
    p = subprocess.Popen([
        'cmake',
    ] + args, cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'cmake failed')


def makeinstall(cwd, target='install'):
    log_verbose('make {}'.format(target))

    if target_platform() == 'windows':
        verbose_flag = '' if FLAGS.verbose else '-clp:ErrorsOnly'
        buildtype_flag = '-p:Configuration={}'.format(FLAGS.build_type)
        p = subprocess.Popen([
            'msbuild.exe', '-m:{}'.format(str(FLAGS.build_parallel)),
            verbose_flag, buildtype_flag, '{}.vcxproj'.format(target)
        ],
                             cwd=cwd)
    else:
        verbose_flag = 'VERBOSE=1' if FLAGS.verbose else 'VERBOSE=0'
        p = subprocess.Popen(
            ['make', '-j',
             str(FLAGS.build_parallel), verbose_flag, target],
            cwd=cwd)

    p.wait()
    fail_if(p.returncode != 0, 'make {} failed'.format(target))


def cmake_core_arg(name, type, value):
    # Return cmake -D setting to set name=value for core build. Use
    # command-line specified value if one is given.
    if name in OVERRIDE_CORE_CMAKE_FLAGS:
        value = OVERRIDE_CORE_CMAKE_FLAGS[name]
    if type is None:
        type = ''
    else:
        type = ':{}'.format(type)
    return '-D{}{}={}'.format(name, type, value)


def cmake_core_enable(name, flag):
    # Return cmake -D setting to set name=flag?ON:OFF for core
    # build. Use command-line specified value for 'flag' if one is
    # given.
    if name in OVERRIDE_CORE_CMAKE_FLAGS:
        value = OVERRIDE_CORE_CMAKE_FLAGS[name]
    else:
        value = 'ON' if flag else 'OFF'
    return '-D{}:BOOL={}'.format(name, value)


def cmake_core_extra_args():
    args = []
    for k, v in EXTRA_CORE_CMAKE_FLAGS.items():
        args.append('-D{}={}'.format(k, v))
    return args


def cmake_backend_arg(backend, name, type, value):
    # Return cmake -D setting to set name=value for backend build. Use
    # command-line specified value if one is given.
    if backend in OVERRIDE_BACKEND_CMAKE_FLAGS:
        if name in OVERRIDE_BACKEND_CMAKE_FLAGS[backend]:
            value = OVERRIDE_BACKEND_CMAKE_FLAGS[backend][name]
    if type is None:
        type = ''
    else:
        type = ':{}'.format(type)
    return '-D{}{}={}'.format(name, type, value)


def cmake_backend_enable(backend, name, flag):
    # Return cmake -D setting to set name=flag?ON:OFF for backend
    # build. Use command-line specified value for 'flag' if one is
    # given.
    value = None
    if backend in OVERRIDE_BACKEND_CMAKE_FLAGS:
        if name in OVERRIDE_BACKEND_CMAKE_FLAGS[backend]:
            value = OVERRIDE_BACKEND_CMAKE_FLAGS[backend][name]
    if value is None:
        value = 'ON' if flag else 'OFF'
    return '-D{}:BOOL={}'.format(name, value)


def cmake_backend_extra_args(backend):
    args = []
    if backend in EXTRA_BACKEND_CMAKE_FLAGS:
        for k, v in EXTRA_BACKEND_CMAKE_FLAGS[backend].items():
            args.append('-D{}={}'.format(k, v))
    return args


def cmake_repoagent_arg(name, type, value):
    # For now there is no override for repo-agents
    if type is None:
        type = ''
    else:
        type = ':{}'.format(type)
    return '-D{}{}={}'.format(name, type, value)


def cmake_repoagent_enable(name, flag):
    # For now there is no override for repo-agents
    value = 'ON' if flag else 'OFF'
    return '-D{}:BOOL={}'.format(name, value)


def cmake_repoagent_extra_args():
    # For now there is no extra args for repo-agents
    args = []
    return args


def core_cmake_args(components, backends, install_dir):
    cargs = [
        cmake_core_arg('CMAKE_BUILD_TYPE', None, FLAGS.build_type),
        cmake_core_arg('CMAKE_INSTALL_PREFIX', 'PATH', install_dir),
        cmake_core_arg('TRITON_VERSION', 'STRING', FLAGS.version),
        cmake_core_arg('TRITON_COMMON_REPO_TAG', 'STRING',
                       components['common']),
        cmake_core_arg('TRITON_CORE_REPO_TAG', 'STRING', components['core']),
        cmake_core_arg('TRITON_BACKEND_REPO_TAG', 'STRING',
                       components['backend']),
        cmake_core_arg('TRITON_THIRD_PARTY_REPO_TAG', 'STRING',
                       components['thirdparty'])
    ]

    cargs.append(
        cmake_core_enable('TRITON_ENABLE_LOGGING', FLAGS.enable_logging))
    cargs.append(cmake_core_enable('TRITON_ENABLE_STATS', FLAGS.enable_stats))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_METRICS', FLAGS.enable_metrics))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_METRICS_GPU',
                          FLAGS.enable_gpu_metrics))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_TRACING', FLAGS.enable_tracing))
    cargs.append(cmake_core_enable('TRITON_ENABLE_NVTX', FLAGS.enable_nvtx))

    cargs.append(cmake_core_enable('TRITON_ENABLE_GPU', FLAGS.enable_gpu))
    cargs.append(
        cmake_core_arg('TRITON_MIN_COMPUTE_CAPABILITY', None,
                       FLAGS.min_compute_capability))

    cargs.append(
        cmake_core_enable('TRITON_ENABLE_MALI_GPU', FLAGS.enable_mali_gpu))

    cargs.append(
        cmake_core_enable('TRITON_ENABLE_GRPC', 'grpc' in FLAGS.endpoint))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_HTTP', 'http' in FLAGS.endpoint))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_SAGEMAKER', 'sagemaker'
                          in FLAGS.endpoint))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_VERTEX_AI', 'vertex-ai'
                          in FLAGS.endpoint))

    cargs.append(
        cmake_core_enable('TRITON_ENABLE_GCS', 'gcs' in FLAGS.filesystem))
    cargs.append(cmake_core_enable('TRITON_ENABLE_S3', 's3'
                                   in FLAGS.filesystem))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_AZURE_STORAGE', 'azure_storage'
                          in FLAGS.filesystem))

    cargs.append(
        cmake_core_enable('TRITON_ENABLE_ENSEMBLE', 'ensemble' in backends))
    cargs.append(
        cmake_core_enable('TRITON_ENABLE_TENSORRT', 'tensorrt' in backends))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append(cmake_core_arg(evar[len('TRITONBUILD_'):], None, eval))

    cargs += cmake_core_extra_args()
    cargs.append(FLAGS.cmake_dir)
    return cargs


def repoagent_repo(ra):
    return '{}_repository_agent'.format(ra)


def repoagent_cmake_args(images, components, ra, install_dir):
    if ra in EXAMPLE_REPOAGENTS:
        args = []
    else:
        fail('unknown agent {}'.format(ra))

    cargs = args + [
        cmake_repoagent_arg('CMAKE_BUILD_TYPE', None, FLAGS.build_type),
        cmake_repoagent_arg('CMAKE_INSTALL_PREFIX', 'PATH', install_dir),
        cmake_repoagent_arg('TRITON_COMMON_REPO_TAG', 'STRING',
                            components['common']),
        cmake_repoagent_arg('TRITON_CORE_REPO_TAG', 'STRING',
                            components['core'])
    ]

    cargs.append(cmake_repoagent_enable('TRITON_ENABLE_GPU', FLAGS.enable_gpu))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append(
                cmake_repoagent_arg(evar[len('TRITONBUILD_'):], None, eval))

    cargs += cmake_repoagent_extra_args()
    cargs.append('..')
    return cargs


def backend_repo(be):
    if (be == 'tensorflow1') or (be == 'tensorflow2'):
        return 'tensorflow_backend'
    if be.startswith("openvino"):
        return 'openvino_backend'
    return '{}_backend'.format(be)


def backend_cmake_args(images, components, be, install_dir, library_paths,
                       variant_index):
    if be == 'onnxruntime':
        args = onnxruntime_cmake_args(images, library_paths)
    elif be.startswith('openvino'):
        args = openvino_cmake_args(be, variant_index)
    elif be == 'tensorflow1':
        args = tensorflow_cmake_args(1, images, library_paths)
    elif be == 'tensorflow2':
        args = tensorflow_cmake_args(2, images, library_paths)
    elif be == 'python':
        args = []
    elif be == 'dali':
        args = dali_cmake_args()
    elif be == 'pytorch':
        args = pytorch_cmake_args(images)
    elif be == 'armnn_tflite':
        args = armnn_tflite_cmake_args()
    elif be == 'fil':
        args = fil_cmake_args(images)
    elif be == 'fastertransformer':
        args = []
    elif be == 'tensorrt':
        args = tensorrt_cmake_args()
    elif be in EXAMPLE_BACKENDS:
        args = []
    else:
        fail('unknown backend {}'.format(be))

    cargs = args + [
        cmake_backend_arg(be, 'CMAKE_BUILD_TYPE', None, FLAGS.build_type),
        cmake_backend_arg(be, 'CMAKE_INSTALL_PREFIX', 'PATH', install_dir),
        cmake_backend_arg(be, 'TRITON_COMMON_REPO_TAG', 'STRING',
                          components['common']),
        cmake_backend_arg(be, 'TRITON_CORE_REPO_TAG', 'STRING',
                          components['core']),
        cmake_backend_arg(be, 'TRITON_BACKEND_REPO_TAG', 'STRING',
                          components['backend'])
    ]

    cargs.append(cmake_backend_enable(be, 'TRITON_ENABLE_GPU',
                                      FLAGS.enable_gpu))
    cargs.append(
        cmake_backend_enable(be, 'TRITON_ENABLE_MALI_GPU',
                             FLAGS.enable_mali_gpu))
    cargs.append(
        cmake_backend_enable(be, 'TRITON_ENABLE_STATS', FLAGS.enable_stats))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append(
                cmake_backend_arg(be, evar[len('TRITONBUILD_'):], None, eval))

    cargs += cmake_backend_extra_args(be)
    cargs.append('..')
    return cargs


def pytorch_cmake_args(images):

    # If platform is jetpack do not use docker based build
    if target_platform() == 'jetpack':
        if 'pytorch' not in library_paths:
            raise Exception(
                "Must specify library path for pytorch using --library-paths=pytorch:<path_to_pytorch>"
            )
        pt_lib_path = library_paths['pytorch'] + "/lib"
        pt_include_paths = ""
        for suffix in [
                'include/torch', 'include/torch/torch/csrc/api/include',
                'include/torchvision'
        ]:
            pt_include_paths += library_paths['pytorch'] + '/' + suffix + ';'
        cargs = [
            cmake_backend_arg('pytorch', 'TRITON_PYTORCH_INCLUDE_PATHS', None,
                              pt_include_paths),
            cmake_backend_arg('pytorch', 'TRITON_PYTORCH_LIB_PATHS', None,
                              pt_lib_path),
        ]
    else:
        if "pytorch" in images:
            image = images["pytorch"]
        else:
            image = 'nvcr.io/nvidia/pytorch:{}-py3'.format(
                FLAGS.upstream_container_version)
        cargs = [
            cmake_backend_arg('pytorch', 'TRITON_PYTORCH_DOCKER_IMAGE', None,
                              image),
        ]

        if FLAGS.enable_gpu:
            cargs.append(
                cmake_backend_enable('pytorch',
                                     'TRITON_PYTORCH_ENABLE_TORCHTRT', True))
    return cargs


def onnxruntime_cmake_args(images, library_paths):
    cargs = [
        cmake_backend_arg('onnxruntime', 'TRITON_BUILD_ONNXRUNTIME_VERSION',
                          None, TRITON_VERSION_MAP[FLAGS.version][2])
    ]

    # TRITON_ENABLE_GPU is already set for all backends in backend_cmake_args()
    if FLAGS.enable_gpu:
        cargs.append(
            cmake_backend_enable('onnxruntime',
                                 'TRITON_ENABLE_ONNXRUNTIME_TENSORRT', True))

    # If platform is jetpack do not use docker based build
    if target_platform() == 'jetpack':
        if 'onnxruntime' not in library_paths:
            raise Exception(
                "Must specify library path for onnxruntime using --library-paths=onnxruntime:<path_to_onnxruntime>"
            )
        ort_lib_path = library_paths['onnxruntime'] + "/lib"
        ort_include_path = library_paths['onnxruntime'] + "/include"
        cargs += [
            cmake_backend_arg('onnxruntime', 'TRITON_ONNXRUNTIME_INCLUDE_PATHS',
                              None, ort_include_path),
            cmake_backend_arg('onnxruntime', 'TRITON_ONNXRUNTIME_LIB_PATHS',
                              None, ort_lib_path),
            cmake_backend_enable('onnxruntime',
                                 'TRITON_ENABLE_ONNXRUNTIME_OPENVINO', False)
        ]
    else:
        if target_platform() == 'windows':
            if 'base' in images:
                cargs.append(
                    cmake_backend_arg('onnxruntime', 'TRITON_BUILD_CONTAINER',
                                      None, images['base']))
        else:
            if 'base' in images:
                cargs.append(
                    cmake_backend_arg('onnxruntime', 'TRITON_BUILD_CONTAINER',
                                      None, images['base']))
            else:
                cargs.append(
                    cmake_backend_arg('onnxruntime',
                                      'TRITON_BUILD_CONTAINER_VERSION', None,
                                      TRITON_VERSION_MAP[FLAGS.version][1]))

            if ((target_machine() != 'aarch64') and
                (TRITON_VERSION_MAP[FLAGS.version][3] is not None)):
                cargs.append(
                    cmake_backend_enable('onnxruntime',
                                         'TRITON_ENABLE_ONNXRUNTIME_OPENVINO',
                                         True))
                cargs.append(
                    cmake_backend_arg(
                        'onnxruntime',
                        'TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION', None,
                        TRITON_VERSION_MAP[FLAGS.version][3]))

    return cargs


def openvino_cmake_args(be, variant_index):
    using_specific_commit_sha = False
    if TRITON_VERSION_MAP[FLAGS.version][4][variant_index][0] == 'SPECIFIC':
        using_specific_commit_sha = True

    ov_version = TRITON_VERSION_MAP[FLAGS.version][4][variant_index][1]
    if ov_version:
        if using_specific_commit_sha:
            use_prebuilt_ov = False
        else:
            use_prebuilt_ov = True
    else:
        # If the OV package version is None, then we are not using prebuilt package
        ov_version = TRITON_VERSION_MAP[FLAGS.version][4][variant_index][0]
        use_prebuilt_ov = False
    if using_specific_commit_sha:
        cargs = [
            cmake_backend_arg(be, 'TRITON_BUILD_OPENVINO_COMMIT_SHA', None,
                              ov_version),
        ]
    else:
        cargs = [
            cmake_backend_arg(be, 'TRITON_BUILD_OPENVINO_VERSION', None,
                              ov_version),
        ]
    cargs.append(
        cmake_backend_arg(be, 'TRITON_OPENVINO_BACKEND_INSTALLDIR', None, be))
    if target_platform() == 'windows':
        if 'base' in images:
            cargs.append(
                cmake_backend_arg(be, 'TRITON_BUILD_CONTAINER', None,
                                  images['base']))
    else:
        if 'base' in images:
            cargs.append(
                cmake_backend_arg(be, 'TRITON_BUILD_CONTAINER', None,
                                  images['base']))
        else:
            cargs.append(
                cmake_backend_arg(be, 'TRITON_BUILD_CONTAINER_VERSION', None,
                                  TRITON_VERSION_MAP[FLAGS.version][1]))
        cargs.append(
            cmake_backend_enable(be, 'TRITON_BUILD_USE_PREBUILT_OPENVINO',
                                 use_prebuilt_ov))
    return cargs


def tensorrt_cmake_args():
    cargs = [
        cmake_backend_enable('tensorrt', 'TRITON_ENABLE_NVTX',
                             FLAGS.enable_nvtx),
    ]
    if target_platform() == 'windows':
        cargs.append(
            cmake_backend_arg('tensorrt', 'TRITON_TENSORRT_INCLUDE_PATHS', None,
                              'c:/TensorRT/include'))

    return cargs


def tensorflow_cmake_args(ver, images, library_paths):
    backend_name = "tensorflow{}".format(ver)

    # If platform is jetpack do not use docker images
    extra_args = []
    if target_platform() == 'jetpack':
        if backend_name in library_paths:
            extra_args = [
                cmake_backend_arg(backend_name, 'TRITON_TENSORFLOW_LIB_PATHS',
                                  None, library_paths[backend_name])
            ]
        else:
            raise Exception(
                f"Must specify library path for {backend_name} using --library-paths={backend_name}:<path_to_{backend_name}>"
            )
    else:
        # If a specific TF image is specified use it, otherwise pull from NGC.
        if backend_name in images:
            image = images[backend_name]
        else:
            image = 'nvcr.io/nvidia/tensorflow:{}-tf{}-py3'.format(
                FLAGS.upstream_container_version, ver)
        extra_args = [
            cmake_backend_arg(backend_name, 'TRITON_TENSORFLOW_DOCKER_IMAGE',
                              None, image)
        ]
    return [
        cmake_backend_arg(backend_name, 'TRITON_TENSORFLOW_VERSION', None, ver)
    ] + extra_args


def dali_cmake_args():
    return [
        cmake_backend_enable('dali', 'TRITON_DALI_SKIP_DOWNLOAD', False),
    ]


def fil_cmake_args(images):
    cargs = [cmake_backend_enable('fil', 'TRITON_FIL_DOCKER_BUILD', True)]
    if 'base' in images:
        cargs.append(
            cmake_backend_arg('fil', 'TRITON_BUILD_CONTAINER', None,
                              images['base']))
    else:
        cargs.append(
            cmake_backend_arg('fil', 'TRITON_BUILD_CONTAINER_VERSION', None,
                              TRITON_VERSION_MAP[FLAGS.version][1]))

    return cargs


def armnn_tflite_cmake_args():
    return [
        cmake_backend_arg('armnn_tflite', 'JOBS', None,
                          multiprocessing.cpu_count()),
    ]


def install_dcgm_libraries(dcgm_version, target_machine):
    if dcgm_version == '':
        fail(
            'unable to determine default repo-tag, DCGM version not known for {}'
            .format(FLAGS.version))
        return ''
    else:
        if target_machine == 'aarch64':
            return '''
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/ /" && \
    apt-get update && apt-get install -y datacenter-gpu-manager=1:{}
'''.format(dcgm_version, dcgm_version)
        else:
            return '''
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && apt-get install -y datacenter-gpu-manager=1:{}
'''.format(dcgm_version, dcgm_version)


def get_container_versions(version, container_version,
                           upstream_container_version):
    if container_version is None:
        if version not in TRITON_VERSION_MAP:
            fail('container version not known for {}'.format(version))
        container_version = TRITON_VERSION_MAP[version][0]
    if upstream_container_version is None:
        if version not in TRITON_VERSION_MAP:
            fail('upstream container version not known for {}'.format(version))
        upstream_container_version = TRITON_VERSION_MAP[version][1]
    return container_version, upstream_container_version


def create_dockerfile_buildbase(ddir, dockerfile_name, argmap):
    df = '''
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])

    df += '''
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
'''
    # Install the windows- or linux-specific buildbase dependencies
    if target_platform() == 'windows':
        df += '''
SHELL ["cmd", "/S", "/C"]
'''
    else:
        df += '''
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip and libarchive-dev is needed by python backend
# uuid-dev and pkg-config is needed for Azure Storage
# scons is needed for armnn_tflite backend build dep
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            ca-certificates \
            autoconf \
            automake \
            build-essential \
            docker.io \
            git \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            libcurl4-openssl-dev \
            libb64-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            rapidjson-dev \
            scons \
            software-properties-common \
            unzip \
            wget \
            zlib1g-dev \
            libarchive-dev \
            pkg-config \
            uuid-dev \
            libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker

# Server build requires recent version of CMake (FetchContent required)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1
'''

    # Copy in the triton source. We remove existing contents first in
    # case the FROM container has something there already.
    if target_platform() == 'windows':
        df += '''
WORKDIR /workspace
RUN rmdir /S/Q * || exit 0
COPY . .
'''
    else:
        df += '''
WORKDIR /workspace
RUN rm -fr *
COPY . .
ENTRYPOINT []
'''
        if FLAGS.enable_gpu:
            df += install_dcgm_libraries(argmap['DCGM_VERSION'],
                                         target_machine())

    df += '''
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
'''

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_build(ddir, dockerfile_name, backends, build_dir):
    df = '''
FROM tritonserver_builder_image AS build
FROM tritonserver_buildbase
COPY --from=build {0} {0}
'''.format(build_dir)

    # If requested, package the source code for all OSS used to build
    # Triton Windows is not delivered as a container (and tar not
    # available) so skip for windows platform.
    if target_platform() != 'windows':
        if not FLAGS.no_core_build and not FLAGS.no_container_source:
            df += '''
RUN mkdir -p {0}/install/third-party-src && \
    (cd {0}/tritonserver/build && \
     tar zcf {0}/install/third-party-src/src.tar.gz third-party-src)
COPY --from=build /workspace/docker/README.third-party-src {0}/install/third-party-src/README
'''.format(build_dir)

    if 'onnxruntime' in backends:
        if target_platform() != 'windows':
            df += '''
# Copy ONNX custom op library and model (needed for testing)
RUN if [ -d {0}/onnxruntime ]; then \
      cp {0}/onnxruntime/install/test/libcustom_op_library.so /workspace/qa/L0_custom_ops/.; \
      cp {0}/onnxruntime/install/test/custom_op_test.onnx /workspace/qa/L0_custom_ops/.; \
    fi
'''.format(build_dir)

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_linux(ddir, dockerfile_name, argmap, backends, repoagents,
                            endpoints, build_dir):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${{BUILD_IMAGE}} AS tritonserver_build

'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])

    # PyTorch backend needs extra CUDA and other dependencies during runtime
    # that are missing in the CPU only base container. These dependencies
    # must be copied from the Triton Min image
    if not FLAGS.enable_gpu and ('pytorch' in backends):
        df += '''
############################################################################
##  Triton Min image
############################################################################
FROM {} AS min_container

'''.format(argmap['GPU_BASE_IMAGE'])

    df += '''
############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}
'''

    df += dockerfile_prepare_container_linux(argmap, backends, FLAGS.enable_gpu,
                                             target_machine())

    df += '''
WORKDIR /opt/tritonserver
COPY --chown=1000:1000 LICENSE .
COPY --chown=1000:1000 TRITON_VERSION .
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .
'''

    if not FLAGS.no_core_build:
        df += '''
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/lib/libtritonserver.so lib/
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/include/triton/core include/triton/core

# Top-level include/core not copied so --chown does not set it correctly,
# so explicit set on all of include
RUN chown -R triton-server:triton-server include
'''.format(build_dir)

        # If requested, include the source code for all OSS used to build Triton
        if not FLAGS.no_container_source:
            df += '''
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/third-party-src third-party-src
'''.format(build_dir)

        # Add feature labels for SageMaker endpoint
        if 'sagemaker' in endpoints:
            df += '''
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
COPY --chown=1000:1000 --from=tritonserver_build /workspace/docker/sagemaker/serve /usr/bin/.
'''

    for noncore in NONCORE_BACKENDS:
        if noncore in backends:
            df += '''
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/backends backends
'''.format(build_dir)
            break

    if len(repoagents) > 0:
        df += '''
COPY --chown=1000:1000 --from=tritonserver_build {0}/install/repoagents repoagents
'''.format(build_dir)

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def dockerfile_prepare_container_linux(argmap, backends, enable_gpu,
                                       target_machine):
    gpu_enabled = 1 if enable_gpu else 0
    # Common steps to produce docker images shared by build.py and compose.py.
    # Sets enviroment variables, installs dependencies and adds entrypoint
    df = '''
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
'''

    # TODO Remove once the ORT-OpenVINO "Exception while Reading network" is fixed
    if 'onnxruntime' in backends:
        df += '''
ENV LD_LIBRARY_PATH /opt/tritonserver/backends/onnxruntime:${LD_LIBRARY_PATH}
'''

    backend_dependencies = ""
    # libgomp1 is needed by both onnxruntime and pytorch backends
    if ('onnxruntime' in backends) or ('pytorch' in backends):
        backend_dependencies = "libgomp1"

    # libgfortran5 is needed by pytorch backend on ARM
    if ('pytorch' in backends) and (target_machine == 'aarch64'):
        backend_dependencies += " libgfortran5"

    df += '''
ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    {gpu_enabled}

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

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            libb64-0d \
            libcurl4-openssl-dev \
            libre2-5 \
            git \
            dirmngr \
            libnuma-dev \
            curl \
            {backend_dependencies} && \
    rm -rf /var/lib/apt/lists/*
'''.format(gpu_enabled=gpu_enabled, backend_dependencies=backend_dependencies)

    if enable_gpu:
        df += install_dcgm_libraries(argmap['DCGM_VERSION'], target_machine)
        df += '''
# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib
'''

    elif 'pytorch' in backends:
        # Add dependencies for pytorch backend. Note: Even though the build is
        # cpu-only, the version of pytorch we are using depends upon libraries
        # like cuda and cudnn. Since these dependencies are not present in ubuntu
        # base image, we must copy these from the Triton min container ourselves.
        df += '''
RUN mkdir -p /usr/local/cuda/lib64/stubs
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusparse.so /usr/local/cuda/lib64/stubs/libcusparse.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusolver.so /usr/local/cuda/lib64/stubs/libcusolver.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcurand.so /usr/local/cuda/lib64/stubs/libcurand.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcufft.so /usr/local/cuda/lib64/stubs/libcufft.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublas.so /usr/local/cuda/lib64/stubs/libcublas.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.11

RUN mkdir -p /usr/local/cuda/targets/x86_64-linux/lib
COPY --from=min_container /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so.11.0 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcupti.so.11.6 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-11.6/targets/x86_64-linux/lib/libnvToolsExt.so.1 /usr/local/cuda/targets/x86_64-linux/lib/.

COPY --from=min_container /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2
COPY --from=min_container /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.8

RUN apt-get update && \
        apt-get install -y --no-install-recommends openmpi-bin

ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
'''

    # Add dependencies needed for python backend
    if 'python' in backends:
        df += '''
# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            python3 libarchive-dev \
            python3-pip \
            libpython3-dev && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade numpy && \
    rm -rf /var/lib/apt/lists/*
'''

    df += '''
WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/
'''

    # The cpu-only build uses ubuntu as the base image, and so the
    # entrypoint files are not available in /opt/nvidia in the base
    # image, so we must provide them ourselves.
    if not enable_gpu:
        df += '''
COPY docker/cpu_only/ /opt/nvidia/
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
'''

    df += '''
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
'''.format(argmap['NVIDIA_BUILD_ID'], argmap['NVIDIA_BUILD_ID'],
           argmap['NVIDIA_BUILD_REF'])

    return df


def create_dockerfile_windows(ddir, dockerfile_name, argmap, backends,
                              repoagents, build_dir):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${{BUILD_IMAGE}} AS tritonserver_build

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${{BASE_IMAGE}}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
LABEL com.nvidia.tritonserver.version="${{TRITON_SERVER_VERSION}}"

RUN setx path "%path%;C:\opt\tritonserver\bin"
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])
    df += '''
WORKDIR /opt/tritonserver
RUN rmdir /S/Q * || exit 0
COPY LICENSE .
COPY TRITON_VERSION .
COPY NVIDIA_Deep_Learning_Container_License.pdf .
COPY --from=tritonserver_build {0}/install/bin bin
COPY --from=tritonserver_build {0}/install/lib/tritonserver.lib lib/
COPY --from=tritonserver_build {0}/install/include/triton/core include/triton/core
'''.format(build_dir)

    for noncore in NONCORE_BACKENDS:
        if noncore in backends:
            df += '''
COPY --from=tritonserver_build {0}/install/backends backends
'''.format(build_dir)
            break

    df += '''
ENTRYPOINT []
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
'''.format(argmap['NVIDIA_BUILD_ID'], argmap['NVIDIA_BUILD_ID'],
           argmap['NVIDIA_BUILD_REF'])

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def container_build(images, backends, repoagents, endpoints):
    # The cmake, build and install directories within the container.
    # Windows uses "\" for the path separator but Docker expects "/"
    # (unix style) separator. We use replace to fix the path for docker usage.
    build_dir = os.path.join(FLAGS.tmp_dir, 'tritonbuild').replace("\\", "/")
    install_dir = os.path.join(build_dir, 'install')
    if target_platform() == 'windows':
        install_dir = os.path.normpath(install_dir)
        cmake_dir = os.path.normpath('c:/workspace')
    else:
        cmake_dir = '/workspace'

    # We can't use docker module for building container because it
    # doesn't stream output and it also seems to handle cache-from
    # incorrectly which leads to excessive rebuilds in the multistage
    # build.
    if 'base' in images:
        base_image = images['base']
    elif target_platform() == 'windows':
        base_image = 'mcr.microsoft.com/dotnet/framework/sdk:4.8'
    elif FLAGS.enable_gpu:
        base_image = 'nvcr.io/nvidia/tritonserver:{}-py3-min'.format(
            FLAGS.upstream_container_version)
    else:
        base_image = 'ubuntu:20.04'

    dockerfileargmap = {
        'NVIDIA_BUILD_REF':
            '' if FLAGS.build_sha is None else FLAGS.build_sha,
        'NVIDIA_BUILD_ID':
            '<unknown>' if FLAGS.build_id is None else FLAGS.build_id,
        'TRITON_VERSION':
            FLAGS.version,
        'TRITON_CONTAINER_VERSION':
            FLAGS.container_version,
        'BASE_IMAGE':
            base_image,
        'DCGM_VERSION':
            '' if FLAGS.version is None or FLAGS.version
            not in TRITON_VERSION_MAP else TRITON_VERSION_MAP[FLAGS.version][5],
    }

    # For cpu-only image we need to copy some cuda libraries and dependencies
    # since we are using a PyTorch container that is not CPU-only
    if not FLAGS.enable_gpu and ('pytorch' in backends) and \
            (target_platform() != 'windows'):
        dockerfileargmap[
            'GPU_BASE_IMAGE'] = 'nvcr.io/nvidia/tritonserver:{}-py3-min'.format(
                FLAGS.upstream_container_version)

    cachefrommap = [
        'tritonserver_buildbase', 'tritonserver_buildbase_cache0',
        'tritonserver_buildbase_cache1'
    ]

    cachefromargs = ['--cache-from={}'.format(k) for k in cachefrommap]
    commonargs = [
        'docker', 'build', '-f',
        os.path.join(FLAGS.build_dir, 'Dockerfile.buildbase')
    ]
    if not FLAGS.no_container_pull:
        commonargs += [
            '--pull',
        ]

    # Windows docker runs in a VM and memory needs to be specified
    # explicitly.
    if target_platform() == 'windows':
        commonargs += ['--memory', FLAGS.container_memory]

    log_verbose('buildbase container {}'.format(commonargs + cachefromargs))
    create_dockerfile_buildbase(FLAGS.build_dir, 'Dockerfile.buildbase',
                                dockerfileargmap)
    try:
        # Create buildbase image, this is an image with all
        # dependencies needed for the build.
        p = subprocess.Popen(commonargs + cachefromargs +
                             ['-t', 'tritonserver_buildbase', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_buildbase failed')

        # Need to extract env from the base image so that we can
        # access library versions.
        buildbase_env_filepath = os.path.join(FLAGS.build_dir, 'buildbase_env')
        with open(buildbase_env_filepath, 'w') as f:
            if target_platform() == 'windows':
                envargs = [
                    'docker', 'run', '--rm', 'tritonserver_buildbase',
                    'cmd.exe', '/c', 'set'
                ]
            else:
                envargs = [
                    'docker', 'run', '--rm', 'tritonserver_buildbase', 'env'
                ]
            log_verbose('buildbase env {}'.format(envargs))
            p = subprocess.Popen(envargs, stdout=f)
            p.wait()
            fail_if(p.returncode != 0,
                    'extracting tritonserver_buildbase env failed')

        buildbase_env = {}
        with open(buildbase_env_filepath, 'r') as f:
            for line in f:
                kv = line.strip().split('=', 1)
                if len(kv) == 2:
                    key, value = kv
                    buildbase_env[key] = value

        # We set the following env in the build docker container
        # launch below to pass necessary versions into the build. By
        # prepending the envvars with TRITONBUILD_ prefix we indicate
        # that the build.py execution within the container should set
        # the corresponding variables in cmake invocation.
        dockerrunenvargs = []
        for k in ['TRT_VERSION', 'DALI_VERSION']:
            if k in buildbase_env:
                dockerrunenvargs += [
                    '--env', 'TRITONBUILD_{}={}'.format(k, buildbase_env[k])
                ]

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
        # --no-container-build is added so that within the buildbase
        # container we just created we do not attempt to do a nested
        # container build
        #
        # Add --version, --container-version and
        # --upstream-container-version flags since they can be set
        # automatically and so may not be in sys.argv
        #
        # --cmake-dir is overridden to 'cmake_dir'
        #
        # --build-dir is added/overridden to 'build_dir'
        #
        # --install-dir is added/overridden to 'install_dir'
        runargs = [
            'python3',
            './build.py',
        ]
        runargs += sys.argv[1:]
        runargs += [
            '--no-container-build',
        ]
        if FLAGS.version is not None:
            runargs += ['--version', FLAGS.version]
        if FLAGS.container_version is not None:
            runargs += ['--container-version', FLAGS.container_version]
        if FLAGS.upstream_container_version is not None:
            runargs += [
                '--upstream-container-version', FLAGS.upstream_container_version
            ]

        runargs += ['--cmake-dir', cmake_dir]
        if target_platform() == 'windows':
            runargs += ['--build-dir', os.path.normpath(build_dir)]
        else:
            runargs += ['--build-dir', build_dir]
        runargs += ['--install-dir', install_dir]

        dockerrunargs = [
            'docker', 'run', '--name', 'tritonserver_builder', '-w',
            '/workspace'
        ]
        if target_platform() == 'windows':
            # Windows docker runs in a VM and memory needs to be
            # specified explicitly.
            dockerrunargs += ['--memory', FLAGS.container_memory]
            dockerrunargs += [
                '-v', '\\\\.\pipe\docker_engine:\\\\.\pipe\docker_engine'
            ]
        else:
            dockerrunargs += ['-v', '/var/run/docker.sock:/var/run/docker.sock']
        dockerrunargs += dockerrunenvargs
        dockerrunargs += [
            'tritonserver_buildbase',
        ]
        dockerrunargs += runargs

        log_verbose(dockerrunargs)
        p = subprocess.Popen(dockerrunargs)
        p.wait()
        fail_if(p.returncode != 0, 'docker run tritonserver_builder failed')

        container = client.containers.get('tritonserver_builder')

        # It is possible to copy the install artifacts from the
        # container at this point (and, for example put them in the
        # specified install directory on the host). But for container
        # build we just want to use the artifacts in the server base
        # container which is created below.
        #mkdir(FLAGS.install_dir)
        #tarfilename = os.path.join(FLAGS.install_dir, 'triton.tar')
        #install_tar, stat_tar = container.get_archive(install_dir)
        #with open(tarfilename, 'wb') as taroutfile:
        #    for d in install_tar:
        #        taroutfile.write(d)
        #untar(FLAGS.install_dir, tarfilename)

        # Build is complete, save the container as the
        # tritonserver_build image. We must do this in two steps:
        #
        #   1. Commit the container as image
        #   "tritonserver_builder_image". This image can't be used
        #   directly because it binds the /var/run/docker.sock mount
        #   and so you would need to always run with that mount
        #   specified... so it can be used this way but very
        #   inconvenient.
        #
        #   2. Perform a docker build to create "tritonserver_build"
        #   from "tritonserver_builder_image" that is essentially
        #   identical but removes the mount.
        try:
            client.images.remove('tritonserver_builder_image', force=True)
        except docker.errors.ImageNotFound:
            pass  # ignore

        container.commit('tritonserver_builder_image', 'latest')
        container.remove(force=True)

        create_dockerfile_build(FLAGS.build_dir, 'Dockerfile.build', backends,
                                build_dir)
        p = subprocess.Popen([
            'docker', 'build', '-t', 'tritonserver_build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile.build'), '.'
        ])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_build failed')

        # Final base image... this is a multi-stage build that uses
        # the install artifacts from the tritonserver_build
        # container.
        if target_platform() == 'windows':
            create_dockerfile_windows(FLAGS.build_dir, 'Dockerfile',
                                      dockerfileargmap, backends, repoagents,
                                      build_dir)
        else:
            create_dockerfile_linux(FLAGS.build_dir, 'Dockerfile',
                                    dockerfileargmap, backends, repoagents,
                                    endpoints, build_dir)
        p = subprocess.Popen([
            'docker', 'build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile')
        ] + ['-t', 'tritonserver', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver failed')

    except Exception as e:
        logging.error(traceback.format_exc())
        fail('container build failed')


def build_backend(be,
                  tag,
                  build_dir,
                  install_dir,
                  github_organization,
                  images,
                  components,
                  library_paths,
                  variant_index=0):
    repo_build_dir = os.path.join(build_dir, be, 'build')
    repo_install_dir = os.path.join(build_dir, be, 'install')

    mkdir(build_dir)
    gitclone(build_dir, backend_repo(be), tag, be, github_organization)
    mkdir(repo_build_dir)
    cmake(
        repo_build_dir,
        backend_cmake_args(images, components, be, repo_install_dir,
                           library_paths, variant_index))
    makeinstall(repo_build_dir)

    backend_install_dir = os.path.join(install_dir, 'backends', be)
    rmdir(backend_install_dir)
    mkdir(backend_install_dir)
    cpdir(os.path.join(repo_install_dir, 'backends', be), backend_install_dir)


def get_tagged_backend(be, version):
    tagged_be = be
    if be == 'openvino':
        if version[0] == 'SPECIFIC':
            tagged_be += "_" + version[1]
        else:
            tagged_be += "_" + version[0].replace('.', '_')
            if version[1] and target_platform() != 'windows':
                tagged_be += "_pre"
    return tagged_be


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

    parser.add_argument('--no-container-build',
                        action="store_true",
                        required=False,
                        help='Do not use Docker container for build.')
    parser.add_argument(
        '--no-container-pull',
        action="store_true",
        required=False,
        help='Do not use Docker --pull argument when building container.')
    parser.add_argument(
        '--container-memory',
        default="8g",
        required=False,
        help='Value for Docker --memory argument. Used only for windows builds.'
    )
    parser.add_argument(
        '--target-platform',
        required=False,
        default=None,
        help=
        'Target platform for build, can be "linux", "windows" or "jetpack". If not specified, build targets the current platform.'
    )
    parser.add_argument(
        '--target-machine',
        required=False,
        default=None,
        help=
        'Target machine/architecture for build. If not specified, build targets the current machine/architecture.'
    )

    parser.add_argument('--build-id',
                        type=str,
                        required=False,
                        help='Build ID associated with the build.')
    parser.add_argument('--build-sha',
                        type=str,
                        required=False,
                        help='SHA associated with the build.')
    parser.add_argument(
        '--build-dir',
        type=str,
        required=True,
        help=
        'Build directory. All repo clones and builds will be performed in this directory.'
    )
    parser.add_argument(
        '--install-dir',
        type=str,
        required=False,
        default=None,
        help='Install directory, default is <builddir>/opt/tritonserver.')
    parser.add_argument(
        '--cmake-dir',
        type=str,
        required=False,
        help='Directory containing the CMakeLists.txt file for Triton server.')
    parser.add_argument(
        '--tmp-dir',
        type=str,
        required=False,
        default='/tmp',
        help=
        'Temporary parent directory used for building inside docker. Default is /tmp.'
    )
    parser.add_argument(
        '--library-paths',
        action='append',
        required=False,
        default=None,
        help=
        'Specify library paths for respective backends in build as <backend-name>[:<library_path>].'
    )
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

    parser.add_argument(
        '--github-organization',
        type=str,
        required=False,
        default='https://github.com/triton-inference-server',
        help=
        'The GitHub organization containing the repos used for the build. Defaults to "https://github.com/triton-inference-server".'
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
        '--container-prebuild-command',
        type=str,
        required=False,
        help=
        'When performing a container build, this command will be executed within the container just before the build it performed.'
    )
    parser.add_argument(
        '--no-container-source',
        action="store_true",
        required=False,
        help='Do not include OSS source code in Docker container.')
    parser.add_argument(
        '--image',
        action='append',
        required=False,
        help=
        'Use specified Docker image in build as <image-name>,<full-image-name>. <image-name> can be "base", "tensorflow1", "tensorflow2", or "pytorch".'
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
    parser.add_argument('--enable-mali-gpu',
                        action="store_true",
                        required=False,
                        help='Enable ARM MALI GPU support.')
    parser.add_argument(
        '--min-compute-capability',
        type=str,
        required=False,
        default='6.0',
        help='Minimum CUDA compute capability supported by server.')

    parser.add_argument(
        '--endpoint',
        action='append',
        required=False,
        help=
        'Include specified endpoint in build. Allowed values are "grpc", "http", "vertex-ai" and "sagemaker".'
    )
    parser.add_argument(
        '--filesystem',
        action='append',
        required=False,
        help=
        'Include specified filesystem in build. Allowed values are "gcs", "azure_storage" and "s3".'
    )
    parser.add_argument(
        '--no-core-build',
        action="store_true",
        required=False,
        help='Do not build Triton core sharead library or executable.')
    parser.add_argument(
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 22.03 -> branch r22.03); otherwise the default <repo-tag> is "main" (e.g. version 22.03dev -> branch main).'
    )
    parser.add_argument(
        '--build-multiple-openvino',
        action="store_true",
        default=False,
        help=
        'Build multiple openVINO versions as specified in TRITON_VERSION_MAP. Be aware that loading backends with different openvino versions simultaneously in triton can cause conflicts'
    )
    parser.add_argument(
        '--repo-tag',
        action='append',
        required=False,
        help=
        'The version of a component to use in the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", "backend" or "thirdparty". If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 22.03 -> branch r22.03); otherwise the default <repo-tag> is "main" (e.g. version 22.03dev -> branch main).'
    )
    parser.add_argument(
        '--repoagent',
        action='append',
        required=False,
        help=
        'Include specified repo agent in build as <repoagent-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 22.03 -> branch r22.03); otherwise the default <repo-tag> is "main" (e.g. version 22.03dev -> branch main).'
    )
    parser.add_argument(
        '--no-force-clone',
        action="store_true",
        default=False,
        help='Do not create fresh clones of repos that have already been cloned.'
    )
    parser.add_argument(
        '--extra-core-cmake-arg',
        action='append',
        required=False,
        help=
        'Extra CMake argument as <name>=<value>. The argument is passed to CMake as -D<name>=<value> and is included after all CMake arguments added by build.py for the core builds.'
    )
    parser.add_argument(
        '--override-core-cmake-arg',
        action='append',
        required=False,
        help=
        'Override specified CMake argument in the build as <name>=<value>. The argument is passed to CMake as -D<name>=<value>. This flag only impacts CMake arguments that are used by build.py. To unconditionally add a CMake argument to the core build use --extra-core-cmake-arg.'
    )
    parser.add_argument(
        '--extra-backend-cmake-arg',
        action='append',
        required=False,
        help=
        'Extra CMake argument for a backend build as <backend>:<name>=<value>. The argument is passed to CMake as -D<name>=<value> and is included after all CMake arguments added by build.py for the backend.'
    )
    parser.add_argument(
        '--override-backend-cmake-arg',
        action='append',
        required=False,
        help=
        'Override specified backend CMake argument in the build as <backend>:<name>=<value>. The argument is passed to CMake as -D<name>=<value>. This flag only impacts CMake arguments that are used by build.py. To unconditionally add a CMake argument to the backend build use --extra-backend-cmake-arg.'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image is None:
        FLAGS.image = []
    if FLAGS.repo_tag is None:
        FLAGS.repo_tag = []
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.endpoint is None:
        FLAGS.endpoint = []
    if FLAGS.filesystem is None:
        FLAGS.filesystem = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []
    if FLAGS.library_paths is None:
        FLAGS.library_paths = []
    if FLAGS.extra_core_cmake_arg is None:
        FLAGS.extra_core_cmake_arg = []
    if FLAGS.override_core_cmake_arg is None:
        FLAGS.override_core_cmake_arg = []
    if FLAGS.override_backend_cmake_arg is None:
        FLAGS.override_backend_cmake_arg = []
    if FLAGS.extra_backend_cmake_arg is None:
        FLAGS.extra_backend_cmake_arg = []

    if FLAGS.install_dir is None:
        FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")

    # FLAGS.cmake_dir defaults to the directory containing build.py.
    if FLAGS.cmake_dir is None:
        from inspect import getsourcefile
        FLAGS.cmake_dir = SCRIPT_DIR

    # Determine the versions. Start with Triton version, if --version
    # is not explicitly specified read from TRITON_VERSION file.
    if FLAGS.version is None:
        with open(os.path.join(SCRIPT_DIR, 'TRITON_VERSION'), "r") as vfile:
            FLAGS.version = vfile.readline().strip()

    log('Building Triton Inference Server')
    log('platform {}'.format(target_platform()))
    log('machine {}'.format(target_machine()))
    log('version {}'.format(FLAGS.version))
    log('cmake dir {}'.format(FLAGS.cmake_dir))
    log('build dir {}'.format(FLAGS.build_dir))
    log('install dir {}'.format(FLAGS.install_dir))

    # Determine the default repo-tag that should be used for images,
    # backends and repo-agents if a repo-tag is not given
    # explicitly. For release branches we use the release branch as
    # the default, otherwise we use 'main'.
    default_repo_tag = 'main'
    cver = FLAGS.container_version
    if cver is None:
        if FLAGS.version not in TRITON_VERSION_MAP:
            fail(
                'unable to determine default repo-tag, container version not known for {}'
                .format(FLAGS.version))
        cver = TRITON_VERSION_MAP[FLAGS.version][0]
    if not cver.endswith('dev'):
        default_repo_tag = 'r' + cver
    log('default repo-tag: {}'.format(default_repo_tag))

    # For other versions use the TRITON_VERSION_MAP unless explicitly
    # given.
    if not FLAGS.no_container_build:
        FLAGS.container_version, FLAGS.upstream_container_version = get_container_versions(
            FLAGS.version, FLAGS.container_version,
            FLAGS.upstream_container_version)

        log('container version {}'.format(FLAGS.container_version))
        log('upstream container version {}'.format(
            FLAGS.upstream_container_version))

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    for be in FLAGS.backend:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        backends[parts[0]] = parts[1]

    # Initialize map of repo agents to build and repo-tag for each.
    repoagents = {}
    for be in FLAGS.repoagent:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('repoagent "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        repoagents[parts[0]] = parts[1]

    # Initialize map of docker images.
    images = {}
    for img in FLAGS.image:
        parts = img.split(',')
        fail_if(
            len(parts) != 2,
            '--image must specify <image-name>,<full-image-registry>')
        fail_if(
            parts[0] not in ['base', 'pytorch', 'tensorflow1', 'tensorflow2'],
            'unsupported value for --image')
        log('image "{}": "{}"'.format(parts[0], parts[1]))
        images[parts[0]] = parts[1]

    # Initialize map of library paths for each backend.
    library_paths = {}
    for lpath in FLAGS.library_paths:
        parts = lpath.split(':')
        if len(parts) == 2:
            log('backend "{}" library path "{}"'.format(parts[0], parts[1]))
            library_paths[parts[0]] = parts[1]

    # Parse any explicitly specified cmake arguments
    for cf in FLAGS.extra_core_cmake_arg:
        parts = cf.split('=')
        fail_if(
            len(parts) != 2,
            '--extra-core-cmake-arg must specify <name>=<value>')
        log('CMake core extra "-D{}={}"'.format(parts[0], parts[1]))
        EXTRA_CORE_CMAKE_FLAGS[parts[0]] = parts[1]

    for cf in FLAGS.override_core_cmake_arg:
        parts = cf.split('=')
        fail_if(
            len(parts) != 2,
            '--override-core-cmake-arg must specify <name>=<value>')
        log('CMake core override "-D{}={}"'.format(parts[0], parts[1]))
        OVERRIDE_CORE_CMAKE_FLAGS[parts[0]] = parts[1]

    for cf in FLAGS.extra_backend_cmake_arg:
        parts = cf.split(':', 1)
        fail_if(
            len(parts) != 2,
            '--extra-backend-cmake-arg must specify <backend>:<name>=<value>')
        be = parts[0]
        parts = parts[1].split('=', 1)
        fail_if(
            len(parts) != 2,
            '--extra-backend-cmake-arg must specify <backend>:<name>=<value>')
        fail_if(
            be not in backends,
            '--extra-backend-cmake-arg specifies backend "{}" which is not included in build'
            .format(be))
        log('backend "{}" CMake extra "-D{}={}"'.format(be, parts[0], parts[1]))
        if be not in EXTRA_BACKEND_CMAKE_FLAGS:
            EXTRA_BACKEND_CMAKE_FLAGS[be] = {}
        EXTRA_BACKEND_CMAKE_FLAGS[be][parts[0]] = parts[1]

    for cf in FLAGS.override_backend_cmake_arg:
        parts = cf.split(':', 1)
        fail_if(
            len(parts) != 2,
            '--override-backend-cmake-arg must specify <backend>:<name>=<value>'
        )
        be = parts[0]
        parts = parts[1].split('=', 1)
        fail_if(
            len(parts) != 2,
            '--override-backend-cmake-arg must specify <backend>:<name>=<value>'
        )
        fail_if(
            be not in backends,
            '--override-backend-cmake-arg specifies backend "{}" which is not included in build'
            .format(be))
        log('backend "{}" CMake override "-D{}={}"'.format(
            be, parts[0], parts[1]))
        if be not in OVERRIDE_BACKEND_CMAKE_FLAGS:
            OVERRIDE_BACKEND_CMAKE_FLAGS[be] = {}
        OVERRIDE_BACKEND_CMAKE_FLAGS[be][parts[0]] = parts[1]

    # If --container-build is specified then we perform the actual
    # build within a build container and then from that create a
    # tritonserver container holding the results of the build.
    if not FLAGS.no_container_build:
        import docker

        container_build(images, backends, repoagents, FLAGS.endpoint)
        sys.exit(0)

    # If there is a container pre-build command assume this invocation
    # is being done within the build container and so run the
    # pre-build command.
    if (FLAGS.container_prebuild_command):
        prebuild_command()

    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2

    # Initialize map of common components and repo-tag for each.
    components = {
        'common': default_repo_tag,
        'core': default_repo_tag,
        'backend': default_repo_tag,
        'thirdparty': default_repo_tag
    }
    for be in FLAGS.repo_tag:
        parts = be.split(':')
        fail_if(
            len(parts) != 2,
            '--repo-tag must specify <component-name>:<repo-tag>')
        fail_if(
            parts[0] not in components,
            '--repo-tag <component-name> must be "common", "core", "backend", or "thirdparty"'
        )
        components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Build the core shared library and the server executable.
    if not FLAGS.no_core_build:
        repo_build_dir = os.path.join(FLAGS.build_dir, 'tritonserver', 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, 'tritonserver',
                                        'install')

        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              core_cmake_args(components, backends, repo_install_dir))
        makeinstall(repo_build_dir)

        core_install_dir = FLAGS.install_dir
        mkdir(core_install_dir)
        cpdir(repo_install_dir, core_install_dir)

    # Build each backend...
    for be in backends:
        # Core backends are not built separately from core so skip...
        if (be in CORE_BACKENDS):
            continue

        tagged_be_list = []
        if (be == 'openvino'):
            tagged_be_list.append(
                get_tagged_backend(be, TRITON_VERSION_MAP[FLAGS.version][4][0]))
            if (FLAGS.build_multiple_openvino):
                skip = True
                for ver in TRITON_VERSION_MAP[FLAGS.version][4]:
                    if not skip:
                        tagged_be_list.append(get_tagged_backend(be, ver))
                    skip = False
        # If armnn_tflite backend, source from external repo for git clone
        if be == 'armnn_tflite':
            github_organization = 'https://gitlab.com/arm-research/smarter/'
        else:
            github_organization = FLAGS.github_organization

        if not tagged_be_list:
            build_backend(be, backends[be], FLAGS.build_dir, FLAGS.install_dir,
                          github_organization, images, components,
                          library_paths)
        else:
            variant_index = 0
            for tagged_be in tagged_be_list:
                build_backend(tagged_be, backends[be], FLAGS.build_dir,
                              FLAGS.install_dir, github_organization, images,
                              components, library_paths, variant_index)
                variant_index += 1

    # Build each repo agent...
    for ra in repoagents:
        repo_build_dir = os.path.join(FLAGS.build_dir, ra, 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, ra, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, repoagent_repo(ra), repoagents[ra], ra,
                 FLAGS.github_organization)
        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              repoagent_cmake_args(images, components, ra, repo_install_dir))
        makeinstall(repo_build_dir)

        repoagent_install_dir = os.path.join(FLAGS.install_dir, 'repoagents',
                                             ra)
        rmdir(repoagent_install_dir)
        mkdir(repoagent_install_dir)
        cpdir(os.path.join(repo_install_dir, 'repoagents', ra),
              repoagent_install_dir)
