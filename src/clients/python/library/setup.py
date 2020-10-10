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

import os
import sys

from setuptools import find_packages
from setuptools import setup
from itertools import chain

if "--plat-name" in sys.argv:
    PLATFORM_FLAG = sys.argv[sys.argv.index("--plat-name") + 1]
else:
    PLATFORM_FLAG = 'any'

if 'VERSION' not in os.environ:
    raise Exception('envvar VERSION must be specified')

VERSION = os.environ['VERSION']

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            pyver, abi, plat = 'py3', 'none', PLATFORM_FLAG
            return pyver, abi, plat
except ImportError:
    bdist_wheel = None

this_directory = os.path.abspath(os.path.dirname(__file__))

def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.startswith("#")]


install_requires = req_file("requirements.txt")
extras_require = {
    'grpc': req_file("requirements_grpc.txt"),
    'http': req_file("requirements_http.txt"),
}

extras_require['all'] = list(chain(extras_require.values()))

platform_package_data = []
if PLATFORM_FLAG != 'any':
    platform_package_data += ['libcshm.so']
    if bool(os.environ.get('CUDA_VERSION', 0)):
        platform_package_data += ['libccudashm.so']

data_files = [
    ("", ["LICENSE.txt"]),
]
if PLATFORM_FLAG != 'any':
    data_files += [("bin", ["perf_analyzer", "perf_client"])]

setup(
    name='tritonclient',
    version=VERSION,
    author='NVIDIA Inc.',
    author_email='sw-dl-triton@nvidia.com',
    description=
    "Python client library and utilities for communicating with Triton Inference Server",
    license='BSD',
    url='https://developer.nvidia.com/nvidia-triton-inference-server',
    keywords=[
        'grpc', 'http', 'triton', 'tensorrt', 'inference', 'server', 'service',
        'client', 'nvidia'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    package_data={
        '': platform_package_data,
    },
    zip_safe=False,
    cmdclass={'bdist_wheel': bdist_wheel},
    data_files=data_files,
)
