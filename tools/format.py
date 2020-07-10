#!/usr/bin/python

# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
import yapf

FLAGS = None
FORMAT_EXTS = ('proto', 'cc', 'cu', 'h')
SKIP_PATHS = ('tools', )


def visit(path):
    if FLAGS.verbose:
        print("visiting " + path)

    valid_ext = False
    python_file = False
    for ext in FORMAT_EXTS:
        if path.endswith('.' + ext):
            valid_ext = True
            break
    if path.endswith('.py'):
        valid_ext = True
        python_file = True
    if not valid_ext:
        if FLAGS.verbose:
            print("skipping due to extension: " + path)
        return True

    for skip in SKIP_PATHS:
        if path.startswith(skip):
            if FLAGS.verbose:
                print("skipping due to path prefix: " + path)
            return True
    if python_file:
        yapf.yapflib.yapf_api.FormatFile(path, in_place=True)
        return True
    else:
        args = ['clang-format-6.0', '--style=file', '-i']
        if FLAGS.verbose:
            args.append('-verbose')
        args.append(path)

        ret = subprocess.call(args)
        if ret != 0:
            print("format failed for " + path)
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('paths',
                        type=str,
                        nargs='*',
                        default=None,
                        help='Directories or files to format')
    FLAGS = parser.parse_args()

    # Check the version of yapf. Needs a consistent version
    # of yapf to prevent unneccessary changes in the code.
    if (yapf.__version__ != '0.30.0'):
        print("Needs yapf 0.30.0, but got yapf {}".format(yapf.__version__))

    if (FLAGS.paths is None) or (len(FLAGS.paths) == 0):
        parser.print_help()
        exit(1)

    ret = True
    for path in FLAGS.paths:
        if not os.path.isdir(path):
            if not visit(path):
                ret = False
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    if not visit(os.path.join(root, name)):
                        ret = False

    exit(0 if ret else 1)
