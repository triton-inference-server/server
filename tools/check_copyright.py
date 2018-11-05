#!/usr/bin/python

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import re

FLAGS = None
SKIP_EXTS = ('jpeg', 'jpg', 'pgm', 'png',
             'log', 'serverlog',
             'preprocessed', 'jmx', 'gz',
             'caffemodel', 'prototxt')
SKIP_PATHS = ('docs/examples/model_repository',
              'serving',
              'src/servables/caffe2/testdata',
              'src/servables/tensorflow/testdata',
              'src/servables/tensorrt/testdata',
              'src/test/testdata',
              'tools/patch',
              'VERSION')

COPYRIGHT ='''
Copyright (c) YYYY, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

copyright_list = [l.rstrip() for i, l in enumerate(COPYRIGHT.splitlines()) if i > 0]

def visit(path):
    if FLAGS.verbose:
        print("visiting " + path)

    for skip in SKIP_EXTS:
        if path.endswith('.' + skip):
            if FLAGS.verbose:
                print("skipping due to extension: " + path)
            return True

    for skip in SKIP_PATHS:
        if path.startswith(skip):
            if FLAGS.verbose:
                print("skipping due to path prefix: " + path)
            return True

    with open(path, 'r') as f:
        first_line = True
        line = None
        try:
            for fline in f:
                line = fline

                # Skip any '#!' or '..' (from rst) lines at the start
                # of the file
                if first_line:
                    first_line = False
                    if fline.startswith("#!") or fline.startswith("..") or fline.startswith("<!--"):
                        continue

                # Skip empty lines...
                if len(fline.strip()) != 0:
                    break
        except UnicodeDecodeError as ex:
            # If we get this exception on the first line then assume a
            # non-text file.
            if not first_line:
                raise ex
            if FLAGS.verbose:
                print("skipping binary file: " + path)
            return True

        if line is None:
            if FLAGS.verbose:
                print("skipping empty file: " + path)
            return True

        line = line.strip()

        # The next line must be the copyright line with the expected
        # year. It must start with either '#' or '//'
        prefix = None
        if line.startswith('#'):
            prefix = '#'
        elif line.startswith('//'):
            prefix = '//'
        else:
            print("incorrect prefix for copyright line, expecting '#' or '//', for " +
                  path + ": " + line)
            return False

        expected_copyright = (prefix + " " + copyright_list[0])
        if line != expected_copyright:
            print("incorrect copyright for " + path)
            print("  expected: " + expected_copyright)
            print("       got: " + line)
            return False

        # Subsequent lines must match the copyright body.
        copyright_idx = 1
        for line in f:
            if copyright_idx >= len(copyright_list):
                break

            line = line.strip()
            if len(copyright_list[copyright_idx]) == 0:
                expected = prefix
            else:
                expected = (prefix + " " + copyright_list[copyright_idx])
            if line != expected:
                print("incorrect copyright body for " + path)
                print("  expected: '" + expected + "'")
                print("       got: '" + line + "'")
                return False
            copyright_idx += 1

        if copyright_idx != len(copyright_list):
            print("missing " + str(len(copyright_list) - copyright_idx) +
                  " lines of the copyright body")
            return False

    if FLAGS.verbose:
        print("copyright correct for " + path)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-y', '--year', type=str, required=True,
                        help='Copyright year')
    parser.add_argument('paths', type=str, nargs='*', default=None,
                        help='Directories or files to check')
    FLAGS = parser.parse_args()

    if FLAGS.paths is None or len(FLAGS.paths) == 0:
        parser.print_help()
        exit(1)

    copyright_list[0] = copyright_list[0].replace('YYYY', str(FLAGS.year))

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
