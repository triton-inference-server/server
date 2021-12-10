# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


def generate_compressed_data():
    with open("raw_data", "rb") as f:
        import zlib
        import gzip
        raw_data = f.read()
        with open("deflate_compressed_data", "wb") as of:
            of.write(zlib.compress(raw_data))
        with open("gzip_compressed_data", "wb") as of:
            of.write(gzip.compress(raw_data))


def validate_compressed_data():
    with open("raw_data", "rb") as f:
        import zlib
        import gzip
        raw_data = f.read()
        with open("generated_deflate_compressed_data", "rb") as cf:
            decompressed_data = zlib.decompress(cf.read())
            if decompressed_data != raw_data:
                exit(1)
        with open("generated_gzip_compressed_data", "rb") as cf:
            decompressed_data = gzip.decompress(cf.read())
            if decompressed_data != raw_data:
                exit(1)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
