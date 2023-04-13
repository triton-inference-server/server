# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
import tritonclient.grpc as grpcclient
from models.instance_init_del.util import get_initialize_count, reset_initialize_count, get_finalize_count, reset_finalize_count, update_instance_group


class Test(unittest.TestCase):

    def test_instance_update(self):
        # Initialize client
        triton = grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=True)
        # Set instance group and reset counters
        group_setting = ("  {\n"
                         "    count: 3\n"
                         "    kind: KIND_CPU\n"
                         "  }\n")
        update_instance_group(group_setting)
        reset_initialize_count()
        reset_finalize_count()
        # Load model
        triton.load_model("instance_init_del")
        # Check instance counters
        self.assertEqual(get_initialize_count(), 3)
        self.assertEqual(get_finalize_count(), 0)
        # Remove 1 instance and load change
        group_setting = ("  {\n"
                         "    count: 2\n"
                         "    kind: KIND_CPU\n"
                         "  }\n")
        update_instance_group(group_setting)
        triton.load_model("instance_init_del")
        # Check instance counters
        self.assertEqual(get_initialize_count(), 3)
        self.assertEqual(get_finalize_count(), 1)


if __name__ == "__main__":
    unittest.main()
