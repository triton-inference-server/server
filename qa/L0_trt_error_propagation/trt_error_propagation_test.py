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
import shutil
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TestTrtErrorPropagation(unittest.TestCase):

    def setUp(self):
        # Initialize client
        self.__triton = grpcclient.InferenceServerClient("localhost:8001",
                                                         verbose=True)

    def test_invalid_trt_model(self):
        model_name = "plan_zero_1_float32"
        model_file_path = "models/" + model_name + "/1/model.plan"
        # Invalidate model file
        shutil.move(model_file_path, "model.plan.backup")
        with open(model_file_path, mode="w") as f:
            f.write("----- invalid model.plan -----\n")
        # Try loading the invalid model
        with self.assertRaises(InferenceServerException) as e:
            self.__triton.load_model(model_name)
        err_msg = str(e.exception)
        self.assertTrue("Internal: unable to create TensorRT engine" in err_msg,
                        "Caught an unexpected exception")
        self.assertTrue(
            "Error Code 4: Internal Error (Engine deserialization failed.)"
            in err_msg,
            "Detailed error message not propagated back to triton client")
        # Restore model file
        shutil.move("model.plan.backup", model_file_path)


if __name__ == '__main__':
    unittest.main()
