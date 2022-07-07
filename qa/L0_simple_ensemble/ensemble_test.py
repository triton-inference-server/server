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

import sys

sys.path.append("../common")
sys.path.append("../clients")

import logging

import os
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import tritonhttpclient


class EnsembleTest(tu.TestResultCollector):

    def _get_infer_count_per_version(self, model_name):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000",
                                                               verbose=True)
        stats = triton_client.get_inference_statistics(model_name)
        self.assertEqual(len(stats["model_stats"]), 2)
        infer_count = [0, 0]
        for model_stat in stats["model_stats"]:
            self.assertEqual(model_stat["name"], model_name,
                             "expected stats for model " + model_name)
            model_version = model_stat['version']
            if model_version == "1":
                infer_count[0] = model_stat["inference_stats"]["success"][
                    "count"]
            elif model_version == "2":
                infer_count[1] = model_stat["inference_stats"]["success"][
                    "count"]
            else:
                self.assertTrue(
                    False, "unexpected version {} for model {}".format(
                        model_version, model_name))
        return infer_count

    def test_ensemble_add_sub(self):
        for bs in (1, 8):
            iu.infer_exact(self, "ensemble_add_sub", (bs, 16), bs, np.int32,
                           np.int32, np.int32)

        infer_count = self._get_infer_count_per_version("simple")
        # The two 'simple' versions should have the same infer count
        if (infer_count[0] != infer_count[1]):
            self.assertTrue(
                False,
                "unexpeced different infer count for different 'simple' versions"
            )

    def test_ensemble_add_sub_one_output(self):
        for bs in (1, 8):
            iu.infer_exact(self,
                           "ensemble_add_sub", (bs, 16),
                           bs,
                           np.int32,
                           np.int32,
                           np.int32,
                           outputs=("OUTPUT0",))

        infer_count = self._get_infer_count_per_version("simple")
        # Only 'simple' version 2 should have non-zero infer count
        # as it is in charge of producing OUTPUT0
        if (infer_count[0] != 0):
            self.assertTrue(
                False, "unexpeced non-zero infer count for 'simple' version 1")
        elif (infer_count[1] == 0):
            self.assertTrue(
                False, "unexpeced zero infer count for 'simple' version 2")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
