#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Explicit model control: unloading must release the engine, reloading must work."""

import subprocess
import unittest

import llmapi_test_util as util
import tritonclient.grpc as grpcclient

PROMPT = "The capital of France is Paris. The capital of Germany is"


def processes(pattern):
    """PIDs of live processes whose command line matches `pattern`."""
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return out.stdout.split()


class LlmapiModelControlTest(unittest.TestCase):
    def setUp(self):
        self.client = grpcclient.InferenceServerClient(util.GRPC_URL)

    def tearDown(self):
        self.client.close()

    def _infer(self):
        result = util.grpc_infer(PROMPT, sampling_param_max_tokens=4)
        self.assertTrue(util.output_text(result))

    def test_unload_releases_resources_and_reload_serves(self):
        self.assertTrue(self.client.is_model_ready(util.MODEL_NAME))
        self._infer()
        loaded_mib = util.gpu_memory_used_mib()

        self.client.unload_model(util.MODEL_NAME)
        self.assertTrue(
            util.wait_until(
                lambda: not self.client.is_model_ready(util.MODEL_NAME), timeout=120
            ),
            "the model is still ready after unload",
        )
        # The LLM API owns the GPU memory in worker processes it spawns; they
        # must exit with the model, or the next load cannot get the memory.
        self.assertTrue(
            util.wait_until(
                lambda: loaded_mib - util.gpu_memory_used_mib() > 4096, timeout=60
            ),
            "GPU memory was not released after unload",
        )
        # Shutdown is asynchronous: the workers exit shortly after the model
        # reports unloaded, so wait for them rather than check at once.
        for pattern in ("mpi4py.futures", "triton_python_backend_stub"):
            self.assertTrue(
                util.wait_until(lambda: not processes(pattern), timeout=30),
                f"processes matching {pattern} survived the unload: "
                f"{processes(pattern)}",
            )

        self.client.load_model(util.MODEL_NAME)
        self.assertTrue(self.client.is_model_ready(util.MODEL_NAME))
        self._infer()

    def test_repeated_reload(self):
        for _ in range(2):
            self.client.unload_model(util.MODEL_NAME)
            self.assertTrue(
                util.wait_until(
                    lambda: not self.client.is_model_ready(util.MODEL_NAME), timeout=120
                )
            )
            self.client.load_model(util.MODEL_NAME)
            self._infer()


if __name__ == "__main__":
    unittest.main()
