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

import os
import pathlib

import numpy as np
import tritonclient.http as httpclient

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class TestCustomMetricsReload:
    def _infer(self, client, model_name):
        result = client.infer(model_name, [], client_timeout=240)
        output0 = result.as_numpy("OUTPUT0")
        assert np.array_equal(output0, np.array([1.0], dtype=np.float32))

    def test_histogram_metric_survives_model_reload(self):
        model_name = "custom_metrics_reload"
        model_path = pathlib.Path("models") / model_name / "1" / "model.py"

        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            assert client.is_model_ready(model_name)

            for _ in range(6):
                self._infer(client, model_name)

            os.utime(model_path)
            client.load_model(model_name)
            assert client.is_model_ready(model_name)

            for _ in range(6):
                self._infer(client, model_name)
