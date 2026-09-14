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
"""logging: identity_fp32_logging under default / verbose / disabled settings.

Ported 1:1 from qa/L0_backend_python/logging/test.sh -- same three server
runs, same expected log-line counts. Each test owns its own server process
so a failure in one setting can't leave stale log lines for the next.
"""

import json
import os
import urllib.request

import numpy as np
import tritonclient.http as httpclient
from conftest import MODEL_NAME, OUTPUT_DIR, log_counts, serve
from tritonclient.utils import np_to_triton_dtype


def _infer(base):
    with httpclient.InferenceServerClient(base.split("://")[-1]) as client:
        input_data = np.array([[1.0]], dtype=np.float32)
        inputs = [
            httpclient.InferInput(
                "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        result = client.infer(MODEL_NAME, inputs)
        output0 = result.as_numpy("OUTPUT0")
        assert output0 is not None
        assert np.all(output0 == input_data)


def test_logging_default(model_repository):
    server_log = os.path.join(OUTPUT_DIR, "logging_server.default.log")
    with serve(model_repository, server_log) as base:
        _infer(base)
    counts = log_counts(server_log)
    assert counts == {
        "specific": 4,
        "info": 4,
        "warning": 4,
        "error": 4,
        "verbose": 0,
    }


def test_logging_verbose(model_repository):
    server_log = os.path.join(OUTPUT_DIR, "logging_server.verbose.log")
    with serve(model_repository, server_log) as base:
        req = urllib.request.Request(
            base + "/v2/logging",
            data=json.dumps({"log_verbose_level": 1}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.status == 200
        _infer(base)
    counts = log_counts(server_log)
    # Verbose is only 3 (not 5): the model must initialize before log
    # settings can be changed, so Initialize's verbose line predates
    # verbose logging being turned on and is never counted.
    assert counts == {
        "specific": 4,
        "info": 4,
        "warning": 4,
        "error": 4,
        "verbose": 3,
    }


def test_logging_disabled(model_repository):
    server_log = os.path.join(OUTPUT_DIR, "logging_server.disabled.log")
    with serve(model_repository, server_log) as base:
        for param in ("log_info", "log_warning", "log_error"):
            req = urllib.request.Request(
                base + "/v2/logging",
                data=json.dumps({param: False}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.status == 200
        _infer(base)
    counts = log_counts(server_log)
    # Only Initialize's lines land (1 each): logging is disabled before
    # Execute/Finalize run, and the model must initialize before the
    # settings request above can even be served.
    assert counts == {
        "specific": 1,
        "info": 1,
        "warning": 1,
        "error": 1,
        "verbose": 0,
    }
