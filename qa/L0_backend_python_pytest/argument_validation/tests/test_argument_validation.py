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
"""argument_validation: the `argument_validation` model's internal unittest.

The model.py runs its own unittest server-side and reports pass/fail as
OUTPUT0 == [1] / [0] over one inference call -- same self-test convention as
qa/L0_backend_python/test_infer_shm_leak.py, inlined here (as in `bls`) so
this scenario needs nothing outside its own tests/ and the shared qa/common
asset.
"""

import os
import sys

import pytest
import tritonclient.grpc as grpcclient
from conftest import GRPC_PORT, MODEL_NAME, OUTPUT_DIR, TRITONSERVER_IPADDR, serve

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "common",
    )
)
import shm_util  # noqa: E402

_KNOWN_LEAK = "Known shared memory leak of 480 bytes detected"


def test_argument_validation(model_repository):
    server_log = os.path.join(OUTPUT_DIR, "argument_validation_server.log")
    with serve(model_repository, server_log):
        detector = shm_util.ShmLeakDetector()
        try:
            with detector.Probe():
                with grpcclient.InferenceServerClient(
                    "%s:%d" % (TRITONSERVER_IPADDR, GRPC_PORT)
                ) as client:
                    result = client.infer(MODEL_NAME, [], client_timeout=240)
                    output0 = result.as_numpy("OUTPUT0")
                    assert output0 == [1], (
                        "python_unittest failed for model %s" % MODEL_NAME
                    )
        except AssertionError as e:
            if _KNOWN_LEAK in str(e):
                pytest.xfail(str(e))
            raise
