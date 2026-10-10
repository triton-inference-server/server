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
"""bls: the BLS chaining matrix (non_decoupled/decoupled x 64/256 MB pool).

Each (pool_mb, trial, model_name) combination gets its own tritonserver
process -- one test, one server, fully isolated -- rather than the original
test.sh's one-server-serves-four-models loop. This is the real behavioural
difference from the bash version: 16 server starts instead of 4, in exchange
for JUnit results and failures that are attributable to exactly one model
instead of a shared server run.

Each MODEL_NAME's model.py runs its own internal unittest server-side and
reports pass/fail as OUTPUT0 == [1] / [0] over one inference call -- this is
qa/L0_backend_python/test_infer_shm_leak.py's TestInferShmLeak.test_shm_leak,
inlined here so this scenario needs nothing outside its own tests/ and the
shared qa/common and qa/python_models assets.
"""

import os
import sys

import pytest
import tritonclient.grpc as grpcclient
from conftest import GRPC_PORT, OUTPUT_DIR, TRITONSERVER_IPADDR, serve

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "common",
    )
)
import shm_util  # noqa: E402

MODEL_NAMES = ["bls", "bls_memory", "bls_memory_async", "bls_async"]
POOL_SIZES_MB = [64, 256]
TRIALS = ["non_decoupled", "decoupled"]

# See qa/L0_backend_python/test_infer_shm_leak.py: a 480-byte leak in the
# bls sub-test is a known, accepted condition, not a regression.
_KNOWN_LEAK = "Known shared memory leak of 480 bytes detected"


def _run_self_test(model_name):
    with grpcclient.InferenceServerClient(
        "%s:%d" % (TRITONSERVER_IPADDR, GRPC_PORT)
    ) as client:
        result = client.infer(model_name, [], client_timeout=240)
        output0 = result.as_numpy("OUTPUT0")
        assert output0 == [1], "python_unittest failed for model %s" % model_name


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("trial", TRIALS)
@pytest.mark.parametrize("pool_mb", POOL_SIZES_MB)
def test_bls_matrix(model_repository, pool_mb, trial, model_name):
    pool_bytes = pool_mb * 1024 * 1024
    server_log = os.path.join(
        OUTPUT_DIR,
        "bls_%s_%s.%dmb.server.log" % (model_name, trial, pool_mb),
    )
    extra_args = ["--cuda-memory-pool-byte-size=0:%d" % pool_bytes]

    with serve(
        model_repository, server_log, bls_kind=trial, extra_args=extra_args
    ) as _:
        detector = shm_util.ShmLeakDetector()
        try:
            with detector.Probe():
                _run_self_test(model_name)
        except AssertionError as e:
            if _KNOWN_LEAK in str(e):
                pytest.xfail(str(e))
            raise

    with open(server_log) as f:
        log_text = f.read()

    if model_name == "bls":
        assert (
            "Request timeout: 11000000000" in log_text
        ), "BLS timeout value not correctly passed to model"

    if pool_mb == 256:
        assert "Failed to allocate memory from CUDA memory pool" not in log_text, (
            "expected to use the CUDA memory pool for all requests at "
            "256 MB (pool size %d, trial %s)" % (pool_mb, trial)
        )
