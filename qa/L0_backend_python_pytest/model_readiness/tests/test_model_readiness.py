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
"""model_readiness: signal-kill readiness + user-defined is_ready() function.

test_model_readiness_client.py is carried over verbatim as the assertion
source; this module drives it with the same per-phase server lifecycle
test.sh used.
"""

import os
import time

import conftest
import pytest
import test_model_readiness_client as client
from conftest import action_log

OUTPUT_DIR = conftest.OUTPUT_DIR
MODEL_NAME = "identity_fp32"


def _run(case_cls, method_name):
    case = case_cls(method_name)
    if hasattr(case, "setUp"):
        case.setUp()
    getattr(case, method_name)()


@pytest.mark.parametrize("signal_num", [11, 9])
def test_model_readiness_after_stub_signal(signal_num):
    model_repository = conftest.identity_model_repository()
    server_log = os.path.join(
        OUTPUT_DIR, "model_readiness_signal_%d_server.log" % signal_num
    )

    with conftest.serve(model_repository, server_log):
        _run(client.TestModelReadiness, "test_model_ready")

        stub_pid = action_log.run(
            ["pgrep", "-f", "triton_python_backend_stub"],
            "Finding the running triton_python_backend_stub PID",
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert stub_pid, "could not find triton_python_backend_stub process"

        action_log.run(
            ["kill", "-%d" % signal_num, stub_pid],
            "Sending signal %d to stub PID %s to trigger readiness recovery"
            % (signal_num, stub_pid),
            check=True,
        )
        time.sleep(1)

        _run(client.TestModelReadiness, "test_model_not_ready")

    with open(server_log) as f:
        server_log_text = f.read()
    expected = (
        "Model '%s' version 1 is not ready: Stub process '%s_0_0' is not healthy."
        % (MODEL_NAME, MODEL_NAME)
    )
    # Expect 2 occurrences: HTTP and gRPC readiness checks.
    assert (
        server_log_text.count(expected) == 2
    ), "expected 2 occurrences of %r in %s, found %d" % (
        expected,
        server_log,
        server_log_text.count(expected),
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "test_is_ready_coroutine_returns_true",
        "test_is_ready_returns_true",
        "test_is_ready_returns_false",
        "test_is_ready_raises_exception",
        "test_is_ready_returns_non_boolean",
        "test_is_ready_takes_long_time",
        "test_multiple_concurrent_ready_and_infer_requests",
        "test_multiple_concurrent_ready_and_infer_requests_decoupled",
    ],
)
def test_user_defined_model_readiness_function(method_name):
    model_repository = conftest.readiness_fn_model_repository()
    server_log = os.path.join(
        OUTPUT_DIR,
        "test_user_defined_model_readiness_function.%s.server.log" % method_name,
    )
    with conftest.serve(
        model_repository, server_log, extra_args=["--strict-readiness=false"]
    ):
        _run(client.TestUserDefinedModelReadinessFunction, method_name)
