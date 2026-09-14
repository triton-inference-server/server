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
"""io: each IOTest method against the exact model repo test.sh built for it.

io_test.py (carried over verbatim) reads its TRIAL global at call time, not
import time, so setting io_test.TRIAL before invoking the method reproduces
test.sh's `export TRIAL=...` without needing to reload the module.
"""

import os

import conftest
import io_test
import pytest

OUTPUT_DIR = conftest.OUTPUT_DIR


def _run(method_name):
    case = io_test.IOTest(method_name)
    case.setUp()
    getattr(case, method_name)()


@pytest.mark.parametrize("trial", ["default", "decoupled"])
def test_ensemble_io(trial, pinned_torch):
    io_test.TRIAL = trial
    model_repository = conftest.ensemble_repo(trial)
    server_log = os.path.join(OUTPUT_DIR, "io_server.ensemble_io.%s.log" % trial)
    with conftest.serve(model_repository, server_log):
        _run("test_ensemble_io")


def test_empty_gpu_output(pinned_torch):
    model_repository = conftest.empty_gpu_output_repo()
    server_log = os.path.join(OUTPUT_DIR, "io_server.empty_gpu_output.log")
    with conftest.serve(model_repository, server_log):
        _run("test_empty_gpu_output")


def test_variable_gpu_output(pinned_torch):
    model_repository = conftest.variable_gpu_output_repo()
    server_log = os.path.join(OUTPUT_DIR, "io_server.variable_gpu_output.log")
    with conftest.serve(model_repository, server_log):
        _run("test_variable_gpu_output")


def test_requested_output(pinned_torch):
    """test_requested_output_default and _decoupled share one server, as in test.sh."""
    model_repository = conftest.requested_output_repo()
    server_log = os.path.join(OUTPUT_DIR, "io_server.requested_output.log")
    with conftest.serve(model_repository, server_log):
        _run("test_requested_output_default")
        _run("test_requested_output_decoupled")


def test_requested_output_decoupled_prior_crash(pinned_torch):
    model_repository = conftest.requested_output_decoupled_prior_crash_repo()
    server_log = os.path.join(
        OUTPUT_DIR, "io_server.requested_output_decoupled_prior_crash.log"
    )
    with conftest.serve(model_repository, server_log):
        _run("test_requested_output_decoupled_prior_crash")
