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


"""Two instances need one gpu_device_ids entry each: the model must fail to
load, and say why, when the parameter is missing or too short."""

import os

import pytest
from conftest import OUTPUT_DIR, ServerStartError, prepare_repository, serve

CASES = {
    "no_gpu_ids": (None, "gpu_device_ids parameter required"),
    "short_gpu_ids": ("0", "gpu_device_ids mismatch"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_is_rejected(case):
    gpu_device_ids, expected_error = CASES[case]
    repo = prepare_repository(
        os.path.join(OUTPUT_DIR, "models_%s" % case), gpu_device_ids=gpu_device_ids
    )
    server_log = os.path.join(OUTPUT_DIR, "%s_server.log" % case)
    with pytest.raises(ServerStartError):
        with serve(repo, server_log):
            pass
    with open(server_log, errors="replace") as f:
        assert expected_error in f.read()
