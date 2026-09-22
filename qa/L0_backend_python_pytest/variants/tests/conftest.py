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
"""Shared paths for the `variants` scenario.

Unlike every other scenario, `variants` (ported from
qa/L0_backend_python/variants/test.sh) never starts a tritonserver process
or does inference -- it is a from-source, CPU-only (-DTRITON_ENABLE_GPU=OFF)
build of python_backend, asserting only that the build succeeds. No model
repository, no server lifecycle: see test_variants.py.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.dirname(HERE)

OUTPUT_DIR = os.path.join(SCENARIO_DIR, "output")

TRITON_REPO_ORGANIZATION = os.environ.get(
    "TRITON_REPO_ORGANIZATION", "https://github.com/triton-inference-server"
)
PYTHON_BACKEND_REPO_TAG = os.environ.get("PYTHON_BACKEND_REPO_TAG", "main")
TRITON_BACKEND_REPO_TAG = os.environ.get("TRITON_BACKEND_REPO_TAG", "main")
TRITON_COMMON_REPO_TAG = os.environ.get("TRITON_COMMON_REPO_TAG", "main")
TRITON_CORE_REPO_TAG = os.environ.get("TRITON_CORE_REPO_TAG", "main")
