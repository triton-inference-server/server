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
"""bls_request_rescheduling's own internal unittest, inlined from
qa/L0_backend_python/test_infer_shm_leak.py::TestInferShmLeak.test_shm_leak
the same way ../../bls/tests/test_bls.py inlines it.
"""

import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "common",
    )
)
import shm_util  # noqa: E402
import tritonclient.grpc as grpcclient  # noqa: E402
from conftest import GRPC_PORT, TRITONSERVER_IPADDR  # noqa: E402

MODEL_NAME = "bls_request_rescheduling"


def test_bls_request_rescheduling_self_test():
    with grpcclient.InferenceServerClient(
        "%s:%d" % (TRITONSERVER_IPADDR, GRPC_PORT)
    ) as client:
        detector = shm_util.ShmLeakDetector()
        with detector.Probe():
            result = client.infer(MODEL_NAME, [], client_timeout=240)
            output0 = result.as_numpy("OUTPUT0")
            assert output0 == [1], "python_unittest failed for model %s" % MODEL_NAME
