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


"""Unary requests through the non-decoupled ensemble."""

import llmapi_test_util as util
import pytest
import requests
import tritonclient.grpc as grpcclient
from ensemble_util import ENSEMBLE_NAME, PREFIX, QUESTION, ensemble_inputs

pytestmark = pytest.mark.usefixtures("unary_ensemble")


def test_http_generate():
    response = requests.post(
        f"{util.HTTP_URL}/v2/models/{ENSEMBLE_NAME}/generate",
        json={"QUESTION": QUESTION, "max_tokens": 16},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    text = response.json()["text_output"]
    assert text.startswith(PREFIX + QUESTION), text
    assert len(text) > len(PREFIX + QUESTION), "no tokens were generated"


def test_grpc_infer():
    with grpcclient.InferenceServerClient(util.GRPC_URL) as client:
        result = client.infer(ENSEMBLE_NAME, ensemble_inputs(), client_timeout=120)
    text = util.output_text(result)
    assert text.startswith(PREFIX + QUESTION), text
    assert len(text) > len(PREFIX + QUESTION), "no tokens were generated"


def test_every_step_counts_the_request():
    models = (ENSEMBLE_NAME, "prompt_template", util.MODEL_NAME)
    before = {m: util.metric_value("nv_inference_request_success", m) for m in models}
    with grpcclient.InferenceServerClient(util.GRPC_URL) as client:
        client.infer(ENSEMBLE_NAME, ensemble_inputs(max_tokens=2), client_timeout=120)
    after = {m: util.metric_value("nv_inference_request_success", m) for m in models}
    assert {m: after[m] - before[m] for m in models} == {m: 1 for m in models}
