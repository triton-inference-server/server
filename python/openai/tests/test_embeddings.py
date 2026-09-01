# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import os
from pathlib import Path

import numpy as np
import pytest

# Results on A6000 GPU. The results vary slightly across GPU models.
EMBEDDING_OUTPUT_FLOAT = [
    -0.0365707763,
    0.076415509,
    0.0111756483,
    0.0361226462,
    -0.0895985439,
    0.000943024294,
    0.0876693726,
    -0.0594895333,
    0.0349566936,
    -0.0937493294,
    0.132199496,
    -0.000293674122,
    0.0232867915,
    -0.022317145,
    0.00537066581,
    -0.0994819254,
    0.0873948932,
    -0.0594052449,
    0.0255441833,
    -0.0896684974,
    -0.05285028,
    0.0055408217,
    -0.0262292251,
    -0.0356511623,
    0.00121368305,
    0.0322951674,
    0.0308073629,
    0.0160015952,
    -0.0187645368,
    -0.00243343855,
    -0.031794291,
    -0.00363291171,
    -0.0096142441,
    0.00763130654,
    -0.0644499287,
    -0.00359453214,
    0.0715369731,
    -0.0835151896,
    0.0429252461,
    -0.0022872088,
    0.0263325982,
    -0.0283530336,
    0.0369597636,
    0.0226341616,
    0.0176856909,
    -0.0551657975,
    -0.0439660996,
    -0.0230927691,
    0.0370879583,
    -0.00998868514,
    -0.0479186773,
    0.0299891923,
    -0.0363422334,
    -0.0903369784,
    0.0643197298,
    0.0223918613,
    0.0467988774,
    -0.0922083259,
    -0.0229205787,
    0.0192605518,
    -0.0187863987,
    -0.00113048754,
    0.0447385423,
    0.0520578064,
    0.0455449931,
    -0.0587379821,
    -0.0332143232,
    -0.0847064033,
    -0.0296805203,
    0.0522436313,
    0.0216157753,
    -0.0455714688,
    0.0491720773,
    -0.0230966564,
    -0.0275325552,
    -0.0460045785,
    0.000419082266,
    -0.0763081461,
    -0.106194891,
    0.0145287318,
    -0.0292449873,
    -0.0782747194,
    -0.0399451442,
    0.0388121083,
    -0.0387705714,
    -0.0696005225,
    0.0101006646,
    -0.0475178808,
    0.021968415,
    0.0317323506,
    0.0508495905,
    -0.0438993014,
    -0.0387620702,
    -0.0523114018,
    0.0394089296,
    -0.0469380654,
    -0.0283870418,
    -0.023771964,
    0.0526082292,
    0.079620786,
    -0.00229133829,
    0.0339719504,
    -0.0288904961,
    -0.00747278007,
    -0.00687310379,
    0.0226157606,
    -0.0149615603,
    0.0292317756,
    -0.0220453553,
    -0.00164654257,
    0.00224191439,
    0.0798983052,
    0.0343615711,
    0.0089583965,
    -0.0600032806,
    -0.0543279201,
    0.00543989427,
    -0.0362421572,
    -0.0285114087,
    0.0446144193,
    0.0502739027,
    0.0574940592,
    -0.0596095882,
    0.0327727199,
    -0.117024891,
    -0.0314645469,
    0.136668995,
    0.0,
    -0.0752000064,
    -0.0291419327,
    0.0460730754,
    -0.0266116355,
    0.130121678,
    0.0257908553,
    -0.035819497,
    0.0506630391,
    -0.0240160581,
    0.0101758447,
    0.0496345982,
    -0.0651563033,
    -0.0362319537,
    0.00318966783,
    0.0281732827,
    0.0163737591,
    -0.0164846163,
    0.0995664597,
    0.0733684897,
    0.00946230628,
    -0.00913426094,
    -0.0742818192,
    0.06195971,
    0.00591040403,
    -0.0686667934,
    0.0317408852,
    -0.0137725631,
    0.00940212607,
    0.0279442761,
    -0.0201757029,
    0.0324154049,
    0.0228322521,
    0.00267707417,
    0.0171691254,
    0.00419936981,
    -0.0130571416,
    -0.0173847061,
    -0.0758403093,
    -0.0392607562,
    -0.0166449342,
    -0.00168224983,
    0.0199382622,
    0.0526281521,
    0.0502447523,
    -0.129478946,
    0.0623795167,
    -0.076566115,
    0.0532120988,
    0.0317721888,
    -0.0135619631,
    -0.0320665911,
    -0.0216510575,
    0.106540799,
    0.0604757369,
    -0.042855531,
    0.0153838852,
    0.0363844968,
    0.0429414809,
    0.0226413272,
    -0.039562691,
    0.0454653203,
    0.0883121043,
    0.0196657199,
    -0.0574896857,
    0.00670713792,
    -0.018773403,
    0.000347356487,
    0.0141483406,
    0.0345573537,
    0.042631086,
    -0.0191322975,
    0.0126261655,
    0.00105785835,
    -0.00561600132,
    -0.033773981,
    0.0439476408,
    -0.0444635749,
    -0.035605859,
    0.0268215071,
    -0.0402722172,
    0.0911638364,
    0.00135396153,
    0.0485091805,
    -0.0246936437,
    -0.0408962481,
    0.0829341561,
    -0.0306067225,
    -0.125902891,
    0.0731693059,
    0.0934899077,
    -0.102206372,
    0.0298629422,
    0.0766771212,
    -0.0273114517,
    -0.024197204,
    0.0,
    -0.0244167317,
    0.0970985219,
    -0.0935180783,
    0.0111901015,
    -0.00198357552,
    -0.0769073963,
    -0.121803105,
    -0.0520681925,
    -0.0417099819,
    0.0248719398,
    -0.0454342291,
    -0.0240311194,
    0.0443824418,
    -0.0372257456,
    -0.0205930769,
    0.0910829455,
    0.0527612604,
    0.0190431513,
    -0.014913708,
    0.0355940796,
    -0.00282354676,
    0.0349615514,
    0.020905517,
    0.0863897428,
    -0.0470590331,
    0.0979593918,
    0.0291745439,
    0.0513898097,
    -0.165631235,
    -0.0391068347,
    0.0751543418,
    -0.043094065,
    0.028035799,
    -0.0311034638,
    0.000913328957,
    0.0944983289,
    -0.0198726766,
    -0.0259133391,
    -0.0108968485,
    0.0387883037,
    0.0520114712,
    -0.0322841145,
    0.0131428279,
    0.0847168788,
    0.0164655242,
    -0.0242556836,
    -0.0100616785,
    -0.066205658,
    0.0352719873,
    -0.0125291245,
    0.00525686424,
    -0.0127795609,
    -0.025437668,
    -0.0697134733,
    -0.0109580038,
    0.00588004105,
    0.0470480993,
    -0.0047857468,
    0.0171248112,
    -0.0650963187,
    -0.0638555363,
    -6.33986347e-05,
    0.0477961302,
    0.0663475767,
    0.0779689029,
    0.0126418332,
    0.0279133115,
    -0.0708932728,
    -0.0341963358,
    0.0108084194,
    -0.0322745182,
    0.0595393293,
    0.0120282508,
    0.0222520698,
    -0.0312622078,
    -0.00225326256,
    -0.0878927261,
    0.0264401436,
    0.0213097129,
    -0.0696384162,
    -0.0348444693,
    -0.0397011451,
    -0.000856154773,
    0.0166215692,
    -0.0223583411,
    0.0652333274,
    0.0340826549,
    -0.0526116341,
    -0.0165710896,
    0.0189326275,
    -0.0277767386,
    -0.0210060794,
    -0.0209114067,
    0.004393938,
    0.022899719,
    -1.13862484e-08,
    0.0633643791,
    -0.00489465985,
    -0.0535186455,
    0.066366978,
    0.00781527814,
    -0.066619873,
    0.0434749424,
    -0.008214131,
    0.00729982974,
    0.108025722,
    -0.179152384,
    0.0326937772,
    0.0249010883,
    0.0834656358,
    0.0171424076,
    0.0380688161,
    0.0632147491,
    -0.0202965494,
    -0.0543595888,
    0.053911671,
    0.0329034328,
    0.0403351337,
    -0.0204342771,
    -0.0667905807,
    0.0286556948,
    0.00270259427,
    -0.0699809119,
    0.0458261557,
    -0.0122208307,
    0.0477884784,
    0.00767229684,
    -0.0723900646,
    -0.0811463594,
    0.0289416574,
    0.0698303133,
    0.0109635908,
    -0.066716738,
    -0.0869814679,
    0.0781401545,
    -0.0747744292,
    -0.0933830217,
    0.0906731561,
    -0.118223637,
    -0.00360242673,
    0.00453409506,
    0.0433466882,
    -0.0145340748,
    0.101847678,
    -0.0519261472,
    0.0441147573,
    -0.0348532163,
    0.0241619237,
    0.0494029559,
    -0.0146116838,
    0.0442838222,
    -0.0998176262,
    0.0255751535,
    -0.00209640572,
    0.0171953607,
    0.0489534624,
    0.0367930681,
    0.0853904262,
    -0.0312620848,
    -0.0702058449,
]


@pytest.mark.skipif(
    os.environ.get("IMAGE_KIND") == "TRTLLM",
    reason="TRT-LLM backend does not support embedding requests",
)
class TestEmbeddings:
    @pytest.fixture(scope="class")
    def client(self, fastapi_client_class_scope):
        yield fastapi_client_class_scope

    @pytest.fixture(scope="class")
    def model(self):
        # Override with embeddings-specific model
        return "all-MiniLM-L6-v2"

    @pytest.fixture(scope="class")
    def tokenizer_model(self):
        return None

    @pytest.fixture(scope="class")
    def model_repository(self):
        # Override with embeddings-specific repository
        return str(Path(__file__).parent / "vllm_embedding_models")

    @pytest.fixture(scope="class")
    def input(self):
        return "The food was delicious and the waiter..."

    def _check_embedding_response(self, response, model, encoding_format="float"):
        assert response.status_code == 200, response.json()
        embedding = response.json()["data"][0]["embedding"]
        assert embedding is not None
        if encoding_format == "base64":
            embedding = np.frombuffer(base64.b64decode(embedding), dtype=np.float32)

        # The results vary slightly across GPU models
        result = np.allclose(EMBEDDING_OUTPUT_FLOAT, embedding, rtol=0, atol=1e-3)
        assert (
            result
        ), f"Embeddings do not match expected output\nExpect {EMBEDDING_OUTPUT_FLOAT},\ngot{embedding}"

        assert response.json()["data"][0]["object"] == "embedding"
        assert response.json()["data"][0]["index"] == 0
        assert response.json()["model"] == model

        usage = response.json().get("usage")
        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["total_tokens"] == 12

    @pytest.mark.parametrize(
        "input",
        [
            "The food was delicious and the waiter...",
            [101, 1996, 2833, 2001, 12090, 1998, 1996, 15610, 1012, 1012, 1012, 102],
        ],
    )
    def test_embeddings_defaults(self, client, model: str, input: str):
        response = client.post(
            "/v1/embeddings",
            json={"model": model, "input": input},
        )

        self._check_embedding_response(response, model)

    # FIXME: Python model cannot unload gracefully if raise error.
    # def test_chat_completions_defaults(
    #     self, client, model: str, messages: List[dict], backend: str
    # ):
    #     response = client.post(
    #         "/v1/chat/completions",
    #         json={"model": model, "messages": messages},
    #     )

    #     assert response.status_code == 400
    #     assert "does not support" in response.json()["detail"]

    @pytest.mark.parametrize(
        "param_key, param_value",
        [
            ("encoding_format", "invalid"),
            ("encoding_format", 0),
        ],
    )
    def test_embeddings_invalid_parameters(
        self, client, param_key, param_value, model: str, input: str
    ):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": model,
                "input": input,
                param_key: param_value,
            },
        )

        # Assert schema validation error
        assert response.status_code == 422, response.json()

    @pytest.mark.parametrize("encoding_format", ["float", "base64"])
    def test_embeddings_parameters(
        self, client, encoding_format, model: str, input: str
    ):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
            },
        )

        self._check_embedding_response(response, model, encoding_format=encoding_format)

    def test_embeddings_empty_request(self, client):
        response = client.post("/v1/embeddings", json={})
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    def test_embeddings_no_model(self, client, input: str):
        response = client.post("/v1/embeddings", json={"input": input})
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    @pytest.mark.parametrize(
        "model, error_code",
        [
            ("", 400),
            (123, 422),
            ("Invalid", 400),
            (None, 422),
        ],
    )
    def test_embeddings_invalid_model(self, client, model: str, input, error_code: int):
        print("Model:", model)
        # Message validation requires min_length of 1
        response = client.post("/v1/embeddings", json={"model": model, "input": input})
        assert response.status_code == error_code
        if error_code == 400:
            assert response.json()["detail"] == f"Unknown model: {model}"
        else:
            assert (
                response.json()["detail"][0]["msg"] == "Input should be a valid string"
            )

    def test_embeddings_no_input(self, client, model: str):
        response = client.post("/v1/embeddings", json={"model": model})
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "input",
        [
            "",
            [],
        ],
    )
    def test_embeddings_empty_input(self, client, model: str, input):
        # Message validation requires min_length of 1
        response = client.post("/v1/embeddings", json={"model": model, "input": input})
        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "Value should have at least 1 item after validation, not 0"
        )

    @pytest.mark.parametrize(
        "input",
        [
            123,
            1.5,
            0,
            None,
        ],
    )
    def test_embeddings_invalid_input(self, client, model: str, input):
        # Message validation requires min_length of 1
        response = client.post("/v1/embeddings", json={"model": model, "input": input})
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Input should be a valid string"
