# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import numpy as np
import pytest

EMBEDDING_OUTPUT_FLOAT = [
    -0.1914404183626175,
    0.4000193178653717,
    0.058502197265625,
    0.18909454345703125,
    -0.4690297544002533,
    0.004936536308377981,
    0.45893096923828125,
    -0.31141534447669983,
    0.18299102783203125,
    -0.4907582700252533,
    0.6920369267463684,
    -0.001537322998046875,
    0.1219015121459961,
    -0.11682561784982681,
    0.02811431884765625,
    -0.5207672119140625,
    0.4574941098690033,
    -0.31097412109375,
    0.13371849060058594,
    -0.4693959653377533,
    -0.2766602337360382,
    0.029005050659179688,
    -0.13730454444885254,
    -0.18662643432617188,
    0.0063533782958984375,
    0.16905848681926727,
    0.1612701416015625,
    0.08376502990722656,
    -0.09822845458984375,
    -0.012738545425236225,
    -0.16643650829792023,
    -0.01901753805577755,
    -0.0503285713493824,
    0.03994830325245857,
    -0.3373819887638092,
    -0.0188166294246912,
    0.374481201171875,
    -0.4371846616268158,
    0.22470474243164062,
    -0.011973063461482525,
    0.13784568011760712,
    -0.1484222412109375,
    0.19347667694091797,
    0.11848513036966324,
    0.09258091449737549,
    -0.2887814939022064,
    -0.2301533967256546,
    -0.12088584899902344,
    0.1941477507352829,
    -0.05228869244456291,
    -0.2508443295955658,
    0.15698719024658203,
    -0.19024403393268585,
    -0.4728952944278717,
    0.336700439453125,
    0.11721674352884293,
    0.24498240649700165,
    -0.4826914370059967,
    -0.119984470307827,
    0.1008249893784523,
    -0.0983428955078125,
    -0.0059178671799600124,
    0.2341969758272171,
    0.2725118100643158,
    0.2384185791015625,
    -0.30748113989830017,
    -0.17387008666992188,
    -0.44342041015625,
    -0.15537135303020477,
    0.27348455786705017,
    0.1131540909409523,
    -0.23855717480182648,
    0.2574056088924408,
    -0.12090619653463364,
    -0.14412720501422882,
    -0.2408244013786316,
    0.0021938085556030273,
    -0.39945730566978455,
    -0.555908203125,
    0.0760548934340477,
    -0.1530914306640625,
    -0.40975189208984375,
    -0.2091045379638672,
    0.20317332446575165,
    -0.20295588672161102,
    -0.3643442690372467,
    0.05287488177418709,
    -0.24874623119831085,
    0.11500009149312973,
    0.1661122590303421,
    0.26618704199790955,
    -0.22980372607707977,
    -0.202911376953125,
    -0.2738393247127533,
    0.20629756152629852,
    -0.24571101367473602,
    -0.1486002653837204,
    -0.12444128841161728,
    0.27539315819740295,
    0.41679826378822327,
    -0.01199467945843935,
    0.1778361052274704,
    -0.15123574435710907,
    -0.0391184501349926,
    -0.035979270935058594,
    0.11838880926370621,
    -0.07832065969705582,
    0.15302227437496185,
    -0.11540285497903824,
    -0.008619308471679688,
    0.011735956184566021,
    0.41825103759765625,
    0.1798756867647171,
    0.0468953438103199,
    -0.31410470604896545,
    -0.28439536690711975,
    0.028476715087890625,
    -0.18972015380859375,
    -0.1492512971162796,
    0.23354721069335938,
    0.2631734311580658,
    0.3009694516658783,
    -0.31204381585121155,
    0.17155838012695312,
    -0.6126009821891785,
    -0.16471035778522491,
    0.7154337763786316,
    0.0,
    -0.3936564028263092,
    -0.15255196392536163,
    0.24118296802043915,
    -0.13930638134479523,
    0.6811599731445312,
    0.135009765625,
    -0.18750762939453125,
    0.26521047949790955,
    -0.1257190704345703,
    0.0532684326171875,
    0.25982680916786194,
    -0.3410797119140625,
    -0.189666748046875,
    0.016697248443961143,
    0.1474812775850296,
    0.085713230073452,
    -0.0862935408949852,
    0.521209716796875,
    0.3840688169002533,
    0.04953320696949959,
    -0.0478159599006176,
    -0.3888498842716217,
    0.3243462145328522,
    0.03093973733484745,
    -0.3594563901424408,
    0.16615693271160126,
    -0.07209650427103043,
    0.049218177795410156,
    0.14628247916698456,
    -0.10561561584472656,
    0.1696879118680954,
    0.1195220947265625,
    0.0140139264985919,
    0.08987680822610855,
    0.02198282815515995,
    -0.06835142523050308,
    -0.09100532531738281,
    -0.3970082700252533,
    -0.20552189648151398,
    -0.0871327742934227,
    -0.008806228637695312,
    0.10437265783548355,
    0.2754974365234375,
    0.2630208432674408,
    -0.67779541015625,
    0.32654380798339844,
    -0.4008077085018158,
    0.2785542905330658,
    0.16632080078125,
    -0.0709940567612648,
    -0.1678619384765625,
    -0.11333879083395004,
    0.5577189326286316,
    0.3165779113769531,
    -0.2243397980928421,
    0.08053144067525864,
    0.1904652863740921,
    0.22478973865509033,
    0.11852264404296875,
    -0.2071024626493454,
    0.2380015105009079,
    0.4622955322265625,
    0.1029459610581398,
    -0.30094656348228455,
    0.0351104736328125,
    -0.09827486425638199,
    0.0018183389911428094,
    0.07406362146139145,
    0.18090057373046875,
    0.2231648713350296,
    -0.1001536026597023,
    0.06609535217285156,
    0.0055376687087118626,
    -0.02939859963953495,
    -0.17679977416992188,
    0.2300567626953125,
    -0.232757568359375,
    -0.1863892823457718,
    0.14040501415729523,
    -0.21081669628620148,
    0.4772237241268158,
    0.00708770751953125,
    0.25393548607826233,
    -0.12926609814167023,
    -0.21408335864543915,
    0.43414306640625,
    -0.16021983325481415,
    -0.6590754389762878,
    0.383026123046875,
    0.4894002377986908,
    -0.5350291132926941,
    0.1563262939453125,
    0.4013887941837311,
    -0.1429697722196579,
    -0.1266673356294632,
    0.0,
    -0.12781651318073273,
    0.5082905888557434,
    -0.4895477294921875,
    0.05857785418629646,
    -0.01038360595703125,
    -0.4025942385196686,
    -0.6376139521598816,
    -0.27256616950035095,
    -0.2183430939912796,
    0.13019943237304688,
    -0.2378387451171875,
    -0.12579791247844696,
    0.23233287036418915,
    -0.1948690414428711,
    -0.10780048370361328,
    0.4768002927303314,
    0.2761942446231842,
    0.09968694299459457,
    -0.07807016372680664,
    0.18632762134075165,
    -0.014780680648982525,
    0.18301646411418915,
    0.10943603515625,
    0.45223236083984375,
    -0.24634425342082977,
    0.5127970576286316,
    0.15272267162799835,
    0.26901498436927795,
    -0.8670451045036316,
    -0.20471616089344025,
    0.3934173583984375,
    -0.22558848559856415,
    0.14676158130168915,
    -0.16282017529010773,
    0.0047810873948037624,
    0.49467912316322327,
    -0.1040293350815773,
    -0.13565094769001007,
    -0.05704273656010628,
    0.2030487060546875,
    0.27226924896240234,
    -0.16900062561035156,
    0.06879997253417969,
    0.44347524642944336,
    0.08619359880685806,
    -0.1269734650850296,
    -0.05267079547047615,
    -0.3465728759765625,
    0.1846415251493454,
    -0.0655873641371727,
    0.027518590912222862,
    -0.06689834594726562,
    -0.13316090404987335,
    -0.3649355471134186,
    -0.0573628731071949,
    0.030780792236328125,
    0.2462870329618454,
    -0.0250523891299963,
    0.08964482694864273,
    -0.34076571464538574,
    -0.3342704772949219,
    -0.000331878662109375,
    0.25020280480384827,
    0.34731578826904297,
    0.4081510007381439,
    0.0661773681640625,
    0.14612038433551788,
    -0.37111154198646545,
    -0.17901070415973663,
    0.0565798282623291,
    -0.1689503937959671,
    0.311676025390625,
    0.06296539306640625,
    0.11648496240377426,
    -0.16365115344524384,
    -0.011795361526310444,
    -0.4601001739501953,
    0.13840866088867188,
    0.1115519180893898,
    -0.3645426332950592,
    -0.182403564453125,
    -0.20782725512981415,
    -0.004481792449951172,
    0.0870104655623436,
    -0.11704126745462418,
    0.34148290753364563,
    0.17841561138629913,
    -0.2754109799861908,
    -0.0867462158203125,
    0.09910837560892105,
    -0.14540545642375946,
    -0.10996246337890625,
    -0.10946687310934067,
    0.023001352325081825,
    0.11987527459859848,
    -5.960464477539063e-8,
    0.3316993713378906,
    -0.025622526183724403,
    -0.28015899658203125,
    0.34741735458374023,
    0.04091135784983635,
    -0.34874120354652405,
    0.22758229076862335,
    -0.042999267578125,
    0.0382130928337574,
    0.5654922127723694,
    -0.9378255009651184,
    0.17114512622356415,
    0.13035202026367188,
    0.4369252622127533,
    0.0897369384765625,
    0.19928233325481415,
    0.33091607689857483,
    -0.10624822229146957,
    -0.2845611572265625,
    0.2822163999080658,
    0.1722426414489746,
    0.2111460417509079,
    -0.1069692000746727,
    -0.3496347963809967,
    0.15000660717487335,
    0.014147520065307617,
    -0.36633554100990295,
    0.23989041149616241,
    -0.06397350877523422,
    0.2501627504825592,
    0.04016287997364998,
    -0.3789469301700592,
    -0.4247843325138092,
    0.1515035629272461,
    0.36554718017578125,
    0.057392120361328125,
    -0.3492482602596283,
    -0.45532989501953125,
    0.4090474545955658,
    -0.3914286196231842,
    -0.4888407289981842,
    0.4746551513671875,
    -0.6188761591911316,
    -0.018857955932617188,
    0.02373504638671875,
    0.22691090404987335,
    -0.07608286291360855,
    0.5331514477729797,
    -0.27182260155677795,
    0.2309315949678421,
    -0.1824493408203125,
    0.12648265063762665,
    0.2586142122745514,
    -0.07648912817239761,
    0.2318166047334671,
    -0.5225245356559753,
    0.133880615234375,
    -0.010974247939884663,
    0.09001413732767105,
    0.2562611997127533,
    0.19260406494140625,
    0.4470011293888092,
    -0.1636505126953125,
    -0.3675130307674408,
]


@pytest.mark.fastapi
class TestEmbeddings:
    @pytest.fixture(scope="class", autouse=True)
    def check_backend(self, backend: str):
        if backend != "vllm":
            pytest.skip("These tests only run with vLLM backend")

    @pytest.fixture(scope="class")
    def client(self, fastapi_client_class_scope):
        yield fastapi_client_class_scope

    @pytest.fixture(scope="class")
    def model(self):
        # Override with embeddings-specific model
        return "all-MiniLM-L6-v2"

    @pytest.fixture(scope="class")
    def model_repository(self):
        # Override with embeddings-specific repository
        return str(Path(__file__).parent / "vllm_embedding_models")

    @pytest.fixture(scope="class")
    def input(self):
        return "The food was delicious and the waiter..."

    def _check_embedding_response(
        self, response, model, dims=len(EMBEDDING_OUTPUT_FLOAT), encoding_format="float"
    ):
        assert response.status_code == 200, response.json()
        embedding = response.json()["data"][0]["embedding"]
        assert embedding is not None
        if encoding_format == "base64":
            embedding = np.frombuffer(base64.b64decode(embedding), dtype=np.float32)
        result = np.allclose(
            EMBEDDING_OUTPUT_FLOAT[:dims], embedding, rtol=1e-5, atol=0.0
        )
        assert result, "Embedding does not match expected output"

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
            ("dimensions", [10]),
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

    @pytest.mark.parametrize("dimensions", [0, 10, 100, -1])
    @pytest.mark.parametrize("encoding_format", ["float", "base64"])
    def test_embeddings_parameters(
        self, client, dimensions, encoding_format, model: str, input: str
    ):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": model,
                "input": input,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
            },
        )

        self._check_embedding_response(
            response, model, dims=dimensions, encoding_format=encoding_format
        )

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
