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

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from tests.utils import setup_fastapi_app, setup_server


@pytest.mark.fastapi
class TestEmbeddings:
    @pytest.fixture(scope="class")
    def client(self):
        model_repository = Path(__file__).parent / "test_models"
        server = setup_server(str(model_repository))
        # Don't need tokenizer or specific backend for mock embedding model
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_embeddings_defaults(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"model": "mock_embedding", "input": "A test sentence."},
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["object"] == "list"
        assert len(response_json["data"]) == 1
        embedding_obj = response_json["data"][0]
        assert embedding_obj["object"] == "embedding"
        assert embedding_obj["index"] == 0
        assert len(embedding_obj["embedding"]) == 384
        assert response_json["model"] == "mock_embedding"
        assert "usage" in response_json

    def test_embeddings_multiple_inputs(self, client):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "mock_embedding",
                "input": ["A test sentence.", "Another test sentence."],
            },
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["object"] == "list"
        assert len(response_json["data"]) == 2
        for i, embedding_obj in enumerate(response_json["data"]):
            assert embedding_obj["object"] == "embedding"
            assert embedding_obj["index"] == i
            assert len(embedding_obj["embedding"]) == 384

    def test_embeddings_invalid_model(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"model": "non_existent_model", "input": "A test sentence."},
        )
        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]
