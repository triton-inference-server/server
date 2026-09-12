# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Offline preparation and HTTP-contract tests; no cloud credentials or GPU."""

import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_models  # noqa: E402
import smoke_test  # noqa: E402


class FakeDownload(io.BytesIO):
    def geturl(self):
        return "https://example.invalid/model"


class PreparationTests(unittest.TestCase):
    def artifact(self, content):
        return {
            "path": prepare_models.CONFIG_PATH,
            "url": "https://example.invalid/config",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def test_download_verifies_content_and_size(self):
        correct = b"fixed model bytes"
        bad = [b"x" * len(correct), correct[:-1], correct + b"x"]
        with tempfile.TemporaryDirectory() as temp:
            for i, body in enumerate([correct] + bad):
                target = Path(temp) / str(i)
                with patch("urllib.request.urlopen", return_value=FakeDownload(body)):
                    if i == 0:
                        prepare_models.fetch(self.artifact(correct), target)
                        self.assertEqual(target.read_bytes(), correct)
                    else:
                        with self.assertRaises(ValueError):
                            prepare_models.fetch(self.artifact(correct), target)

    def test_lfs_pointer_is_not_a_model(self):
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:fake\n"
        with tempfile.TemporaryDirectory() as temp:
            with patch("urllib.request.urlopen", return_value=FakeDownload(pointer)):
                with self.assertRaises(ValueError):
                    prepare_models.fetch(
                        self.artifact(b"a real model"), Path(temp) / "model"
                    )

    def test_failed_prepare_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "build"
            with patch("urllib.request.urlopen", return_value=FakeDownload(b"bad")):
                with self.assertRaises(ValueError):
                    prepare_models.prepare(
                        out, {"artifacts": [self.artifact(b"correct")]}
                    )
            self.assertFalse(out.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_prepare_does_not_replace_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "keep"
            sentinel.write_text("existing data")
            with self.assertRaises(ValueError):
                prepare_models.prepare(Path(temp), {"artifacts": []})
            self.assertEqual(sentinel.read_text(), "existing data")

    def test_path_traversal_is_rejected(self):
        for relative in [
            "../secret",
            "/tmp/secret",
            "model_repository/../../secret",
            "notices\\secret",
            "",
        ]:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                prepare_models.destination(Path("/tmp/build"), relative)

    def test_complete_repository_has_verifiable_final_config(self):
        source = b'name: "densenet_onnx"\n'
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "build"
            with patch("urllib.request.urlopen", return_value=FakeDownload(source)):
                prepare_models.prepare(out, {"artifacts": [self.artifact(source)]})
            config = (out / prepare_models.CONFIG_PATH).read_bytes()
            self.assertIn(b"KIND_GPU", config)
            self.assertIn(b"versions: [1]", config)
            checksum, path = (out / "SHA256SUMS").read_text().strip().split("  ")
            self.assertEqual(
                hashlib.sha256((out / path).read_bytes()).hexdigest(), checksum
            )

    def test_runtime_pin_matches_dockerfile(self):
        root = Path(__file__).resolve().parents[1]
        runtime = json.loads((root / "pins.json").read_text())["runtime"]
        self.assertIn(
            runtime["tag"] + "@" + runtime["digest"], (root / "Dockerfile").read_text()
        )


def valid_response():
    return {
        "model_name": "densenet_onnx",
        "model_version": "1",
        "outputs": [
            {
                "name": "fc6_1",
                "datatype": "FP32",
                "shape": [1000],
                "data": [0.25] * 1000,
            }
        ],
    }


class OutputTests(unittest.TestCase):
    def test_rejects_wrong_metadata_and_nonfinite_or_boolean_values(self):
        changes = [
            lambda r: r.update(model_version="2"),
            lambda r: r.update(model_name="other"),
            lambda r: r["outputs"][0].update(shape=[1, 1000]),
            lambda r: r["outputs"][0].update(datatype="INT32"),
            lambda r: r["outputs"][0].update(name="wrong"),
            lambda r: r["outputs"][0].update(data=[0.0] * 999),
            lambda r: r["outputs"][0]["data"].__setitem__(0, math.nan),
            lambda r: r["outputs"][0]["data"].__setitem__(0, True),
        ]
        for change in changes:
            response = valid_response()
            change(response)
            with self.assertRaises(ValueError):
                smoke_test.validate_output(response)

    def test_reference_comparison_detects_wrong_numerical_result(self):
        smoke_test.compare_reference([0.1] * 1000, [0.10001] * 1000)
        with self.assertRaises(smoke_test.CheckError):
            smoke_test.compare_reference([0.1] * 1000, [0.2] * 1000)

    def test_url_and_token_validation(self):
        for url in [
            "http://example.com",
            "https://user:password@example.com",
            "https://example.com/?token=x",
            "https://example.com/#x",
            "https://example.com/\n",
        ]:
            with self.subTest(url=url), self.assertRaises(smoke_test.CheckError):
                smoke_test.Client(url, "test-token", allow_local_http=True)
        with self.assertRaises(smoke_test.CheckError):
            smoke_test.Client("https://example.com", "secret\r\nInjected: x")


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, code, body=b""):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def handle_request(self, payload=None):
                state = self.server.state
                state["requests"].append(
                    (self.path, self.headers.get("Authorization"), payload)
                )
                if state.get("redirect"):
                    self.send_response(302)
                    self.send_header("Location", state["redirect"])
                    self.end_headers()
                    return
                if (
                    not state.get("auth_disabled")
                    and self.headers.get("Authorization") != "Bearer test-token"
                ):
                    self.respond(401)
                elif self.path == "/v2/health/ready":
                    self.respond(200)
                elif self.path == smoke_test.MODEL_READY:
                    self.respond(state.get("model_ready", 200))
                elif self.path == smoke_test.INFER:
                    self.respond(
                        200, state.get("body", json.dumps(valid_response()).encode())
                    )
                else:
                    self.respond(404)

            def do_GET(self):
                self.handle_request()

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                self.handle_request(payload)

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.state = {"requests": []}
        self.client = smoke_test.Client(self.url, "test-token", allow_local_http=True)

    def test_ready_auth_and_real_http_inference(self):
        self.client.wait_ready(1, 1)
        self.client.check_auth(1)
        self.assertEqual(len(self.client.infer(1)), 1000)
        calls = self.server.state["requests"]
        self.assertEqual(len(calls), 7)
        payload = calls[-1][2]
        self.assertEqual(payload["inputs"][0]["shape"], [3, 224, 224])
        self.assertEqual(len(payload["inputs"][0]["data"]), 150528)
        self.assertEqual(calls[-1][0], smoke_test.INFER)

    def test_server_ready_without_model_is_not_success(self):
        self.server.state["model_ready"] = 503
        with self.assertRaisesRegex(smoke_test.CheckError, "deadline"):
            self.client.wait_ready(0.05, 1)

    def test_bad_readiness_token_fails_immediately(self):
        self.client.token = "wrong"
        with self.assertRaisesRegex(smoke_test.CheckError, "authentication"):
            self.client.wait_ready(1, 1)
        self.assertEqual(len(self.server.state["requests"]), 1)

    def test_public_endpoint_fails_auth_check(self):
        self.server.state["auth_disabled"] = True
        with self.assertRaisesRegex(smoke_test.CheckError, "not rejected"):
            self.client.check_auth(1)

    def test_redirect_is_not_followed_with_a_token(self):
        self.server.state["redirect"] = self.url + "/capture"
        with self.assertRaisesRegex(smoke_test.CheckError, "302"):
            self.client.wait_ready(1, 1)
        self.assertEqual(len(self.server.state["requests"]), 1)

    def test_malformed_and_oversized_responses(self):
        for body in [b"not json", b"x" * (smoke_test.MAX_RESPONSE_BYTES + 1)]:
            self.server.state["body"] = body
            with self.assertRaises(smoke_test.CheckError):
                self.client.infer(1)

    def test_main_does_not_log_response_secrets(self):
        self.server.state["body"] = b"test-token secret body"
        stderr = io.StringIO()
        with patch.dict("os.environ", {"ENDPOINT_AUTH_TOKEN": "test-token"}):
            with patch.object(
                sys, "argv", ["smoke_test.py", "--url", self.url, "--allow-local-http"]
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(smoke_test.main(), 1)
        self.assertNotIn("test-token", stderr.getvalue())
        self.assertNotIn("secret body", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
