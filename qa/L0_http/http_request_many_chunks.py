#!/usr/bin/python
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

import socket
import unittest


class HTTPRequestManyChunksTest(unittest.TestCase):
    def setUp(self):
        self._model_name = "simple"
        self._local_host = "localhost"
        self._http_port = 8000
        self._malicious_chunk_count = (
            1000000  # large enough to cause a stack overflow if using alloca()
        )
        self._parse_error = (
            "failed to parse the request JSON buffer: Invalid value. at 0"
        )

    def send_chunked_request(
        self, header: str, chunk_count: int, expected_response: str
    ):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        header = (
            f"{header}"
            f"Host: {self._local_host}:{self._http_port}\r\n"
            f"Content-Type: application/octet-stream\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        try:
            s.connect((self._local_host, self._http_port))
            # HTTP request with chunked encoding
            s.sendall((header.encode()))

            # Send chunked payload
            for _ in range(chunk_count):
                s.send(b"1\r\nA\r\n")
            # End chunked encoding
            s.sendall(b"0\r\n\r\n")

            # Receive response
            response = b""
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                except socket.timeout:
                    break
            self.assertIn(expected_response, response.decode())
        except Exception as e:
            raise (e)
        finally:
            s.close()

    def test_infer(self):
        request_header = (
            f"POST /v2/models/{self._model_name}/infer HTTP/1.1\r\n"
            f"Inference-Header-Content-Length: 0\r\n"
        )

        self.send_chunked_request(
            request_header,
            self._malicious_chunk_count,
            "Raw request must only have 1 input (found 1) to be deduced but got 2 inputs in 'simple' model configuration",
        )

    def test_registry_index(self):
        request_header = f"POST /v2/repository/index HTTP/1.1\r\n"

        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_model_control(self):
        load_request_header = (
            f"POST /v2/repository/models/{self._model_name}/load HTTP/1.1\r\n"
        )
        unload_request_header = load_request_header.replace("/load", "/unload")

        self.send_chunked_request(
            load_request_header, self._malicious_chunk_count, self._parse_error
        )
        self.send_chunked_request(
            unload_request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_trace(self):
        request_header = (
            f"POST /v2/models/{self._model_name}/trace/setting HTTP/1.1\r\n"
        )

        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_logging(self):
        request_header = f"POST /v2/logging HTTP/1.1\r\n"

        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_system_shm_register(self):
        request_header = f"POST /v2/systemsharedmemory/region/test_system_shm_register/register HTTP/1.1\r\n"

        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_cuda_shm_register(self):
        request_header = f"POST /v2/cudasharedmemory/region/test_cuda_shm_register/register HTTP/1.1\r\n"

        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )

    def test_generate(self):
        request_header = f"POST /v2/models/{self._model_name}/generate HTTP/1.1\r\n"
        self.send_chunked_request(
            request_header, self._malicious_chunk_count, self._parse_error
        )


if __name__ == "__main__":
    unittest.main()
