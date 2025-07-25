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


class SagemakerRequestManyChunksTest(unittest.TestCase):
    def setUp(self):
        self._local_host = "localhost"
        self._sagemaker_port = 8080
        self._malicious_chunk_count = (
            1000000  # large enough to cause a stack overflow if using alloca()
        )

    def send_chunked_request(
        self, header: str, chunk_count: int, expected_response: str
    ):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        header = (
            f"{header}"
            f"Host: {self._local_host}:{self._sagemaker_port}\r\n"
            f"Content-Type: application/octet-stream\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        try:
            s.connect((self._local_host, self._sagemaker_port))
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

    def test_load_model(self):
        request_header = (
            f"POST /models HTTP/1.1\r\n" f"X-Amzn-SageMaker-Target-Model: ZZZZZZZ\r\n"
        )
        self.send_chunked_request(
            request_header,
            self._malicious_chunk_count,
            "failed to parse the request JSON buffer: Invalid value. at 0",
        )


if __name__ == "__main__":
    unittest.main()
