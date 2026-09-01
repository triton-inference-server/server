#!/usr/bin/python
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

import socket
import sys
import unittest

sys.path.append("../common")
from test_util import MIB, get_server_process_from_env, wait_for_stable_rss


class SagemakerRequestManyChunksTest(unittest.TestCase):
    def setUp(self):
        self._local_host = "localhost"
        self._sagemaker_port = 8080
        # Must match server kMaxChunkedChunks (http_server.cc).
        self._k_max_chunked_chunks = 65536
        self._over_max_chunks_error = f"Chunked request body exceeds maximum of {self._k_max_chunked_chunks} non-empty chunks. Send fewer or larger HTTP chunks."

    def _sagemaker_chunked_header(self):
        return (
            f"POST /models HTTP/1.1\r\n" f"X-Amzn-SageMaker-Target-Model: ZZZZZZZ\r\n"
        )

    def send_chunked_request(
        self,
        header: str,
        chunk_count: int,
        expected_response: str,
        expected_http_status=400,
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

            # Send chunked payload (server may close early when over chunk limit)
            for _ in range(chunk_count):
                try:
                    s.send(b"1\r\nA\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            try:
                s.sendall(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Server may close/reset early after detecting chunk-limit violation.
                # In that case, failing to send the terminating chunk is expected.
                pass

            # Receive response
            response = b""
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                except ConnectionResetError:
                    break
                except socket.timeout:
                    break
            self.assertTrue(
                response,
                "expected error response body, but socket closed/reset before any bytes",
            )
            status_line = response.split(b"\r\n", 1)[0].decode(errors="replace")
            self.assertTrue(
                status_line.startswith(f"HTTP/1.1 {expected_http_status} "),
                f"expected HTTP status {expected_http_status}, got {status_line!r}",
            )
            self.assertIn(expected_response, response.decode())
        except Exception as e:
            raise (e)
        finally:
            s.close()

    def test_chunked_at_max_chunks(self):
        self.send_chunked_request(
            self._sagemaker_chunked_header(),
            self._k_max_chunked_chunks,
            "failed to parse the request JSON buffer: Invalid value. at 0",
        )

    def test_chunked_rejected_over_max_chunks(self):
        self.send_chunked_request(
            self._sagemaker_chunked_header(),
            self._k_max_chunked_chunks + 1,
            self._over_max_chunks_error,
        )

    def test_chunked_over_max_chunks_reject_with_bounded_rss_growth(self):
        many_chunks = 1000000

        # verify server is running
        server = get_server_process_from_env("SERVER_PID")
        self.assertTrue(server.is_running())

        # warm up and wait until RSS is stable.
        self.send_chunked_request(
            self._sagemaker_chunked_header(),
            1000000,  # way over max chunks
            self._over_max_chunks_error,
        )
        # Wait until RSS is stable across several measurements before continuing.
        server = get_server_process_from_env("SERVER_PID")
        wait_for_stable_rss(server)

        # Monitor RSS growth over 100 requests.
        repeat_request_count = 100
        rss_before = server.memory_info().rss
        # TODO: Why sagemaker server occasionally grows >10 MiB but http server always smaller than 1 MiB?
        max_rss_growth_bytes = 20 * MIB

        for _ in range(repeat_request_count):
            self.send_chunked_request(
                self._sagemaker_chunked_header(),
                many_chunks,  # way over max chunks
                self._over_max_chunks_error,
            )

        rss_after = server.memory_info().rss
        growth = rss_after - rss_before
        print(
            f"RSS: before={rss_before / MIB:.1f} MiB, "
            f"after={rss_after / MIB:.1f} MiB, "
            f"growth={growth / MIB:.1f} MiB, "
            f"limit={max_rss_growth_bytes / MIB:.0f} MiB",
            flush=True,
        )
        self.assertLess(
            growth,
            max_rss_growth_bytes,
            f"Server RSS grew by {growth / MIB:.1f} MiB after "
            f"{repeat_request_count} over-limit chunked infer requests "
            f"(limit {max_rss_growth_bytes / MIB:.0f} MiB).",
        )


if __name__ == "__main__":
    unittest.main()
