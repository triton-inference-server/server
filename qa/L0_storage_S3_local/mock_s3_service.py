#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockS3Service:
    __address = "localhost"
    __port = 8080

    def __init__(self):
        # Test passed when:
        # - at least one HEAD request is received; and
        # - at least one GET request is received; and
        # - all received requests do not advertise for HTTP/2.
        test_results = {"head_count": 0, "get_count": 0, "http2_ads": False}

        class RequestValidator(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def __CheckHttp2Ads(self):
                if "connection" in self.headers:
                    v = self.headers["connection"].lower()
                    if "upgrade" in v or "http2" in v:
                        test_results["http2_ads"] = True
                if (
                    "upgrade" in self.headers
                    and "h2c" in self.headers["upgrade"].lower()
                ):
                    test_results["http2_ads"] = True
                if "http2-settings" in self.headers:
                    test_results["http2_ads"] = True

            def do_HEAD(self):
                self.__CheckHttp2Ads()
                test_results["head_count"] += 1
                self.send_response(200)
                self.end_headers()

            def do_GET(self):
                self.__CheckHttp2Ads()
                test_results["get_count"] += 1
                self.send_error(
                    404,
                    "Thank you for using the mock s3 service!",
                    "Your bucket is not found here!",
                )

        self.__test_results = test_results
        self.__server = HTTPServer((self.__address, self.__port), RequestValidator)
        self.__service_thread = threading.Thread(target=self.__server.serve_forever)

    def __enter__(self):
        self.__service_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__server.shutdown()
        self.__server.server_close()
        self.__service_thread.join()

    def TestPassed(self):
        return (
            self.__test_results["head_count"] > 0
            and self.__test_results["get_count"] > 0
            and not self.__test_results["http2_ads"]
        )


if __name__ == "__main__":
    # Initialize mock service
    mock_s3_service = MockS3Service()

    # Start service and poll until test passed or timed-out
    with mock_s3_service:
        poll_interval = 1  # seconds
        timeout = 10  # seconds
        elapsed_time = 0  # seconds
        while not mock_s3_service.TestPassed() and elapsed_time < timeout:
            elapsed_time += poll_interval
            time.sleep(poll_interval)

    # Print the result
    if mock_s3_service.TestPassed():
        print("TEST PASSED")
    else:
        print("TEST FAILED")
