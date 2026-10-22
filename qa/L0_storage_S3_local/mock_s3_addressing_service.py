#!/usr/bin/env python3

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

# Mock S3 service that records how the Triton S3 client addresses the bucket.
#
# The S3 addressing style is observable from the request the client sends:
#   - path-style:            Host: <endpoint>            path: /<bucket>/<key>
#   - virtual-hosted-style:  Host: <bucket>.<endpoint>   path: /<key>
#
# This lets the test assert whether Triton emitted path-style or
# virtual-hosted-style requests, which is what the S3_USE_VIRTUAL_ADDRESSING
# env var / use_virtual_addressing credential field controls. It intentionally
# does not require a real S3-compatible store that forbids path-style.

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockS3AddressingService:
    def __init__(self, address="0.0.0.0", port=8080, bucket="dummy-bucket"):
        self.__address = address
        self.__port = port
        self.__bucket = bucket

        # Records observed addressing style across all received requests.
        results = {
            "request_count": 0,
            "virtual_hosted_count": 0,
            "path_style_count": 0,
        }
        bucket_name = bucket

        class RequestValidator(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def __classify(self):
                results["request_count"] += 1
                # The Host header carries the bucket prefix only for
                # virtual-hosted-style addressing.
                host = self.headers.get("host", "").lower()
                if host.startswith(bucket_name.lower() + "."):
                    results["virtual_hosted_count"] += 1
                else:
                    results["path_style_count"] += 1

            def do_HEAD(self):
                self.__classify()
                self.send_response(200)
                self.end_headers()

            def do_GET(self):
                self.__classify()
                self.send_error(404, "mock s3 service", "not found here")

            # Silence default logging to keep the test log readable.
            def log_message(self, fmt, *args):
                return

        self.__results = results
        self.__server = HTTPServer((self.__address, self.__port), RequestValidator)
        self.__service_thread = threading.Thread(target=self.__server.serve_forever)

    def __enter__(self):
        self.__service_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__server.shutdown()
        self.__server.server_close()
        self.__service_thread.join()

    def ReceivedRequest(self):
        return self.__results["request_count"] > 0

    def UsedVirtualHosted(self):
        # Virtual-hosted only when at least one request arrived and every
        # request used the bucket-prefixed host.
        return (
            self.__results["request_count"] > 0
            and self.__results["virtual_hosted_count"] > 0
            and self.__results["path_style_count"] == 0
        )

    def UsedPathStyle(self):
        return (
            self.__results["request_count"] > 0
            and self.__results["path_style_count"] > 0
            and self.__results["virtual_hosted_count"] == 0
        )

    def Results(self):
        return dict(self.__results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bucket", type=str, default="dummy-bucket")
    parser.add_argument(
        "--expect",
        choices=["virtual", "path"],
        required=True,
        help="Addressing style the Triton S3 client is expected to use.",
    )
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    service = MockS3AddressingService(port=args.port, bucket=args.bucket)
    with service:
        elapsed = 0
        # Wait until at least one request is observed or timeout.
        while not service.ReceivedRequest() and elapsed < args.timeout:
            elapsed += 1
            time.sleep(1)
        # Give the client a brief window to send follow-up requests so the
        # classification reflects the full exchange.
        time.sleep(2)

    results = service.Results()
    if args.expect == "virtual":
        passed = service.UsedVirtualHosted()
    else:
        passed = service.UsedPathStyle()

    print("Observed requests:", results)
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print("TEST FAILED")
        sys.exit(1)
