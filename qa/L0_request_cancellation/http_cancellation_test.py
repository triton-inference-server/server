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

import http.client
import json
import os
import time
import unittest


class HttpCancellationTest(unittest.TestCase):
    _model_name = "execute_cancel"
    _model_delay = 10.0  # seconds
    _cancelled_log = "[execute_cancel] Request cancelled at"

    def _infer_body(self, delay):
        return json.dumps(
            {
                "inputs": [
                    {
                        "name": "EXECUTE_DELAY",
                        "shape": [1, 1],
                        "datatype": "FP32",
                        "data": [delay],
                    }
                ]
            }
        )

    def _send_infer(self, conn, delay):
        conn.request(
            "POST",
            f"/v2/models/{self._model_name}/infer",
            body=self._infer_body(delay),
            headers={"Content-Type": "application/json"},
        )

    def _read_server_log(self):
        with open(os.environ["SERVER_LOG"]) as f:
            return f.read()

    def _wait_for_server_log(self, text, timeout):
        deadline = time.time() + timeout  # seconds
        while time.time() < deadline:
            if text in self._read_server_log():
                return True
            time.sleep(0.5)
        return False

    def _assert_cancelled_on_disconnect(self, conn):
        start_time = time.time()  # seconds
        time.sleep(2)  # ensure the inference has started
        conn.close()
        # The model checks for cancellation every second, so the cancellation
        # must be observed well before the model delay elapses.
        self.assertTrue(
            self._wait_for_server_log(self._cancelled_log, timeout=5),
            "request not cancelled after the client disconnected",
        )
        duration = time.time() - start_time  # seconds
        self.assertLess(duration, self._model_delay)

    def test_http_infer_client_disconnect(self):
        conn = http.client.HTTPConnection("localhost", 8000)
        self._send_infer(conn, self._model_delay)
        self._assert_cancelled_on_disconnect(conn)

    def test_http_generate_client_disconnect(self):
        conn = http.client.HTTPConnection("localhost", 8000)
        conn.request(
            "POST",
            f"/v2/models/{self._model_name}/generate",
            body=json.dumps({"EXECUTE_DELAY": [self._model_delay]}),
            headers={"Content-Type": "application/json"},
        )
        self._assert_cancelled_on_disconnect(conn)

    def test_http_infer_client_connected(self):
        conn = http.client.HTTPConnection("localhost", 8000)
        # Send more than one request to check that the keep-alive connection
        # is still usable after a request completes.
        for _ in range(2):
            self._send_infer(conn, 2.0)
            response = conn.getresponse()
            self.assertIn("not cancelled", response.read().decode())
        conn.close()
        self.assertNotIn(self._cancelled_log, self._read_server_log())


if __name__ == "__main__":
    unittest.main()
