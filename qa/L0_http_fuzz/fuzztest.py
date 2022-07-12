# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import sys

sys.path.append("../common")

import unittest
import test_util as tu
import sqlite3
from boofuzz import *
import glob
import os


class FuzzTest(tu.TestResultCollector):

    def _run_fuzz(self, url, logger):
        session = Session(
            target=Target(connection=TCPSocketConnection("127.0.0.1", 8000)),
            fuzz_loggers=logger,
            keep_web_open=False)

        s_initialize(name="Request" + url)
        with s_block("Request-Line"):
            s_group("Method", [
                "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS",
                "TRACE"
            ])
            s_delim(" ", name="space-1")
            s_string(url, name="Request-URI")
            s_delim(" ", name="space-2")
            s_string("HTTP/1.1", name="HTTP-Version")
            s_static("\r\n", name="Request-Line-CRLF")
        s_static("\r\n", "Request-CRLF")

        session.connect(s_get("Request" + url))
        session.fuzz()

    def test_failures_from_db(self):
        url_list = [
            "/v2", "/v2/models/simple", "/v2/models/simple/infer",
            "/v2/models/simple/versions/v1", "/v2/models/simple/config",
            "/v2/models/simple/stats", "/v2/models/simple/ready",
            "/v2/health/ready", "/v2/health/live", "/v2/repository/index",
            "/v2/repository/models/simple/unload",
            "/v2/repository/models/simple/load",
            "/v2/systemsharedmemory/status", "/v2/systemsharedmemory/register",
            "/v2/systemsharedmemory/unregister",
            "/v2/systemsharedmemory/region/xx/status",
            "/v2/cudasharedmemory/status", "/v2/cudasharedmemory/register",
            "/v2/cudasharedmemory/unregister",
            "/v2/cudasharedmemory/region/xx/status"
        ]

        csv_log = open('fuzz_results.csv', 'w')
        logger = [FuzzLoggerCsv(file_handle=csv_log)]

        for url in url_list:
            self._run_fuzz(url, logger)

            # Get latest db file
            files = glob.glob('boofuzz-results/*')
            dbfile = max(files, key=os.path.getctime)

            conn = sqlite3.connect(dbfile)
            c = conn.cursor()

            # Get number of failures, should be 0
            self.assertEqual(
                len([
                    x for x in c.execute(
                        "SELECT * FROM steps WHERE type=\"fail\"")
                ]), 0)


if __name__ == "__main__":
    unittest.main()
