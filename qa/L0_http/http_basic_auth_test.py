#!/usr/bin/python
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
import sys
import unittest

sys.path.append("../common")

import test_util as tu
import tritonclient.http as tritonhttpclient
import tritonclient.http.aio as asynctritonhttpclient
from tritonclient.http.aio.auth import BasicAuth as AsyncBasicAuth
from tritonclient.http.auth import BasicAuth


class HTTPBasicAuthTest(tu.TestResultCollector):
    def setUp(self):
        # Use the nginx port
        self._client = tritonhttpclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(BasicAuth("username", "password"))

    def test_client_call(self):
        self.assertTrue(self._client.is_server_live())

    def tearDown(self):
        self._client.close()


class HTTPBasicAuthAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Use the nginx port
        self._client = asynctritonhttpclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(AsyncBasicAuth("username", "password"))

    async def test_client_call(self):
        self.assertTrue(await self._client.is_server_live())

    async def asyncTearDown(self):
        await self._client.close()


if __name__ == "__main__":
    unittest.main()
