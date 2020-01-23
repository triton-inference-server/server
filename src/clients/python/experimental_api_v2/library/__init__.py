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

from geventhttpclient import HTTPClient
from geventhttpclient.url import URL

class InferenceServerContext:
    """An InferenceServerContext object is used to perform any kind of
    communication with the InferenceServer from client side.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : str
        The protocol used to communicate with the server.

    connection_count : int
        The number of connections to create for this context

    verbose : bool
        If True generate verbose output.

    Raises
        ------
        Exception
            If unable to create a context.

    """

    def __init__(self, url, protocol="http", connection_count=1, verbose=False):
        if protocol.lower() == "http":
            self.parsed_url = URL("http://" + url)
            self.client_stub = HTTPClient.from_url(self.parsed_url,
                                                   concurrency=connection_count)
        else:
            raise ValueError("unexpected protocol: " + protocol +
                                  ", expecting HTTP")
        self.verbose = verbose


    def is_server_live(self):
        """Contact the inference server and get liveness.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        Exception
            If unable to get liveness.

        """
        self.response = self.client_stub.get("/api/health/live")
        return self.response.status_code == 200

    def is_server_ready(self):
        """Contact the inference server and get readiness.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        Exception
            If unable to get readiness.

        """
        self.response = self.client_stub.get("/api/health/ready")
        return self.response.status_code == 200
