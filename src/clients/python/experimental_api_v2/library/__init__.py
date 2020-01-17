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
from geventhttpclient.connectionpool import ConnectionPool
from geventhttpclient.url import URL
from google.protobuf import text_format
import rapidjson as json

import tensorrtserverV2.api.request_status_pb2 as request_status

def _raise_error(msg):
    """
    Raise error with the provided message
    """
    rstatus = request_status.RequestStatus()
    rstatus.msg = msg
    raise InferenceServerException(rstatus, msg_only=True)

def _raise_if_error(nv_status):
    """
    Raise InferenceServerException if 'nv_status' is non-success.
    Otherwise return the request ID.
    """
    rstatus = text_format.Parse(nv_status, request_status.RequestStatus())
    if not rstatus.code == request_status.RequestStatusCode.SUCCESS:
        raise InferenceServerException(rstatus)
    else:
        return rstatus.request_id


class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : RequestStatus Protobuf
        The protobuf message describing the error

    """

    def __init__(self, err, msg_only=False):
        self._msg = err.msg
        if msg_only:
            self._server_id = None
            self._request_id = 0
        else:
            self._server_id = err.server_id
            self._request_id = err.request_id

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        if self._server_id is not None:
            msg = '[' + self._server_id + ' ' + str(
                self._request_id) + '] ' + msg
        return msg

    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        return self._msg

    def server_id(self):
        """Get the ID of the server associated with this exception.

        Returns
        -------
        str
            The ID of the server associated with this exception, or
            None if no server is associated.

        """
        return self._server_id

    def request_id(self):
        """Get the ID of the request with this exception.

        Returns
        -------
        int
            The ID of the request associated with this exception, or
            0 (zero) if no request is associated.

        """
        return self._request_id


class InferenceServerHTTPClient:
    """An InferenceServerHTTPClient object is used to perform any kind of
    communication with the InferenceServer using http protocol.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8000'.

    connection_count : int
        The number of connections to create for this client.
        Default value is 1.

    connection_timeout : float
        The timeout value for the connection. Default value
        is 60.0 sec.
    
    network_timeout : float
        The timeout value for the network. Default value is
        60.0 sec

    verbose : bool
        If True generate verbose output. Default value is False.

    Raises
        ------
        Exception
            If unable to create a client.

    """

    def __init__(self,
                 url,
                 connection_count=1,
                 connection_timeout=60.0,
                 network_timeout=60.0,
                 verbose=False):
        self._last_request_id = None
        self._parsed_url = URL("http://" + url)
        self._client_stub = HTTPClient.from_url(
            self._parsed_url,
            concurrency=connection_count,
            connection_timeout=connection_timeout,
            network_timeout=network_timeout)
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close the client. Any future calls to server
        will result in an Error.

        """
        self._client_stub.close()

    def get_last_request_id(self):
        """Get the request ID of the most recent request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id

    def is_server_live(self):
        """Contact the inference server and get liveness.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        InferenceServerException
            If unable to get liveness.

        """
        self.response = self._client_stub.get("/api/health/live")
        self._last_request_id = _raise_if_error(self.response['NV-Status'])
        return self.response.status_code == 200

    def is_server_ready(self):
        """Contact the inference server and get readiness.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        InferenceServerException
            If unable to get readiness.

        """
        self.response = self._client_stub.get("/api/health/ready")
        self._last_request_id = _raise_if_error(self.response['NV-Status'])
        return self.response.status_code == 200

    def is_model_ready(self, model_name, model_version=-1):
        """Contact the inference server and get the readiness of specified model.

         Parameters
        ----------
        model_name: str
            The name of the model

        model_version: int
            The version of the model. The default value is -1 which means by default
            the server will choose the model version based on its version policy
            for the model.

        Returns
        -------
        bool
            True if the model is ready to be served, False if otherwise.

        Raises
        ------
        InferenceServerException
            If unable to get model readiness.

        """
        self.response = self._client_stub.get("/api/status/" + model_name +
                                              "?format=json")
        self._last_request_id = _raise_if_error(self.response['NV-Status'])

        status = json.loads(self.response.read())
        if model_version == -1:
            _raise_error('Currently not supported. Will be implemented as part of HTTP V2 API')
        else:
            if model_name in status['modelStatus'].keys() and \
                str(model_version) in status['modelStatus']\
                [model_name]['versionStatus'].keys() and \
                status['modelStatus'][model_name]['versionStatus'] \
                [str(model_version)]["readyState"] == 'MODEL_READY':
                return True
            else:
                return False

    def get_server_status(self):
        """Contact the inference server and get its status.

        Returns
        -------
        object
            The JSON object holding the status

        Raises
        ------
        InferenceServerException
            If unable to get status.

        """

        self.response = self._client_stub.get("/api/status?format=json")
        self._last_request_id = _raise_if_error(self.response['NV-Status'])

        status = json.loads(self.response.read())
        return status

    def get_model_status(self, model_name):
        """Contact the inference server and get the status for specified model.

        Parameters
        ----------
        model_name: str
        The name of the model

        Returns
        -------
        object
            The JSON object holding the model status

        Raises
        ------
        InferenceServerException
            If unable to get model status.

        """

        self.response = self._client_stub.get("/api/status/" + model_name +
                                              "?format=json")
        self._last_request_id = _raise_if_error(self.response['NV-Status'])

        status = json.loads(self.response.read())
        return status

    def load_model(self, model_name):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        self.response = self._client_stub.post("/api/modelcontrol/load/" +
                                               model_name)
        self._last_request_id = _raise_if_error(self.response['NV-Status'])

    def unload_model(self, model_name):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """
        self.response = self._client_stub.post("/api/modelcontrol/unload/" +
                                               model_name)
        self._last_request_id = _raise_if_error(self.response['NV-Status'])
