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
import rapidjson as json
from google.protobuf import text_format

import tensorrtserver.api.request_status_pb2 as request_status
from tensorrtserverV2.common import *


def raise_if_error(nv_status):
    """
    Raise InferenceServerException if 'nv_status' is non-success.
    Otherwise return the request ID.
    """
    rstatus = text_format.Parse(nv_status, request_status.RequestStatus())
    if not rstatus.code == request_status.RequestStatusCode.SUCCESS:
        raise InferenceServerException(msg=rstatus.msg)
    else:
        return rstatus.request_id


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
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
        self.response = self._client_stub.get("/v2/health/live")
        self._last_request_id = raise_if_error(self.response['NV-Status'])
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
        self.response = self._client_stub.get("/v2/health/ready")
        self._last_request_id = raise_if_error(self.response['NV-Status'])
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
        raise_error('Not implemented')

    def get_server_metadata(self):
        """Contact the inference server and get its metadata.

        Returns
        -------
        object
            The JSON object holding the metadata

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """

        self.response = self._client_stub.get("/v2")
        self._last_request_id = raise_if_error(self.response['NV-Status'])

        status = json.loads(self.response.read())
        return status

    def get_model_metadata(self, model_name, model_version=-1):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model

        Returns
        -------
        object
            The JSON object holding the model metadata

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """
        #self.response = self._client_stub.get("/api/status/" + model_name +
        #                                      "?format=json")
        #self._last_request_id = raise_if_error(self.response['NV-Status'])

        #status = json.loads(self.response.read())
        #return status
        raise_error('Not implemented')
        return None

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
        #self.response = self._client_stub.post("/api/modelcontrol/load/" +
        #                                       model_name)
        #self._last_request_id = raise_if_error(self.response['NV-Status'])
        raise_error('Not implemented')
        return None

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
        #self.response = self._client_stub.post("/api/modelcontrol/unload/" +
        #                                       model_name)
        #self._last_request_id = raise_if_error(self.response['NV-Status'])
        raise_error('Not implemented')
        return None


#def infer(self, inputs, outputs, model_name, model_version=-1, batch_size=1, flags=0, correlation_id=0):
#    """Run inference using the supplied 'inputs' to calculate the outputs
#    specified by 'outputs'.
#       Parameters
#    ----------
#    inputs : list
#        A list of InferInput objects, each describing data for a input
#        tensor required by the model.
#       outputs : list
#        A list of InferOutput objects, each describing how the output
#        data must be returned. Only the output tensors present in the
#        list will be requested from the server.
#       batch_size : int
#        The batch size of the inference. Each input must provide
#        an appropriately sized batch of inputs.
#       flags : int
#        The flags to use for the inference. The bitwise-or of
#        InferRequestHeader.Flag values.
#       corr_id : int
#        The correlation id of the inference. Used to differentiate
#        sequences.
#       Returns
#    -------
#    InferResult
#        The object holding the result of the inference, including the
#        statistics.
#       Raises
#    ------
#    InferenceServerException
#        If server fails to perform inference.
#       """
