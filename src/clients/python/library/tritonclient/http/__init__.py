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

try:
    from geventhttpclient import HTTPClient
    from geventhttpclient.url import URL
    import gevent
    import gevent.pool
    from urllib.parse import quote, quote_plus
    import rapidjson as json
    import numpy as np
    import struct
except ModuleNotFoundError as error:
    raise RuntimeError(
        'The installation does not include http support. Specify \'http\' or \'all\' while installing the tritonclient package to include the support'
    ) from error

from tritonclient.utils import *


def _get_error(response):
    """
    Returns the InferenceServerException object if response
    indicates the error. If no error then return None
    """
    if response.status_code != 200:
        error_response = json.loads(response.read())
        return InferenceServerException(msg=error_response["error"])
    else:
        return None


def _raise_if_error(response):
    """
    Raise InferenceServerException if received non-Success
    response from the server
    """
    error = _get_error(response)
    if error is not None:
        raise error


def _get_query_string(query_params):
    params = []
    for key, value in query_params.items():
        if isinstance(value, list):
            for item in value:
                params.append("%s=%s" %
                              (quote_plus(key), quote_plus(str(item))))
        else:
            params.append("%s=%s" % (quote_plus(key), quote_plus(str(value))))
    if params:
        return "&".join(params)
    return ''


def _get_inference_request(inputs, request_id, outputs, sequence_id,
                           sequence_start, sequence_end, priority, timeout):
    infer_request = {}
    parameters = {}
    if request_id != "":
        infer_request['id'] = request_id
    if sequence_id != 0:
        parameters['sequence_id'] = sequence_id
        parameters['sequence_start'] = sequence_start
        parameters['sequence_end'] = sequence_end
    if priority != 0:
        parameters['priority'] = priority
    if timeout is not None:
        parameters['timeout'] = timeout

    infer_request['inputs'] = [
        this_input._get_tensor() for this_input in inputs
    ]
    if outputs:
        infer_request['outputs'] = [
            this_output._get_tensor() for this_output in outputs
        ]
    else:
        # no outputs specified so set 'binary_data_output' True in the
        # request so that all outputs are returned in binary format
        parameters['binary_data_output'] = True

    if parameters:
        infer_request['parameters'] = parameters

    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)
        return request_body, json_size

    return request_body, None


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using http protocol. None
    of the methods are thread safe. The object is intended to be used
    by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined
    behavior.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8000'.
    verbose : bool
        If True generate verbose output. Default value is False.
    concurrency : int
        The number of connections to create for this client.
        Default value is 1.
    connection_timeout : float
        The timeout value for the connection. Default value
        is 60.0 sec.
    network_timeout : float
        The timeout value for the network. Default value is
        60.0 sec
    max_greenlets : int
        Determines the maximum allowed number of worker greenlets
        for handling asynchronous inference requests. Default value
        is None, which means there will be no restriction on the
        number of greenlets created.
    ssl : bool
        If True, channels the requests to encrypted https scheme.
        Default value is False.
    ssl_options : dict
        Any options supported by `ssl.wrap_socket` specified as
        dictionary. The argument is ignored if 'ssl' is specified
        False.
    ssl_context_factory : SSLContext callable
        It must be a callbable that returns a SSLContext. The default
        value is None which use `ssl.create_default_context`. The
        argument is ignored if 'ssl' is specified False.
    insecure : bool
        If True, then does not match the host name with the certificate.
        Default value is False. The argument is ignored if 'ssl' is
        specified False.

    Raises
        ------
        Exception
            If unable to create a client.

    """

    def __init__(self,
                 url,
                 verbose=False,
                 concurrency=1,
                 connection_timeout=60.0,
                 network_timeout=60.0,
                 max_greenlets=None,
                 ssl=False,
                 ssl_options=None,
                 ssl_context_factory=None,
                 insecure=False):
        scheme = "https://" if ssl else "http://"
        self._parsed_url = URL(scheme + url)
        self._client_stub = HTTPClient.from_url(
            self._parsed_url,
            concurrency=concurrency,
            connection_timeout=connection_timeout,
            network_timeout=network_timeout,
            ssl_options=ssl_options,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure)
        self._pool = gevent.pool.Pool(max_greenlets)
        self._verbose = verbose

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
        self._pool.join()
        self._client_stub.close()

    def _get(self, request_uri, headers, query_params):
        """Issues the GET request to the server

         Parameters
        ----------
        request_uri: str
            The request URI to be used in GET request.
        headers: dict
            Additional HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        geventhttpclient.response.HTTPSocketPoolResponse
            The response from server.
        """
        if query_params is not None:
            request_uri = request_uri + "?" + _get_query_string(query_params)

        if self._verbose:
            print("GET {}, headers {}".format(request_uri, headers))

        if headers is not None:
            response = self._client_stub.get(request_uri, headers=headers)
        else:
            response = self._client_stub.get(request_uri)

        if self._verbose:
            print(response)

        return response

    def _post(self, request_uri, request_body, headers, query_params):
        """Issues the POST request to the server

        Parameters
        ----------
        request_uri: str
            The request URI to be used in POST request.
        request_body: str
            The body of the request
        headers: dict
            Additional HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        geventhttpclient.response.HTTPSocketPoolResponse
            The response from server.
        """
        if query_params is not None:
            request_uri = request_uri + "?" + _get_query_string(query_params)

        if self._verbose:
            print("POST {}, headers {}\n{}".format(request_uri, headers,
                                                   request_body))

        if headers is not None:
            response = self._client_stub.post(request_uri=request_uri,
                                              body=request_body,
                                              headers=headers)
        else:
            response = self._client_stub.post(request_uri=request_uri,
                                              body=request_body)

        if self._verbose:
            print(response)

        return response

    def is_server_live(self, headers=None, query_params=None):
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        Exception
            If unable to get liveness.

        """

        request_uri = "v2/health/live"
        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)

        return response.status_code == 200

    def is_server_ready(self, headers=None, query_params=None):
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        Exception
            If unable to get readiness.

        """
        request_uri = "v2/health/ready"
        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)

        return response.status_code == 200

    def is_model_ready(self,
                       model_name,
                       model_version="",
                       headers=None,
                       query_params=None):
        """Contact the inference server and get the readiness of specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.
        model_version: str
            The version of the model to check for readiness. The default value
            is an empty string which means then the server will choose a version
            based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        Exception
            If unable to get model readiness.

        """
        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/ready".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/ready".format(quote(model_name))

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)

        return response.status_code == 200

    def get_server_metadata(self, headers=None, query_params=None):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """
        request_uri = "v2"
        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def get_model_metadata(self,
                           model_name,
                           model_version="",
                           headers=None,
                           query_params=None):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get metadata. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """
        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}".format(quote(model_name))

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def get_model_config(self,
                         model_name,
                         model_version="",
                         headers=None,
                         query_params=None):
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get configuration. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model config.

        Raises
        ------
        InferenceServerException
            If unable to get model configuration.

        """
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/config".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/config".format(quote(model_name))

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def get_model_repository_index(self, headers=None, query_params=None):
        """Get the index of model repository contents

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model repository index.

        Raises
        ------
        InferenceServerException
            If unable to get the repository index.

        """
        request_uri = "v2/repository/index"
        response = self._post(request_uri=request_uri,
                              request_body="",
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def load_model(self, model_name, headers=None, query_params=None):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        request_uri = "v2/repository/models/{}/load".format(quote(model_name))
        response = self._post(request_uri=request_uri,
                              request_body="",
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            print("Loaded model '{}'".format(model_name))

    def unload_model(self, model_name, headers=None, query_params=None):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """
        request_uri = "v2/repository/models/{}/unload".format(quote(model_name))
        response = self._post(request_uri=request_uri,
                              request_body="",
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            print("Loaded model '{}'".format(model_name))

    def get_inference_statistics(self,
                                 model_name="",
                                 model_version="",
                                 headers=None,
                                 query_params=None):
        """Get the inference statistics for the specified model name and
        version.

        Parameters
        ----------
        model_name : str
            The name of the model to get statistics. The default value is
            an empty string, which means statistics of all models will
            be returned.
        model_version: str
            The version of the model to get inference statistics. The
            default value is an empty string which means then the server
            will return the statistics of all available model versions.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model inference statistics.

        Raises
        ------
        InferenceServerException
            If unable to get the model inference statistics.

        """

        if model_name != "":
            if type(model_version) != str:
                raise_error("model version must be a string")
            if model_version != "":
                request_uri = "v2/models/{}/versions/{}/stats".format(
                    quote(model_name), model_version)
            else:
                request_uri = "v2/models/{}/stats".format(quote(model_name))
        else:
            request_uri = "v2/models/stats"

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def get_system_shared_memory_status(self,
                                        region_name="",
                                        headers=None,
                                        query_params=None):
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if region_name != "":
            request_uri = "v2/systemsharedmemory/region/{}/status".format(
                quote(region_name))
        else:
            request_uri = "v2/systemsharedmemory/status"

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def register_system_shared_memory(self,
                                      name,
                                      key,
                                      byte_size,
                                      offset=0,
                                      headers=None,
                                      query_params=None):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        key : str
            The key of the underlying memory object that contains the
            system shared memory region.
        byte_size : int
            The size of the system shared memory region, in bytes.
        offset : int
            Offset, in bytes, within the underlying memory object to
            the start of the system shared memory region. The default
            value is zero.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory.

        """
        request_uri = "v2/systemsharedmemory/region/{}/register".format(
            quote(name))

        register_request = {
            'key': key,
            'offset': offset,
            'byte_size': byte_size
        }
        request_body = json.dumps(register_request)

        response = self._post(request_uri=request_uri,
                              request_body=request_body,
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            print("Registered system shared memory with name '{}'".format(name))

    def unregister_system_shared_memory(self,
                                        name="",
                                        headers=None,
                                        query_params=None):
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region.

        """
        if name != "":
            request_uri = "v2/systemsharedmemory/region/{}/unregister".format(
                quote(name))
        else:
            request_uri = "v2/systemsharedmemory/unregister"

        response = self._post(request_uri=request_uri,
                              request_body="",
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            if name is not "":
                print("Unregistered system shared memory with name '{}'".format(
                    name))
            else:
                print("Unregistered all system shared memory regions")

    def get_cuda_shared_memory_status(self,
                                      region_name="",
                                      headers=None,
                                      query_params=None):
        """Request cuda shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active cuda shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding cuda shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if region_name != "":
            request_uri = "v2/cudasharedmemory/region/{}/status".format(
                quote(region_name))
        else:
            request_uri = "v2/cudasharedmemory/status"

        response = self._get(request_uri=request_uri,
                             headers=headers,
                             query_params=query_params)
        _raise_if_error(response)

        content = response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    def register_cuda_shared_memory(self,
                                    name,
                                    raw_handle,
                                    device_id,
                                    byte_size,
                                    headers=None,
                                    query_params=None):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        raw_handle : bytes
            The raw serialized cudaIPC handle in base64 encoding.
        device_id : int
            The GPU device ID on which the cudaIPC handle was created.
        byte_size : int
            The size of the cuda shared memory region, in bytes.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to register the specified cuda shared memory.

        """
        request_uri = "v2/cudasharedmemory/region/{}/register".format(
            quote(name))

        register_request = {
            'raw_handle': {
                'b64': raw_handle
            },
            'device_id': device_id,
            'byte_size': byte_size
        }
        request_body = json.dumps(register_request)

        response = self._post(request_uri=request_uri,
                              request_body=request_body,
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            print("Registered cuda shared memory with name '{}'".format(name))

    def unregister_cuda_shared_memory(self,
                                      name="",
                                      headers=None,
                                      query_params=None):
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region.

        """
        if name != "":
            request_uri = "v2/cudasharedmemory/region/{}/unregister".format(
                quote(name))
        else:
            request_uri = "v2/cudasharedmemory/unregister"

        response = self._post(request_uri=request_uri,
                              request_body="",
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)
        if self._verbose:
            if name is not "":
                print("Unregistered cuda shared memory with name '{}'".format(
                    name))
            else:
                print("Unregistered all cuda shared memory regions")

    def infer(self,
              model_name,
              inputs,
              model_version="",
              outputs=None,
              request_id="",
              sequence_id=0,
              sequence_start=False,
              sequence_end=False,
              priority=0,
              timeout=None,
              headers=None,
              query_params=None):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """

        request_body, json_size = _get_inference_request(
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout)

        if json_size is not None:
            if headers is None:
                headers = {}
            headers["Inference-Header-Content-Length"] = json_size

        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/infer".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/infer".format(quote(model_name))

        response = self._post(request_uri=request_uri,
                              request_body=request_body,
                              headers=headers,
                              query_params=query_params)
        _raise_if_error(response)

        return InferResult(response, self._verbose)

    def async_infer(self,
                    model_name,
                    inputs,
                    model_version="",
                    outputs=None,
                    request_id="",
                    sequence_id=0,
                    sequence_start=False,
                    sequence_end=False,
                    priority=0,
                    timeout=None,
                    headers=None,
                    query_params=None):
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'. Even though this call is
        non-blocking, however, the actual number of concurrent requests to
        the server will be limited by the 'concurrency' parameter specified
        while creating this client. In other words, if the inflight
        async_infer exceeds the specified 'concurrency', the delivery of
        the exceeding request(s) to server will be blocked till the slot is
        made available by retrieving the results of previously issued requests.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        InferAsyncRequest object
            The handle to the asynchronous inference request.

        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        def wrapped_post(request_uri, request_body, headers, query_params):
            return self._post(request_uri, request_body, headers, query_params)

        request_body, json_size = _get_inference_request(
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout)

        if json_size is not None:
            if headers is None:
                headers = {}
            headers["Inference-Header-Content-Length"] = json_size

        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/infer".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/infer".format(quote(model_name))

        g = self._pool.apply_async(
            wrapped_post, (request_uri, request_body, headers, query_params))

        # Schedule the greenlet to run in this loop iteration
        g.start()

        # Relinquish control to greenlet loop. Using non-zero
        # value to ensure the control is transferred to the
        # event loop.
        gevent.sleep(0.01)

        if self._verbose:
            verbose_message = "Sent request"
            if request_id is not "":
                verbose_message = verbose_message + " '{}'".format(request_id)
            print(verbose_message)

        return InferAsyncRequest(g, self._verbose)


class InferAsyncRequest:
    """An object of InferAsyncRequest class is used to describe
    a handle to an ongoing asynchronous inference request.

    Parameters
    ----------
    greenlet : gevent.Greenlet
        The greenlet object which will provide the results.
        For further details about greenlets refer
        http://www.gevent.org/api/gevent.greenlet.html.

    verbose : bool
        If True generate verbose output. Default value is False.
    """

    def __init__(self, greenlet, verbose=False):
        self._greenlet = greenlet
        self._verbose = verbose

    def get_result(self, block=True, timeout=None):
        """Get the results of the associated asynchronous inference.
        Parameters
        ----------
        block : bool
            If block is True, the function will wait till the
            corresponding response is received from the server.
            Default value is True.
        timeout : int
            The maximum wait time for the function. This setting is
            ignored if the block is set False. Default is None,
            which means the function will block indefinitely till
            the corresponding response is received.

        Returns
        -------
        InferResult
            The object holding the result of the async inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference or failed to respond
            within specified timeout.
        """

        try:
            response = self._greenlet.get(block=block, timeout=timeout)
        except gevent.Timeout as e:
            raise_error("failed to obtain inference response")

        _raise_if_error(response)
        return InferResult(response, self._verbose)


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.
    """

    def __init__(self, name, shape, datatype):
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._parameters = {}
        self._data = None
        self._raw_data = None

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._name

    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._datatype

    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        return self._shape

    def set_shape(self, shape):
        """Set the shape of input.

        Parameters
        ----------
        shape : list
            The shape of the associated input.
        """
        self._shape = shape

    def set_data_from_numpy(self, input_tensor, binary_data=True):
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        binary_data : bool
            Indicates whether to set data for the input in binary format
            or explicit tensor within JSON. The default value is True,
            which means the data will be delivered as binary data in the
            HTTP body after the JSON object.

        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        dtype = np_to_triton_dtype(input_tensor.dtype)
        if self._datatype != dtype:
            raise_error(
                "got unexpected datatype {} from numpy array, expected {}".
                format(dtype, self._datatype))
        valid_shape = True
        if len(self._shape) != len(input_tensor.shape):
            valid_shape = False
        else:
            for i in range(len(self._shape)):
                if self._shape[i] != input_tensor.shape[i]:
                    valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1],
                    str(self._shape)[1:-1]))

        self._parameters.pop('shared_memory_region', None)
        self._parameters.pop('shared_memory_byte_size', None)
        self._parameters.pop('shared_memory_offset', None)

        if not binary_data:
            self._parameters.pop('binary_data_size', None)
            self._raw_data = None
            if self._datatype == "BYTES":
                self._data = [val for val in input_tensor.flatten()]
            else:
                self._data = [val.item() for val in input_tensor.flatten()]
        else:
            self._data = None
            if self._datatype == "BYTES":
                self._raw_data = serialize_byte_tensor(input_tensor).tobytes()
            else:
                self._raw_data = input_tensor.tobytes()
            self._parameters['binary_data_size'] = len(self._raw_data)

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """
        self._data = None
        self._raw_data = None
        self._parameters.pop('binary_data_size', None)

        self._parameters['shared_memory_region'] = region_name
        self._parameters['shared_memory_byte_size'] = byte_size
        if offset != 0:
            self._parameters['shared_memory_offset'].int64_param = offset

    def _get_binary_data(self):
        """Returns the raw binary data if available

        Returns
        -------
        bytes
            The raw data for the input tensor
        """
        return self._raw_data

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.

        Returns
        -------
        dict
            The underlying tensor specification as dict
        """
        tensor = {
            'name': self._name,
            'shape': self._shape,
            'datatype': self._datatype
        }
        if self._parameters:
            tensor['parameters'] = self._parameters

        if self._parameters.get('shared_memory_region') is None and \
                self._raw_data is None:
            tensor['data'] = self._data
        return tensor


class InferRequestedOutput:
    """An object of InferRequestedOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object.
    binary_data : bool
        Indicates whether to return result data for the output in
        binary format or explicit tensor within JSON. The default
        value is True, which means the data will be delivered as
        binary data in the HTTP body after JSON object. This field
        will be unset if shared memory is set for the output.
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name, binary_data=True, class_count=0):
        self._name = name
        self._parameters = {}
        if class_count != 0:
            self._parameters['classification'] = class_count
        self._binary = binary_data
        self._parameters['binary_data'] = binary_data

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._name

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Marks the output to return the inference result in
        specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """
        if 'classification' in self._parameters:
            raise_error("shared memory can't be set on classification output")
        if self._binary:
            self._parameters['binary_data'] = False

        self._parameters['shared_memory_region'] = region_name
        self._parameters['shared_memory_byte_size'] = byte_size
        if offset != 0:
            self._parameters['shared_memory_offset'] = offset

    def unset_shared_memory(self):
        """Clears the shared memory option set by the last call to
        InferRequestedOutput.set_shared_memory(). After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

        self._parameters['binary_data'] = self._binary
        self._parameters.pop('shared_memory_region', None)
        self._parameters.pop('shared_memory_byte_size', None)
        self._parameters.pop('shared_memory_offset', None)

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.

        Returns
        -------
        dict
            The underlying tensor as a dict
        """
        tensor = {'name': self._name}
        if self._parameters:
            tensor['parameters'] = self._parameters
        return tensor


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : dict
        The inference response from the server
    verbose : bool
        If True generate verbose output. Default value is False.
    """

    def __init__(self, response, verbose):
        header_length = response.get('Inference-Header-Content-Length')
        if header_length is None:
            content = response.read()
            if verbose:
                print(content)
            self._result = json.loads(content)
        else:
            header_length = int(header_length)
            content = response.read(length=header_length)
            if verbose:
                print(content)
            self._result = json.loads(content)

            # Maps the output name to the index in buffer for quick retrieval
            self._output_name_to_buffer_map = {}
            # Read the remaining data off the response body.
            self._buffer = response.read()
            buffer_index = 0
            for output in self._result['outputs']:
                parameters = output.get("parameters")
                if parameters is not None:
                    this_data_size = parameters.get("binary_data_size")
                    if this_data_size is not None:
                        self._output_name_to_buffer_map[
                            output['name']] = buffer_index
                        buffer_index = buffer_index + this_data_size

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        if self._result.get('outputs') is not None:
            for output in self._result['outputs']:
                if output['name'] == name:
                    datatype = output['datatype']
                    has_binary_data = False
                    parameters = output.get("parameters")
                    if parameters is not None:
                        this_data_size = parameters.get("binary_data_size")
                        if this_data_size is not None:
                            has_binary_data = True
                            if this_data_size != 0:
                                start_index = self._output_name_to_buffer_map[
                                    name]
                                end_index = start_index + this_data_size
                                if datatype == 'BYTES':
                                    # String results contain a 4-byte string length
                                    # followed by the actual string characters. Hence,
                                    # need to decode the raw bytes to convert into
                                    # array elements.
                                    np_array = deserialize_bytes_tensor(
                                        self._buffer[start_index:end_index])
                                else:
                                    np_array = np.frombuffer(
                                        self._buffer[start_index:end_index],
                                        dtype=triton_to_np_dtype(datatype))
                            else:
                                np_array = np.empty(0)
                    if not has_binary_data:
                        np_array = np.array(output['data'],
                                            dtype=triton_to_np_dtype(datatype))
                    np_array = np.resize(np_array, output['shape'])
                    return np_array
        return None

    def get_output(self, name):
        """Retrieves the output tensor corresponding to the named ouput.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.

        Returns
        -------
        Dict
            If an output tensor with specified name is present in
            the infer resonse then returns it as a json dict,
            otherwise returns None.
        """
        for output in self._result['outputs']:
            if output['name'] == name:
                return output

        return None

    def get_response(self):
        """Retrieves the complete response

        Returns
        -------
        dict
            The underlying response dict.
        """
        return self._result
