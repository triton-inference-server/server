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

import json
import threading

import triton_python_backend_utils as pb_utils

# Nothing cancels the requests, so the test fails if the log has this message.
CANCELLED_MESSAGE = "Request reported as cancelled"


class TritonPythonModel:
    """Asks whether requests were cancelled after the server released them.

    Each execute first checks the requests of the previous execute, whose
    response factories the server has released by then. The decoupled model
    also checks each request right after sending its final response. A negative
    input makes execute raise, so the server releases that request on the
    error path. While a request is released, a thread keeps asking whether it
    was cancelled, so a query is often in flight at that moment.
    """

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self._decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        self._released = []
        self._pollers = []

    def execute(self, requests):
        for request in self._released:
            self._check(request)
        self._stop_pollers()
        self._released = list(requests)
        for request in requests:
            self._start_poller(request)

        responses = []
        for request in requests:
            in_np = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            if (in_np < 0).any():
                raise pb_utils.TritonModelException("Negative input")
            response = pb_utils.InferenceResponse([pb_utils.Tensor("OUTPUT0", in_np)])
            if self._decoupled:
                request.get_response_sender().send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                self._check(request)
            else:
                responses.append(response)
        return None if self._decoupled else responses

    def finalize(self):
        self._stop_pollers()

    def _start_poller(self, request):
        started = threading.Event()
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                cancelled = request.is_cancelled()
                started.set()
                if cancelled:
                    pb_utils.Logger.log_error(CANCELLED_MESSAGE)
                    return

        thread = threading.Thread(target=poll)
        thread.start()
        started.wait()
        self._pollers.append((stop, thread))

    def _stop_pollers(self):
        for stop, thread in self._pollers:
            stop.set()
            thread.join()
        self._pollers = []

    @staticmethod
    def _check(request):
        if request.is_cancelled() or request.get_response_sender().is_cancelled():
            # The error response of a request that already sent its final
            # response is dropped, so the log carries the failure too.
            pb_utils.Logger.log_error(CANCELLED_MESSAGE)
            raise pb_utils.TritonModelException(CANCELLED_MESSAGE)
