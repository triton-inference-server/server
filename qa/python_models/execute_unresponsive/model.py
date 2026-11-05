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

import os
import signal
import subprocess
import threading
import time

import triton_python_backend_utils as pb_utils

# A negative input stops the stub process for longer than the server waits for
# a health update, so the server stops waiting for the response while the
# request is still being executed.
STALL_SECONDS = 3


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
            value = float(in_tensor.as_numpy()[0])
            if value < 0:
                pid = os.getpid()
                subprocess.Popen(
                    ["/bin/sh", "-c", f"sleep {STALL_SECONDS}; kill -CONT {pid}"],
                    start_new_session=True,
                )
                os.kill(pid, signal.SIGSTOP)
                # This thread logs after the execution has finished, which it
                # can only do once the stub releases the GIL.
                threading.Thread(target=self._log_finished, args=(value,)).start()
            out_tensor = pb_utils.Tensor("OUT", in_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def _log_finished(self, value):
        time.sleep(1)
        pb_utils.Logger.log_info(f"Execution with input {value:.0f} finished")
