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

import numpy as np
import triton_python_backend_utils as pb_utils

# Output is streamed in small fragments to mimic token-by-token generation,
# which exercises the streaming tool-call parser.
_CHUNK_SIZE = 8


def _decode_prompt(request):
    tensor = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
    value = tensor.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _response(text):
    return pb_utils.InferenceResponse(
        [
            pb_utils.Tensor("text_output", np.array([[text]], dtype=object)),
            pb_utils.Tensor("num_input_tokens", np.array([[1]], dtype=np.int32)),
            pb_utils.Tensor("num_output_tokens", np.array([[1]], dtype=np.int32)),
        ]
    )


class TritonPythonModel:
    """Decoupled mock that streams a Mistral-format tool call.

    The argument length is driven by the prompt so tests can exercise both
    sides of the streaming tool-call parse-size limit:
      * a prompt containing "big-tool-args" -> a large argument that exceeds
        the limit;
      * any other prompt -> a small argument that stays within it.
    """

    def execute(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            prompt = _decode_prompt(request)
            arg = "A" * (2000 if "big-tool-args" in prompt else 5)
            text = (
                '[TOOL_CALLS][{"name": "get_current_weather", '
                '"arguments": {"city": "' + arg + '"}}]'
            )
            for start in range(0, len(text), _CHUNK_SIZE):
                sender.send(_response(text[start : start + _CHUNK_SIZE]))
            sender.send(
                _response(""),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        return None
