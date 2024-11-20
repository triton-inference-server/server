# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import Union

import tritonserver
from pydantic import Field
from pydantic.dataclasses import dataclass
from tritonfrontend._api._error_mapping import handle_triton_error
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError,
    TritonFrontendHttp,
)


class KServeHttp:
    @dataclass
    class Options:
        address: str = "0.0.0.0"
        port: int = Field(8000, ge=0, le=65535)
        reuse_port: bool = False
        thread_count: int = Field(8, gt=0)
        header_forward_pattern: str = ""
        # DLIS-7215: Add restricted protocol support
        # restricted_protocols: list

    @handle_triton_error
    def __init__(self, server: tritonserver, options: "KServeHttp.Options" = None):
        server_ptr = server._ptr()  # TRITONSERVER_Server pointer

        # If no options provided, default options are selected
        if options is None:
            options = KServeHttp.Options()

        if not isinstance(options, KServeHttp.Options):
            raise InvalidArgumentError(
                "Incorrect type for options. options argument must be of type KServeHttp.Options"
            )

        # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>>
        options_dict: dict[str, Union[int, bool, str]] = options.__dict__

        self.triton_frontend = TritonFrontendHttp(server_ptr, options_dict)

    def __enter__(self):
        self.triton_frontend.start()
        return self

    @handle_triton_error
    def __exit__(self, exc_type, exc_value, traceback):
        self.triton_frontend.stop()
        if exc_type:
            raise exc_type(exc_value)

    @handle_triton_error
    def start(self):
        self.triton_frontend.start()

    @handle_triton_error
    def stop(self):
        self.triton_frontend.stop()
