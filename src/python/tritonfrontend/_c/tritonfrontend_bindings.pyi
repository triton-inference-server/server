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

from tritonfrontend import AlreadyExistsError as AlreadyExistsError
from tritonfrontend import InternalError as InternalError
from tritonfrontend import InvalidArgumentError as InvalidArgumentError
from tritonfrontend import NotFoundError as NotFoundError
from tritonfrontend import TritonError as TritonError
from tritonfrontend import UnavailableError as UnavailableError
from tritonfrontend import UnknownError as UnknownError
from tritonfrontend import UnsupportedError as UnsupportedError

class TritonFrontendGrpc:
    def __init__(self, arg0: int, arg1: dict[str, bool | int | str]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class TritonFrontendHttp:
    def __init__(self, arg0: int, arg1: dict[str, bool | int | str]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class TritonFrontendMetrics:
    def __init__(self, arg0: int, arg1: dict[str, bool | int | str]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...