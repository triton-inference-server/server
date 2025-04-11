# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/abstract_tool_parser.py
# Copyright 2024 The vLLM team.
from typing import Callable, Dict, List, Optional, Union

from engine.utils.tokenizer import AnyTokenizer
from schemas.openai import (
    ChatCompletionMessageToolCalls,
    ChatCompletionStreamResponseDelta,
)


class ToolCallParser:
    """The Base Tool Call Parser for parsing the Tool Call from the responses,
    Two inferfaces are supported: the one-time parser for synchronized response
    and streaming parser for streaming response.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        self.prev_tool_call_arr: List[Dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

        self.model_tokenizer = tokenizer

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionMessageToolCalls:
        raise NotImplementedError(
            "BaseToolCallParser.parse_tool_calls has not been implemented!"
        )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> ChatCompletionStreamResponseDelta:
        raise NotImplementedError(
            "BaseToolCallParser.parse_tool_calls_streaming has not been implemented!"
        )


class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def get_tool_parser_cls(cls, name) -> type:
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool parser: '{name}' not found in tool_call_parsers")

    @classmethod
    def _register_module(
        cls,
        module: type,
        module_name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
    ) -> None:
        if not issubclass(module, ToolCallParser):
            raise TypeError(
                f"module must be subclass of ToolCallParser, but got {type(module)}"
            )
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(
                    f"{name} is already registered " f"at {existed_module.__module__}"
                )
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
        module: Union[type, None] = None,
    ) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                "name must be None, an instance of str, " f"but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register
