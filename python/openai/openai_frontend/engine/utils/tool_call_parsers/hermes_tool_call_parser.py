# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form reproduce the above copyright
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
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py
# Copyright 2024 The vLLM team.

import ast
import json
import re
import uuid
from typing import Union

import partial_json_parser
from engine.utils.tokenizer import AnyTokenizer
from engine.utils.tool_call_parsers.tool_call_parser import (
    ToolCallParser,
    ToolParserManager,
)
from partial_json_parser.core.options import Allow
from schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Function1,
    Function2,
)


@ToolParserManager.register_module("hermes")
class HermesToolParser(ToolCallParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        self.scratch_pad_regex = re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]

        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]

        self.buffered_delta_text = ""

    def tool_call_delta_buffer(self, delta_text: str):
        if (
            delta_text in self.tool_call_start_token_array
            or delta_text in self.tool_call_end_token_array
        ):
            if (
                delta_text == self.tool_call_start_token_array[-1]
                or delta_text == self.tool_call_end_token_array[-1]
            ):
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionResponseMessage:
        if self.tool_call_start_token not in full_text:
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

        try:
            function_call_tuples = self.tool_call_regex.findall(full_text)

            def parse_function_call(s):
                s = s.strip()
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Unable to parse function call: {s}")

            raw_function_calls = [
                parse_function_call(match[0] if match[0] else match[1])
                for match in function_call_tuples
            ]
            tool_calls = ChatCompletionMessageToolCalls(
                root=[
                    ChatCompletionMessageToolCall(
                        id=f"cmpl-{uuid.uuid1()}",
                        type="function",
                        function=Function1(
                            name=function_call["name"],
                            arguments=json.dumps(
                                function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for function_call in raw_function_calls
                ]
            )

            content = full_text[: full_text.find(self.tool_call_start_token)] or ""
            return ChatCompletionResponseMessage(
                tool_calls=tool_calls, content=content, role=role
            )

        except Exception as e:
            print(f"Exception during tool parse: {e}")
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> Union[ChatCompletionStreamResponseDelta, None]:
        delta_text = self.tool_call_delta_buffer(delta_text)
        if (
            len(current_text) >= len(self.buffered_delta_text)
            and current_text[-len(self.buffered_delta_text) :]
            == self.buffered_delta_text
        ):
            current_text = current_text[: -len(self.buffered_delta_text)] + delta_text

        if self.tool_call_start_token not in current_text:
            return ChatCompletionStreamResponseDelta(content=delta_text)

        try:
            prev_tool_start_count = current_text.count(self.tool_call_start_token) - (
                1 if delta_text == self.tool_call_start_token else 0
            )
            prev_tool_end_count = current_text.count(self.tool_call_end_token) - (
                1 if delta_text == self.tool_call_end_token else 0
            )
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)

            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                return ChatCompletionStreamResponseDelta(content=delta_text)

            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                tool_call_portion = (
                    current_text.split(self.tool_call_start_token)[-1]
                    if len(delta_text) > 1
                    else None
                )
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                delta = None

            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]

            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count > prev_tool_end_count
            ):
                if not self.prev_tool_call_arr or len(self.prev_tool_call_arr) == 0:
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                function=Function2(arguments=diff),
                            )
                        ]
                    )

            else:
                text = delta_text.replace(self.tool_call_start_token, "").replace(
                    self.tool_call_end_token, ""
                )
                return ChatCompletionStreamResponseDelta(content=text)

            try:
                current_tool_call = (
                    partial_json_parser.loads(tool_call_portion or "{}", flags)
                    if tool_call_portion
                    else None
                )
            except (
                partial_json_parser.core.exceptions.MalformedJSON,
                json.JSONDecodeError,
            ):
                return None

            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                type="function",
                                id=f"cmpl-{uuid.uuid1()}",
                                function=Function2(name=function_name),
                            )
                        ]
                    )
                else:
                    return None

            if tool_call_portion is None:
                return (
                    ChatCompletionStreamResponseDelta(content=delta_text)
                    if delta_text
                    else None
                )

            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            if not cur_arguments and not prev_arguments:
                delta = None
            elif not cur_arguments and prev_arguments:
                delta = None
            elif cur_arguments and not prev_arguments:
                function_name = current_tool_call.get("name")
                match = re.search(
                    r'\{"name":\s*"'
                    + re.escape(function_name)
                    + r'"\s*,\s*"arguments":\s*(.*)',
                    tool_call_portion.strip(),
                    re.DOTALL,
                )
                if match:
                    cur_arguments_json = match.group(1)
                else:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                if delta_text not in cur_arguments_json:
                    return None
                args_delta_start_loc = cur_arguments_json.rindex(delta_text) + len(
                    delta_text
                )
                arguments_delta = cur_arguments_json[:args_delta_start_loc]

                delta = ChatCompletionStreamResponseDelta(
                    tool_calls=[
                        ChatCompletionMessageToolCallChunk(
                            index=self.current_tool_id,
                            function=Function2(arguments=arguments_delta),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

            elif cur_arguments and prev_arguments:
                try:
                    json.loads(tool_call_portion)
                    is_complete_json = True
                except Exception:
                    is_complete_json = False

                if (
                    isinstance(delta_text, str)
                    and len(delta_text.rstrip()) >= 1
                    and delta_text.rstrip()[-1] == "}"
                    and is_complete_json
                ):
                    delta_text = delta_text.rstrip()[:-1]

                delta = ChatCompletionStreamResponseDelta(
                    tool_calls=[
                        ChatCompletionMessageToolCallChunk(
                            index=self.current_tool_id,
                            function=Function2(arguments=delta_text),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += delta_text

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            return None
