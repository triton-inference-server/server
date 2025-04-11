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
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/llama_tool_parser.py
# Copyright 2024 The vLLM team.
import json
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

from .utils import find_common_prefix, is_complete_json, partial_json_loads


@ToolParserManager.register_module("llama3")
class Llama3JsonToolParser(ToolCallParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        self.bot_token = "<|python_tag|>"

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionResponseMessage:
        """
        Extract the tool calls from a complete model response.
        """
        # case -- if a tool call token is not present, return a text response
        if not (full_text.startswith(self.bot_token) or full_text.startswith("{")):
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

        original_full_text = full_text
        try:
            # FIXME: tensorrt_llm backend might generate some unnecessary text messages
            # after the tool call json text starting with "assistant\n\n".
            if backend != "vllm":
                last_index = full_text.find("assistant\n\n")
                if last_index > 0:
                    full_text = full_text[:last_index]

            # load the JSON, and then use it to build the Function and
            # Tool Call
            dec = json.JSONDecoder()
            function_call_arr = []

            # depending on the prompt format the Llama model may or may not
            # prefix the output with the <|python_tag|> token
            start_idx = (
                len(self.bot_token) if full_text.startswith(self.bot_token) else 0
            )
            while start_idx < len(full_text):
                (obj, end_idx) = dec.raw_decode(full_text[start_idx:])
                start_idx += end_idx + len("; ")
                function_call_arr.append(obj)

            tool_calls = ChatCompletionMessageToolCalls(
                root=[
                    ChatCompletionMessageToolCall(
                        id=f"cmpl-{uuid.uuid1()}",
                        type="function",
                        function=Function1(
                            name=raw_function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                raw_function_call["arguments"]
                                if "arguments" in raw_function_call
                                else raw_function_call["parameters"]
                            ),
                        ),
                    )
                    for raw_function_call in function_call_arr
                ]
            )

            # get any content before the tool call
            ret = ChatCompletionResponseMessage(
                tool_calls=tool_calls, content="", role=role
            )
            return ret

        except Exception as e:
            # return information to just treat the tool call as regular JSON
            return ChatCompletionResponseMessage(
                tool_calls=None, content=original_full_text, role=role
            )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> Union[ChatCompletionStreamResponseDelta, None]:
        if not (
            current_text.startswith(self.bot_token) or current_text.startswith("{")
        ):
            return ChatCompletionStreamResponseDelta(content=delta_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(
                        is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None

            # select as the current tool call the one we're on the state at
            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        delta = ChatCompletionStreamResponseDelta(
                            tool_calls=[
                                ChatCompletionMessageToolCallChunk(
                                    index=self.current_tool_id,
                                    function=Function2(
                                        arguments=argument_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return delta

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                type="function",
                                id=f"cmpl-{uuid.uuid1()}",
                                function=Function2(name=function_name).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = ChatCompletionStreamResponseDelta(
                            tool_calls=[
                                ChatCompletionMessageToolCallChunk(
                                    index=self.current_tool_id,
                                    function=Function2(
                                        arguments=argument_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            return None
