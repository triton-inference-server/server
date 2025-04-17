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
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py
# Copyright 2024 The vLLM team.
import json
import re
from random import choices
from string import ascii_letters, digits
from typing import Dict, List, Union

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

from .utils import extract_intermediate_diff

ALPHANUMERIC = ascii_letters + digits


def generate_mistral_random_id():
    # Mistral Tool Call Ids must be alphanumeric with a maximum length of 9.
    # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
    return "".join(choices(ALPHANUMERIC, k=9))


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolCallParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[
            str
        ] = []  # map what has been streamed for each tool so far to a list
        self.bot_token = "[TOOL_CALLS]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionResponseMessage:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """

        # case -- if a tool call token is not present, return a text response
        if not (full_text.startswith(self.bot_token) or full_text.startswith("[")):
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

        # first remove the BOT token
        tool_content = full_text.replace(self.bot_token, "").strip()
        try:
            # we first try to directly load the json as parsing very nested
            # jsons is difficult
            try:
                function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's a easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            # Tool Call
            tool_calls = ChatCompletionMessageToolCalls(
                root=[
                    ChatCompletionMessageToolCall(
                        id=generate_mistral_random_id(),
                        type="function",
                        function=Function1(
                            name=raw_function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                raw_function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for raw_function_call in function_call_arr
                ]
            )

            # get any content before the tool call
            content = (
                full_text.split(self.bot_token)[0]
                if full_text.startswith(self.bot_token)
                else ""
            )
            return ChatCompletionResponseMessage(
                tool_calls=tool_calls, content=content, role=role
            )

        except Exception:
            # return information to just treat the tool call as regular JSON
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> Union[ChatCompletionStreamResponseDelta, None]:
        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        # tensorrt_llm backend likely doesn't generate the bos token
        if not (self.bot_token in current_text or "[" in current_text):
            return ChatCompletionStreamResponseDelta(content=delta_text)

        # handle if we detected the BOT token which means the start of tool
        # calling
        if self.bot_token == delta_text.strip():
            # if it's the only token, return None, so we don't send a chat
            # completion any don't send a control token
            return None

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            # replace BOT token with empty string, and convert single quotes
            # to double to allow parsing as JSON since mistral uses single
            # quotes instead of double for tool calls
            parsable_arr = current_text.split(self.bot_token)[-1]

            # tool calls are generated in an array, so do partial JSON
            # parsing on the entire array
            try:
                tool_call_arr: List[Dict] = partial_json_parser.loads(
                    parsable_arr, flags
                )
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None

            # select as the current tool call the one we're on the state at

            current_tool_call: Dict = (
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
                    diff: Union[str, None] = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )

                        delta = ChatCompletionStreamResponseDelta(
                            tool_calls=[
                                ChatCompletionMessageToolCallChunk(
                                    index=self.current_tool_id,
                                    function=Function2(arguments=diff).model_dump(
                                        exclude_none=True
                                    ),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return delta

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                type="function",
                                id=generate_mistral_random_id(),
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
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("'", '"')
                if '"}' in new_text:
                    new_text = new_text[: new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)[
                        :-2
                    ]

                    if new_text not in cur_arguments_json:
                        return None
                    arguments_delta = cur_arguments_json[
                        : cur_arguments_json.rindex(new_text) + len(new_text)
                    ]
                    delta = ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                function=Function2(
                                    arguments=arguments_delta
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )
                    delta = ChatCompletionStreamResponseDelta(
                        tool_calls=[
                            ChatCompletionMessageToolCallChunk(
                                index=self.current_tool_id,
                                function=Function2(arguments=argument_diff).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                else:
                    # try parsing it with regular JSON - if it works we're
                    # at the end, and we need to send the difference between
                    # tokens streamed so far and the valid JSON
                    delta = None

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            return None
