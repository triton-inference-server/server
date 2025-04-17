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
import json
import os
from typing import Dict, List, Optional

import openai
import pytest
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
)

# resources for testing the tool callings
WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, "
                    "e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "must the two-letter abbreviation for the state "
                    "that the city is in, e.g. 'CA' which would "
                    "mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

WEATHER_FORECAST_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, "
                    "e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "must the two-letter abbreviation for the state "
                    "that the city is in, e.g. 'CA' which would "
                    "mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast",
                },
            },
            "required": ["city", "state", "unit", "num_days"],
        },
    },
}

MESSAGES_ASKING_FOR_TOOLS: List[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You're a helpful assistant! Answer the users question best you can.",
    },
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"},
]

MESSAGES_WITH_TOOL_RESPONSE: List[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You're a helpful assistant! Answer the users question best you can.",
    },
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "123456789",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"city": "Dallas", "state": "TX", '
                    '"unit": "fahrenheit"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "123456789", "content": "98"},
]

WEATHER_FORECAST_TOOL_CHOICE: ChatCompletionNamedToolChoiceParam = {
    "function": {"name": "get_n_day_weather_forecast"},
    "type": "function",
}


@pytest.mark.openai
class TestAsyncClientToolCalling:
    @pytest.fixture(scope="class")
    def client(self, server):
        return server.get_async_client()

    def validate_tool_calls_present(
        self, tool_calls: Optional[List[ChatCompletionMessageToolCall]], skip_id=False
    ):
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function is not None
        assert isinstance(tool_calls[0].id, str)
        if not skip_id:
            assert len(tool_calls[0].id) >= 9

    def validate_weather_tool_arguments(self, parsed_arguments: Dict):
        assert isinstance(parsed_arguments, Dict)
        assert isinstance(parsed_arguments.get("city"), str)
        assert isinstance(parsed_arguments.get("state"), str)
        assert isinstance(parsed_arguments.get("unit"), str)
        assert parsed_arguments.get("city") == "Dallas"
        assert parsed_arguments.get("state") in ("TX", "Texas")
        assert parsed_arguments.get("unit") == "fahrenheit"

    def validate_weather_forcast_tool_arguments(self, parsed_arguments: Dict):
        assert isinstance(parsed_arguments, Dict)
        assert isinstance(parsed_arguments.get("city"), str)
        assert isinstance(parsed_arguments.get("state"), str)
        assert isinstance(parsed_arguments.get("unit"), str)
        assert isinstance(parsed_arguments.get("num_days"), int)
        assert parsed_arguments.get("city") == "Dallas"
        assert parsed_arguments.get("state") in ("TX", "Texas")
        assert parsed_arguments.get("unit") == "fahrenheit"

    @pytest.mark.asyncio
    async def test_tool_call_and_choice(self, client: openai.AsyncOpenAI, model: str):
        chat_completion = await client.chat.completions.create(
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            model=model,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
        )

        choice = chat_completion.choices[0]
        stop_reason = chat_completion.choices[0].finish_reason
        tool_calls = chat_completion.choices[0].message.tool_calls

        # make sure a tool call is present
        self.validate_tool_calls_present(tool_calls)
        assert stop_reason == "tool_calls"

        # make sure the weather tool was called (classic example) with arguments
        assert tool_calls[0].function.name == WEATHER_TOOL["function"]["name"]
        assert tool_calls[0].function.arguments is not None
        assert isinstance(tool_calls[0].function.arguments, str)

        # make sure the arguments parse properly
        parsed_arguments = json.loads(tool_calls[0].function.arguments)
        self.validate_weather_tool_arguments(parsed_arguments)

        function_name: Optional[str] = None
        function_args_str: str = ""
        tool_call_id: Optional[str] = None
        role_name: Optional[str] = None
        finish_reason_count: int = 0

        # make the same request, streaming
        stream = await client.chat.completions.create(
            model=model,
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
            stream=True,
        )

        async for chunk in stream:
            assert chunk.choices[0].index == 0

            if chunk.choices[0].finish_reason:
                finish_reason_count += 1
                assert chunk.choices[0].finish_reason == "tool_calls"

            # if a role is being streamed make sure it wasn't already set to
            # something else
            if chunk.choices[0].delta.role:
                assert not role_name or role_name == "assistant"
                role_name = "assistant"

            # if a tool call is streamed make sure there's exactly one
            # (based on the request parameters
            streamed_tool_calls = chunk.choices[0].delta.tool_calls

            if streamed_tool_calls and len(streamed_tool_calls) > 0:
                assert len(streamed_tool_calls) == 1
                tool_call = streamed_tool_calls[0]

                # if a tool call ID is streamed, make sure one hasn't been already
                if tool_call.id:
                    assert not tool_call_id
                    tool_call_id = tool_call.id

                # if parts of the function start being streamed
                if tool_call.function:
                    # if the function name is defined, set it. it should be streamed
                    # IN ENTIRETY, exactly one time.
                    if tool_call.function.name:
                        assert function_name is None
                        assert isinstance(tool_call.function.name, str)
                        function_name = tool_call.function.name
                    if tool_call.function.arguments:
                        assert isinstance(tool_call.function.arguments, str)
                        function_args_str += tool_call.function.arguments

        assert finish_reason_count == 1
        assert role_name == "assistant"
        assert isinstance(tool_call_id, str) and (len(tool_call_id) >= 9)

        # validate the name and arguments
        assert function_name == WEATHER_TOOL["function"]["name"]
        assert function_name == tool_calls[0].function.name
        assert isinstance(function_args_str, str)

        # validate arguments
        streamed_args = json.loads(function_args_str)
        self.validate_weather_tool_arguments(streamed_args)

        # make sure everything matches non-streaming except for ID
        assert function_name == tool_calls[0].function.name
        assert choice.message.role == role_name
        assert choice.message.tool_calls[0].function.name == function_name

        # compare streamed with non-streamed args Dict-wise, not string-wise
        # because character-to-character comparison might not work e.g. the tool
        # call parser adding extra spaces or something like that. we care about the
        # dicts matching not byte-wise match
        assert parsed_arguments == streamed_args

    @pytest.mark.asyncio
    async def test_tool_call_with_reply_response(
        self, client: openai.AsyncOpenAI, model: str, backend: str
    ):
        chat_completion = await client.chat.completions.create(
            messages=MESSAGES_WITH_TOOL_RESPONSE,
            temperature=0,
            max_tokens=128,
            model=model,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
        )

        choice = chat_completion.choices[0]

        assert choice.finish_reason != "tool_calls"  # "stop"
        assert choice.message.role == "assistant"
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content is not None

        stream = await client.chat.completions.create(
            messages=MESSAGES_WITH_TOOL_RESPONSE,
            temperature=0,
            max_tokens=128,
            model=model,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
            stream=True,
        )

        chunks: List[str] = []
        finish_reason_count = 0
        role_sent: bool = False

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.role:
                assert not role_sent
                assert delta.role == "assistant"
                role_sent = True

            if delta.content:
                chunks.append(delta.content)

            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
                assert chunk.choices[0].finish_reason == choice.finish_reason

            assert not delta.tool_calls or len(delta.tool_calls) == 0

        assert role_sent
        assert finish_reason_count == 1
        assert len(chunks)

        # validate if steaming and non-streaming generates the same content
        assert "".join(chunks) == choice.message.content

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") == "TRTLLM",
        reason="latest release version of Tensorrt LLM 0.18 doesn't support guided decoding",
    )
    @pytest.mark.asyncio
    async def test_tool_call_with_named_tool_choice(
        self, client: openai.AsyncOpenAI, model: str
    ):
        chat_completion = await client.chat.completions.create(
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            model=model,
            tool_choice=WEATHER_FORECAST_TOOL_CHOICE,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
        )

        choice = chat_completion.choices[0]
        stop_reason = chat_completion.choices[0].finish_reason
        tool_calls = chat_completion.choices[0].message.tool_calls

        # make sure a tool call is present
        self.validate_tool_calls_present(tool_calls, skip_id=True)
        assert stop_reason != "tool_calls"

        # make sure the weather tool was called (classic example) with arguments
        assert tool_calls[0].function.name == WEATHER_FORECAST_TOOL["function"]["name"]
        assert tool_calls[0].function.arguments is not None
        assert isinstance(tool_calls[0].function.arguments, str)

        # make sure the arguments parse properly
        parsed_arguments = json.loads(tool_calls[0].function.arguments)
        self.validate_weather_forcast_tool_arguments(parsed_arguments)

        function_name: Optional[str] = None
        function_args_str: str = ""
        tool_call_id: Optional[str] = None
        role_name: Optional[str] = None
        finish_reason_count: int = 0

        # make the same request, streaming
        stream = await client.chat.completions.create(
            model=model,
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            tool_choice=WEATHER_FORECAST_TOOL_CHOICE,
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
            stream=True,
        )

        async for chunk in stream:
            assert chunk.choices[0].index == 0

            if chunk.choices[0].finish_reason:
                finish_reason_count += 1
                assert chunk.choices[0].finish_reason != "tool_calls"

            # if a role is being streamed make sure it wasn't already set to
            # something else
            if chunk.choices[0].delta.role:
                assert not role_name or role_name == "assistant"
                role_name = "assistant"

            # if a tool call is streamed make sure there's exactly one
            # (based on the request parameters
            streamed_tool_calls = chunk.choices[0].delta.tool_calls

            if streamed_tool_calls and len(streamed_tool_calls) > 0:
                assert len(streamed_tool_calls) == 1
                tool_call = streamed_tool_calls[0]

                # if a tool call ID is streamed, make sure one hasn't been already
                if tool_call.id:
                    assert not tool_call_id
                    tool_call_id = tool_call.id

                # if parts of the function start being streamed
                if tool_call.function:
                    # if the function name is defined, set it. it should be streamed
                    # IN ENTIRETY, exactly one time.
                    if tool_call.function.name:
                        assert isinstance(tool_call.function.name, str)
                        function_name = tool_call.function.name
                    if tool_call.function.arguments:
                        assert isinstance(tool_call.function.arguments, str)
                        function_args_str += tool_call.function.arguments

        assert finish_reason_count == 1
        assert role_name == "assistant"

        # validate the name and arguments
        assert function_name == WEATHER_FORECAST_TOOL["function"]["name"]
        assert function_name == tool_calls[0].function.name
        assert isinstance(function_args_str, str)

        # validate arguments
        streamed_args = json.loads(function_args_str)
        self.validate_weather_forcast_tool_arguments(streamed_args)

        # make sure everything matches non-streaming except for ID
        assert function_name == tool_calls[0].function.name
        assert choice.message.role == role_name
        assert choice.message.tool_calls[0].function.name == function_name

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") == "TRTLLM",
        reason="latest release version of Tensorrt LLM 0.18 doesn't support guided decoding",
    )
    @pytest.mark.asyncio
    async def test_tool_call_with_required_tool_choice(
        self, client: openai.AsyncOpenAI, model: str
    ):
        chat_completion = await client.chat.completions.create(
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            model=model,
            tool_choice="required",
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
        )

        choice = chat_completion.choices[0]
        stop_reason = chat_completion.choices[0].finish_reason
        tool_calls = chat_completion.choices[0].message.tool_calls

        # make sure a tool call is present
        self.validate_tool_calls_present(tool_calls, skip_id=True)
        assert stop_reason != "tool_calls"

        # make sure the weather tool was called (classic example) with arguments
        assert tool_calls[0].function.name == WEATHER_TOOL["function"]["name"]
        assert tool_calls[0].function.arguments is not None
        assert isinstance(tool_calls[0].function.arguments, str)

        # make sure the arguments parse properly
        parsed_arguments = json.loads(tool_calls[0].function.arguments)
        self.validate_weather_tool_arguments(parsed_arguments)

        function_name: Optional[str] = None
        function_args_str: str = ""
        tool_call_id: Optional[str] = None
        role_name: Optional[str] = None
        finish_reason_count: int = 0

        # make the same request, streaming
        stream = await client.chat.completions.create(
            model=model,
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_tokens=128,
            tool_choice="required",
            tools=[WEATHER_TOOL, WEATHER_FORECAST_TOOL],
            logprobs=False,
            stream=True,
        )

        async for chunk in stream:
            assert chunk.choices[0].index == 0

            if chunk.choices[0].finish_reason:
                finish_reason_count += 1
                assert chunk.choices[0].finish_reason != "tool_calls"

            # if a role is being streamed make sure it wasn't already set to
            # something else
            if chunk.choices[0].delta.role:
                assert not role_name or role_name == "assistant"
                role_name = "assistant"

            # if a tool call is streamed make sure there's exactly one
            # (based on the request parameters
            streamed_tool_calls = chunk.choices[0].delta.tool_calls

            if streamed_tool_calls and len(streamed_tool_calls) > 0:
                assert len(streamed_tool_calls) == 1
                tool_call = streamed_tool_calls[0]

                # if a tool call ID is streamed, make sure one hasn't been already
                if tool_call.id:
                    assert not tool_call_id
                    tool_call_id = tool_call.id

                # if parts of the function start being streamed
                if tool_call.function:
                    # if the function name is defined, set it. it should be streamed
                    # IN ENTIRETY, exactly one time.
                    if tool_call.function.name:
                        assert isinstance(tool_call.function.name, str)
                        function_name = tool_call.function.name
                    if tool_call.function.arguments:
                        assert isinstance(tool_call.function.arguments, str)
                        function_args_str += tool_call.function.arguments

        assert finish_reason_count == 1
        assert role_name == "assistant"

        # validate the name and arguments
        assert function_name == WEATHER_TOOL["function"]["name"]
        assert function_name == tool_calls[0].function.name
        assert isinstance(function_args_str, str)

        # validate arguments
        streamed_args = json.loads(function_args_str)
        self.validate_weather_tool_arguments(streamed_args)

        # make sure everything matches non-streaming except for ID
        assert function_name == tool_calls[0].function.name
        assert choice.message.role == role_name
        assert choice.message.tool_calls[0].function.name == function_name

    @pytest.mark.asyncio
    async def test_inconsistent_tool_choice_and_tools(
        self, client: openai.AsyncOpenAI, model: str
    ):
        # tool choice function but the tools are empty
        with pytest.raises(openai.BadRequestError):
            await client.chat.completions.create(
                messages=MESSAGES_ASKING_FOR_TOOLS,
                temperature=0,
                max_tokens=128,
                model=model,
                tool_choice=WEATHER_FORECAST_TOOL_CHOICE,
                logprobs=False,
            )
        # tool choice function that is not provided in the tools
        with pytest.raises(openai.BadRequestError):
            await client.chat.completions.create(
                messages=MESSAGES_ASKING_FOR_TOOLS,
                temperature=0,
                max_tokens=128,
                model=model,
                tool_choice=WEATHER_FORECAST_TOOL_CHOICE,
                tools=[WEATHER_TOOL],
                logprobs=False,
            )

        # tool choice required but tools is empty
        with pytest.raises(openai.BadRequestError):
            await client.chat.completions.create(
                messages=MESSAGES_ASKING_FOR_TOOLS,
                temperature=0,
                max_tokens=128,
                model=model,
                tool_choice="required",
                tools=[],
                logprobs=False,
            )
