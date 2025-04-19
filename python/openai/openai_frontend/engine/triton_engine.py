# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import tritonserver
from engine.engine import LLMEngine
from engine.utils.chat import load_chat_template, parse_chat_messages
from engine.utils.tokenizer import get_tokenizer
from engine.utils.tool_call_parsers import ToolCallParser, ToolParserManager
from engine.utils.triton import (
    _create_trtllm_inference_request,
    _create_vllm_inference_request,
    _get_output,
    _get_vllm_lora_names,
    _validate_triton_responses_non_streaming,
)
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionNamedToolChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption1,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    Function1,
    Function2,
    Model,
    ObjectType,
)


# TODO: Improve type hints
@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model
    # Tokenizers used for chat templates
    tokenizer: Optional[Any]
    # LoRA names supported by the backend
    lora_names: Optional[List[str]]
    # Time that model was loaded by Triton
    create_time: int
    # Conversion format between OpenAI and Triton requests
    request_converter: Callable


class TritonLLMEngine(LLMEngine):
    def __init__(
        self,
        server: tritonserver.Server,
        tokenizer: str,
        backend: Optional[str] = None,
        lora_separator: Optional[str] = None,
        tool_call_parser: Optional[str] = None,
        chat_template: Optional[str] = None,
    ):
        # Assume an already configured and started server
        self.server = server
        self.tokenizer = self._get_tokenizer(tokenizer)
        # TODO: Reconsider name of "backend" vs. something like "request_format"
        self.backend = backend
        self.lora_separator = lora_separator

        # NOTE: Creation time and model metadata will be static at startup for
        # now, and won't account for dynamically loading/unloading models.
        self.create_time = int(time.time())
        self.model_metadata = self._get_model_metadata()
        self.tool_call_parser = (
            ToolParserManager.get_tool_parser_cls(tool_call_parser)
            if tool_call_parser
            else None
        )
        self.chat_template = load_chat_template(chat_template)

    def ready(self) -> bool:
        return self.server.ready()

    def metrics(self) -> str:
        return self.server.metrics()

    def models(self) -> List[Model]:
        models = []
        for metadata in self.model_metadata.values():
            model_names = [metadata.name]
            if (
                self.lora_separator is not None
                and len(self.lora_separator) > 0
                and metadata.lora_names is not None
            ):
                for lora_name in metadata.lora_names:
                    model_names.append(
                        f"{metadata.name}{self.lora_separator}{lora_name}"
                    )

            for model_name in model_names:
                models.append(
                    Model(
                        id=model_name,
                        created=metadata.create_time,
                        object=ObjectType.model,
                        owned_by="Triton Inference Server",
                    ),
                )

        return models

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | AsyncIterator[str]:
        model_name, lora_name = self._get_model_and_lora_name(request.model)
        metadata = self.model_metadata.get(model_name)
        self._validate_chat_request(request, metadata, lora_name)

        conversation = parse_chat_messages(request.messages)

        add_generation_prompt = True

        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )

        prompt = metadata.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tool_dicts,
            chat_template=self.chat_template,
        )

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(metadata.model, prompt, request, lora_name)
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        default_role = "assistant"
        role = self._get_first_response_role(
            conversation, add_generation_prompt, default_role
        )

        tool_call_parser = (
            self.tool_call_parser(metadata.tokenizer) if self.tool_call_parser else None
        )

        if request.stream:
            return self._streaming_chat_iterator(
                request_id,
                metadata.backend,
                created,
                request,
                role,
                tool_call_parser,
                responses,
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        response_message, finish_reason = self._get_chat_completion_response_message(
            request=request,
            request_id=request_id,
            tool_call_parser=tool_call_parser,
            text=text,
            role=role,
            backend=metadata.backend,
        )

        return CreateChatCompletionResponse(
            id=request_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            created=created,
            model=request.model,
            system_fingerprint=None,
            object=ObjectType.chat_completion,
        )

    def _get_chat_completion_response_message(
        self,
        request: CreateChatCompletionRequest,
        request_id: str,
        tool_call_parser: ToolCallParser,
        text: str,
        role: str,
        backend: str,
    ) -> Tuple[ChatCompletionResponseMessage, ChatCompletionFinishReason]:
        response_message: ChatCompletionResponseMessage
        auto_tools_called = False
        tool_function_name = self._get_named_function_name(request=request)
        if tool_function_name:
            response_message = ChatCompletionResponseMessage(
                content="",
                role=role,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=request_id,
                        type="function",
                        function=Function1(name=tool_function_name, arguments=text),
                    )
                ],
            )
        elif (
            tool_call_parser
            and request.tools
            and (
                request.tool_choice is None
                or request.tool_choice.root == ChatCompletionToolChoiceOption1.auto
            )
        ):
            response_message = tool_call_parser.parse_tool_calls(text, role, backend)
            auto_tools_called = (
                response_message.tool_calls is not None
                and len(response_message.tool_calls.root) > 0
            )
        else:
            response_message = ChatCompletionResponseMessage(
                content=text, role=role, tool_calls=None
            )

        finish_reason = (
            ChatCompletionFinishReason.tool_calls
            if auto_tools_called
            else ChatCompletionFinishReason.stop
        )

        return response_message, finish_reason

    async def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | AsyncIterator[str]:
        # Validate request and convert to Triton format
        model_name, lora_name = self._get_model_and_lora_name(request.model)
        metadata = self.model_metadata.get(model_name)
        self._validate_completion_request(request, metadata, lora_name)

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(
                metadata.model, request.prompt, request, lora_name
            )
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        if request.stream:
            return self._streaming_completion_iterator(
                request_id, created, request.model, responses
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        choice = Choice(
            finish_reason=FinishReason.stop,
            index=0,
            logprobs=None,
            text=text,
        )
        return CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=request.model,
        )

    # TODO: This behavior should be tested further
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    # TODO: Expose explicit flag to catch edge cases
    def _determine_request_converter(self, backend: str):
        # Allow manual override of backend request format if provided by user
        if self.backend:
            backend = self.backend

        # Request conversion from OpenAI format to backend-specific format
        if backend == "vllm":
            return _create_vllm_inference_request

        # Use TRT-LLM format as default for everything else. This could be
        # an ensemble, a python or BLS model, a TRT-LLM backend model, etc.
        return _create_trtllm_inference_request

    def _get_model_and_lora_name(self, request_model_name: str):
        if self.lora_separator is None or len(self.lora_separator) == 0:
            return request_model_name, None

        names = request_model_name.split(self.lora_separator)
        if len(names) != 2:
            return request_model_name, None

        return names[0], names[1]

    def _get_tokenizer(self, tokenizer_name: str):
        tokenizer = None
        if tokenizer_name:
            tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def _get_model_metadata(self) -> Dict[str, TritonModelMetadata]:
        # One tokenizer and creation time shared for all loaded models for now.
        model_metadata = {}

        # Read all triton models and store the necessary metadata for each
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            backend = model.config()["backend"]
            # Explicitly handle ensembles to avoid any runtime validation errors
            if not backend and model.config()["platform"] == "ensemble":
                backend = "ensemble"
            print(f"Found model: {name=}, {backend=}")

            lora_names = None
            if self.backend == "vllm" or backend == "vllm":
                lora_names = _get_vllm_lora_names(
                    self.server.options.model_repository, name, model.version
                )

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=self.tokenizer,
                lora_names=lora_names,
                create_time=self.create_time,
                request_converter=self._determine_request_converter(backend),
            )
            model_metadata[name] = metadata

        return model_metadata

    def _get_streaming_chat_response_chunk(
        self,
        choice: ChatCompletionStreamingResponseChoice,
        request_id: str,
        created: int,
        model: str,
    ) -> CreateChatCompletionStreamResponse:
        return CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
        )

    def _get_first_streaming_chat_response(
        self, request_id: str, created: int, model: str, role: str
    ) -> CreateChatCompletionStreamResponse:
        # First chunk has no content and sets the role
        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=role, content="", function_call=None
            ),
            logprobs=None,
            finish_reason=None,
        )
        chunk = self._get_streaming_chat_response_chunk(
            choice, request_id, created, model
        )
        return chunk

    async def _streaming_chat_iterator(
        self,
        request_id: str,
        backend: str,
        created: int,
        request: CreateChatCompletionRequest,
        role: str,
        tool_call_parser: ToolCallParser,
        responses: AsyncIterable,
    ) -> AsyncIterator[str]:
        model = request.model

        tool_function_name = self._get_named_function_name(request=request)

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            tool_call_parser
            and not tool_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        previous_text = ""

        chunk = self._get_first_streaming_chat_response(
            request_id, created, model, role
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        async for response in responses:
            delta_text = _get_output(response)

            (
                response_delta,
                finish_reason,
                current_text,
            ) = self._get_streaming_response_delta(
                previous_text=previous_text,
                delta_text=delta_text,
                tool_function_name=tool_function_name,
                tool_choice_auto=tool_choice_auto,
                tool_call_parser=tool_call_parser,
                backend=backend,
                is_final_response=response.final,
            )
            previous_text = current_text

            # if the response delta is None (e.g. because it was a
            # "control token" for tool calls or the parser otherwise
            # wasn't ready to send a token, then
            # get the next token without streaming a chunk
            if response_delta is None and finish_reason is None:
                continue

            if finish_reason and response_delta is None:
                response_delta = ChatCompletionStreamResponseDelta(content="")

            choice = ChatCompletionStreamingResponseChoice(
                index=0,
                delta=response_delta,
                logprobs=None,
                finish_reason=finish_reason,
            )

            chunk = self._get_streaming_chat_response_chunk(
                choice, request_id, created, model
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _get_streaming_response_delta(
        self,
        previous_text: str,
        delta_text: str,
        tool_function_name: Optional[str],
        tool_choice_auto: bool,
        tool_call_parser: ToolCallParser,
        backend: str,
        is_final_response: bool,
    ) -> Tuple[
        Optional[ChatCompletionStreamResponseDelta],
        Optional[ChatCompletionFinishReason],
        str,
    ]:
        response_delta: Optional[ChatCompletionStreamResponseDelta]
        current_text = ""
        if tool_function_name:
            response_delta = ChatCompletionStreamResponseDelta(
                tool_calls=[
                    ChatCompletionMessageToolCallChunk(
                        index=0,
                        function=Function2(
                            name=tool_function_name, arguments=delta_text
                        ),
                    )
                ]
            )
        elif tool_choice_auto:
            current_text = previous_text + delta_text
            response_delta = tool_call_parser.parse_tool_calls_streaming(
                current_text=current_text, delta_text=delta_text, backend=backend
            )
        else:
            response_delta = ChatCompletionStreamResponseDelta(
                role=None, content=delta_text, function_call=None
            )

        if is_final_response:
            auto_tools_called = False
            if tool_call_parser:
                auto_tools_called = len(tool_call_parser.prev_tool_call_arr) > 0
                index = (
                    len(tool_call_parser.prev_tool_call_arr) - 1
                    if auto_tools_called
                    else 0
                )
            else:
                index = 0

            # check to make sure we haven't "forgotten" to stream
            # any tokens that were generated but previously
            # matched by partial json parsing, such as '}'.
            # only happens if we are NOT using guided decoding
            if (
                self._should_check_for_unstreamed_tool_arg_tokens(
                    response_delta=response_delta,
                    auto_tools_called=auto_tools_called,
                )
                and tool_call_parser
            ):
                latest_delta_len = 0
                if (
                    isinstance(response_delta.tool_calls[0].function, Function2)
                ) and isinstance(response_delta.tool_calls[0].function.arguments, str):
                    latest_delta_len = len(
                        response_delta.tool_calls[0].function.arguments
                    )
                # get the expected call based on partial JSON
                # parsing which "autocompletes" the JSON
                expected_call = json.dumps(
                    tool_call_parser.prev_tool_call_arr[index].get("arguments", {}),
                    ensure_ascii=False,
                )
                # get what we've streamed so far for arguments
                # for the current tool
                actual_call = tool_call_parser.streamed_args_for_tool[index]
                if latest_delta_len > 0:
                    actual_call = actual_call[:-latest_delta_len]

                # check to see if there's anything left to stream
                remaining_call = expected_call.replace(actual_call, "", 1)

                response_delta = ChatCompletionStreamResponseDelta(
                    tool_calls=[
                        ChatCompletionMessageToolCallChunk(
                            index=index,
                            function=Function2(arguments=remaining_call).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )

            finish_reason = (
                ChatCompletionFinishReason.tool_calls
                if auto_tools_called
                else ChatCompletionFinishReason.stop
            )
        else:
            finish_reason = None

        return response_delta, finish_reason, current_text

    def _validate_chat_request(
        self,
        request: CreateChatCompletionRequest,
        metadata: TritonModelMetadata,
        lora_name: str | None,
    ):
        """
        Validates a chat request to align with currently supported features.
        """

        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.tokenizer:
            raise Exception("Unknown tokenizer")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        if (
            metadata.lora_names is not None
            and lora_name is not None
            and lora_name not in metadata.lora_names
        ):
            raise Exception(f"Unknown LoRA: {lora_name}; for model: {request.model}")

        # Reject unsupported features if requested
        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs:
            raise Exception("logit bias and log probs not currently supported")

        self._verify_chat_tool_call_settings(request=request)

    def _verify_chat_tool_call_settings(self, request: CreateChatCompletionRequest):
        if (
            request.tool_choice
            and request.tool_choice.root == ChatCompletionToolChoiceOption1.required
            and not request.tools
        ):
            raise Exception(
                '"required" tool choice requires CreateChatCompletionRequest.tools to be provided'
            )

        if (
            request.tool_choice
            and isinstance(request.tool_choice.root, ChatCompletionNamedToolChoice)
            and not request.tools
        ):
            raise Exception(
                "Named tool choice requires CreateChatCompletionRequest.tools to be provided"
            )

        if (
            request.tool_choice
            and request.tool_choice.root == ChatCompletionToolChoiceOption1.auto
            and self.tool_call_parser is None
        ):
            raise Exception('"auto" tool choice requires --tool-call-parser to be set')

        if (
            request.tool_choice is None
            and request.tools
            and self.tool_call_parser is None
        ):
            raise Exception(
                "having tools in the request requires --tool-call-parser to be set"
            )

    async def _streaming_completion_iterator(
        self, request_id: str, created: int, model: str, responses: AsyncIterable
    ) -> AsyncIterator[str]:
        async for response in responses:
            text = _get_output(response)
            choice = Choice(
                finish_reason=FinishReason.stop if response.final else None,
                index=0,
                logprobs=None,
                text=text,
            )
            chunk = CreateCompletionResponse(
                id=request_id,
                choices=[choice],
                system_fingerprint=None,
                object=ObjectType.text_completion,
                created=created,
                model=model,
            )

            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_completion_request(
        self,
        request: CreateCompletionRequest,
        metadata: TritonModelMetadata,
        lora_name: str | None,
    ):
        """
        Validates a completions request to align with currently supported features.
        """
        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        if (
            metadata.lora_names is not None
            and lora_name is not None
            and lora_name not in metadata.lora_names
        ):
            raise Exception(f"Unknown LoRA: {lora_name}; for model: {request.model}")

        # Reject unsupported features if requested
        if request.suffix is not None:
            raise Exception("suffix is not currently supported")

        if not request.prompt:
            raise Exception("prompt must be non-empty")

        # Currently only support single string as input
        if not isinstance(request.prompt, str):
            raise Exception("only single string input is supported")

        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.best_of and request.best_of > 1:
            raise Exception(
                f"Received best_of={request.best_of}, but only single choice (best_of=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs is not None:
            raise Exception("logit bias and log probs not supported")

    def _should_stream_with_auto_tool_parsing(
        self, request: CreateChatCompletionRequest
    ):
        has_tools = request.tools and self.tool_call_parser
        auto_tool = (
            request.tool_choice is None
            or request.tool_choice.root == ChatCompletionToolChoiceOption1.auto
        )
        return has_tools and auto_tool

    def _should_check_for_unstreamed_tool_arg_tokens(
        self, response_delta: ChatCompletionStreamResponseDelta, auto_tools_called
    ):
        return bool(
            auto_tools_called
            and self.tool_call_parser
            and response_delta
            and response_delta.tool_calls
            and response_delta.tool_calls[0]
            and response_delta.tool_calls[0].function
            and response_delta.tool_calls[0].function.arguments is not None
        )

    def _get_named_function_name(
        self, request: CreateChatCompletionRequest
    ) -> Optional[str]:
        if request.tool_choice and isinstance(
            request.tool_choice.root, ChatCompletionNamedToolChoice
        ):
            tool_choice_function_name = request.tool_choice.root.function.name
        else:
            tool_choice_function_name = None

        if (
            request.tool_choice
            and request.tool_choice.root == ChatCompletionToolChoiceOption1.required
        ):
            tool_choice_required_function_name = request.tools[0].function.name
        else:
            tool_choice_required_function_name = None

        return tool_choice_function_name or tool_choice_required_function_name
