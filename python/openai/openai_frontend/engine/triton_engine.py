# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
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
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import tritonserver
from engine.engine import LLMEngine
from engine.utils.chat import load_chat_template, parse_chat_messages
from engine.utils.tokenizer import get_tokenizer
from engine.utils.tool_call_parsers import ToolCallParser, ToolParserManager
from engine.utils.triton import (
    RequestKind,
    TritonLoraConfig,
    _create_trtllm_embedding_request,
    _create_trtllm_generate_request,
    _create_vllm_embedding_request,
    _create_vllm_generate_request,
    _get_openai_chat_format_logprobs_from_vllm_response,
    _get_openai_completion_format_logprobs_from_vllm_response,
    _get_output,
    _get_usage_from_response,
    _parse_lora_configs,
    _StreamingUsageAccumulator,
    _validate_triton_responses_non_streaming,
)
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionLogprobs,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionNamedToolChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption1,
    Choice,
    CompletionUsage,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    EmbeddingObject,
    FinishReason,
    Function1,
    Function2,
    Model,
    ObjectType,
)
from utils.utils import ClientError, ServerError


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
    lora_configs: Optional[List[TritonLoraConfig]]
    # Name of the input tensor enabling "echo" parameter in /v1/completions endpoint
    echo_tensor_name: Optional[str]
    # Time that model was loaded by Triton
    create_time: int
    # Conversion format between OpenAI and Triton requests
    inference_request_converter: Callable
    embedding_request_converter: Callable


class TritonLLMEngine(LLMEngine):
    def __init__(
        self,
        server: tritonserver.Server,
        tokenizer: str,
        default_max_tokens: int,
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
        self.default_max_tokens = default_max_tokens

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
                and metadata.lora_configs is not None
            ):
                for lora_config in metadata.lora_configs:
                    model_names.append(
                        f"{metadata.name}{self.lora_separator}{lora_config.name}"
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
            metadata.inference_request_converter(
                metadata.model,
                prompt,
                request,
                self._get_lora_config(model_name, lora_name),
                metadata.echo_tensor_name,
                self.default_max_tokens,
            )
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

        usage = _get_usage_from_response(
            response, metadata.backend, RequestKind.GENERATION
        )

        # Parse logprobs if requested
        logprobs_data = None
        if request.logprobs:
            openai_logprobs = _get_openai_chat_format_logprobs_from_vllm_response(
                response
            )
            if openai_logprobs:
                logprobs_data = ChatCompletionLogprobs(content=openai_logprobs)

        return CreateChatCompletionResponse(
            id=request_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    logprobs=logprobs_data,
                    finish_reason=finish_reason,
                )
            ],
            created=created,
            model=request.model,
            system_fingerprint=None,
            object=ObjectType.chat_completion,
            usage=usage,
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
            metadata.inference_request_converter(
                metadata.model,
                request.prompt,
                request,
                self._get_lora_config(model_name, lora_name),
                metadata.echo_tensor_name,
                self.default_max_tokens,
            )
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        if request.stream:
            return self._streaming_completion_iterator(
                request_id, created, request, responses, metadata.backend
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        usage = _get_usage_from_response(
            response, metadata.backend, RequestKind.GENERATION
        )

        # Parse logprobs if requested
        logprobs_data = None
        if request.logprobs is not None and request.logprobs > 0:
            logprobs_data = _get_openai_completion_format_logprobs_from_vllm_response(
                response
            )

        choice = Choice(
            finish_reason=FinishReason.stop,
            index=0,
            logprobs=logprobs_data,
            text=text,
        )
        return CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=request.model,
            usage=usage,
        )

    async def embedding(
        self, request: CreateEmbeddingRequest
    ) -> CreateEmbeddingResponse:
        # Validate request and convert to Triton format
        model_name, _ = self._get_model_and_lora_name(request.model)
        metadata = self.model_metadata.get(model_name)
        self._validate_embedding_request(request, metadata)

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.embedding_request_converter(
                metadata.model,
                request,
            )
        )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]

        # Extract embedding from response (currently stored as JSON string in text_output)
        embedding_json = _get_output(response)
        embedding_list = json.loads(embedding_json)

        usage = _get_usage_from_response(
            response, metadata.backend, RequestKind.EMBEDDING
        )

        embedding = self._get_embedding(embedding_list, request.encoding_format)
        embedding_obj = EmbeddingObject(
            embedding=embedding, index=0, object="embedding"
        )

        return CreateEmbeddingResponse(
            object="list",
            data=[embedding_obj],
            model=request.model,
            usage=usage,
        )

    @staticmethod
    def _get_embedding(
        embedding: List[float], encoding_format: Literal["float", "base64"]
    ) -> Union[list[float], str]:
        if encoding_format == "float":
            return embedding
        else:
            embedding_bytes = np.array(embedding, dtype="float32").tobytes()
            return base64.b64encode(embedding_bytes).decode("utf-8")

    # TODO: This behavior should be tested further
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    # TODO: Expose explicit flag to catch edge cases
    def _determine_request_converter(self, backend: str, request_type: RequestKind):
        # Allow manual override of backend request format if provided by user
        if self.backend:
            backend = self.backend

        # Request conversion from OpenAI format to backend-specific format
        if backend == "vllm":
            if request_type == RequestKind.GENERATION:
                return _create_vllm_generate_request
            else:
                return _create_vllm_embedding_request

        # Use TRT-LLM format as default for everything else. This could be
        # an ensemble, a python or BLS model, a TRT-LLM backend model, etc.
        if request_type == RequestKind.GENERATION:
            return _create_trtllm_generate_request
        else:
            return _create_trtllm_embedding_request

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

            lora_configs = _parse_lora_configs(
                self.server.options.model_repository,
                name,
                model.version,
                backend if self.backend is None else self.backend,
            )

            echo_tensor_name = None
            for input in model.config()["input"]:
                if input["name"] in [
                    "exclude_input_in_output",
                    "sampling_param_exclude_input_from_output",
                ]:
                    echo_tensor_name = input["name"]
                    break

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=self.tokenizer,
                lora_configs=lora_configs,
                echo_tensor_name=echo_tensor_name,
                create_time=self.create_time,
                inference_request_converter=self._determine_request_converter(
                    backend, RequestKind.GENERATION
                ),
                embedding_request_converter=self._determine_request_converter(
                    backend, RequestKind.EMBEDDING
                ),
            )
            model_metadata[name] = metadata

        return model_metadata

    def _get_streaming_chat_response_chunk(
        self,
        choice: ChatCompletionStreamingResponseChoice,
        request_id: str,
        created: int,
        model: str,
        usage: Optional[CompletionUsage] = None,
    ) -> CreateChatCompletionStreamResponse:
        return CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
            usage=usage,
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
            choice, request_id, created, model, usage=None
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
        include_usage = request.stream_options and request.stream_options.include_usage
        usage_accumulator = _StreamingUsageAccumulator(backend)

        chunk = self._get_first_streaming_chat_response(
            request_id, created, model, role
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        async for response in responses:
            delta_text = _get_output(response)
            if include_usage:
                usage_accumulator.update(response)

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

            # Parse logprobs for this chunk if requested
            chunk_logprobs = None
            if request.logprobs:
                openai_logprobs = _get_openai_chat_format_logprobs_from_vllm_response(
                    response
                )
                if openai_logprobs:
                    chunk_logprobs = ChatCompletionLogprobs(content=openai_logprobs)

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
                logprobs=chunk_logprobs,
                finish_reason=finish_reason,
            )

            chunk = self._get_streaming_chat_response_chunk(
                choice, request_id, created, model, usage=None
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        # Send the final usage chunk if requested via stream_options.
        if include_usage:
            usage_payload = usage_accumulator.get_final_usage()
            if usage_payload:
                final_usage_chunk = CreateChatCompletionStreamResponse(
                    id=request_id,
                    choices=[],
                    created=created,
                    model=model,
                    system_fingerprint=None,
                    object=ObjectType.chat_completion_chunk,
                    usage=usage_payload,
                )
                yield f"data: {final_usage_chunk.model_dump_json(exclude_unset=True)}\n\n"

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
            # only happens if we are NOT using structured outputs
            # or guided decoding
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
            raise ClientError(f"Unknown model: {request.model}")

        if not metadata.tokenizer:
            raise ServerError("Unknown tokenizer")

        if not metadata.backend:
            raise ServerError("Unknown backend")

        if not metadata.inference_request_converter:
            raise ServerError(
                f"Unknown inference request format for model: {request.model}"
            )

        if not metadata.embedding_request_converter:
            raise ServerError(
                f"Unknown embedding request format for model: {request.model}"
            )

        if (
            metadata.lora_configs is not None
            and lora_name is not None
            and lora_name
            not in [lora_config.name for lora_config in metadata.lora_configs]
        ):
            raise ClientError(f"Unknown LoRA: {lora_name}; for model: {request.model}")

        # Reject unsupported features if requested
        if request.n and request.n > 1:
            raise ClientError(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.logit_bias is not None:
            raise ClientError("logit bias is not currently supported")

        # Logprobs are only supported for vLLM backend currently
        if metadata.backend != "vllm" and (
            request.logprobs or request.top_logprobs is not None
        ):
            raise ClientError(
                "logprobs are currently available only for the vLLM backend"
            )

        if request.top_logprobs is not None and not request.logprobs:
            raise ClientError("`top_logprobs` can only be used when `logprobs` is True")

        self._verify_chat_tool_call_settings(request=request)

        if request.stream_options and not request.stream:
            raise ClientError("`stream_options` can only be used when `stream` is True")

    def _verify_chat_tool_call_settings(self, request: CreateChatCompletionRequest):
        if (
            request.tool_choice
            and request.tool_choice.root == ChatCompletionToolChoiceOption1.required
            and not request.tools
        ):
            raise ClientError(
                '"required" tool choice requires CreateChatCompletionRequest.tools to be provided'
            )

        if (
            request.tool_choice
            and isinstance(request.tool_choice.root, ChatCompletionNamedToolChoice)
            and not request.tools
        ):
            raise ClientError(
                "Named tool choice requires CreateChatCompletionRequest.tools to be provided"
            )

        if (
            request.tool_choice
            and request.tool_choice.root == ChatCompletionToolChoiceOption1.auto
            and self.tool_call_parser is None
        ):
            raise ClientError(
                '"auto" tool choice requires --tool-call-parser to be set'
            )

        if (
            request.tool_choice is None
            and request.tools
            and self.tool_call_parser is None
        ):
            raise ClientError(
                "having tools in the request requires --tool-call-parser to be set"
            )

    async def _streaming_completion_iterator(
        self,
        request_id: str,
        created: int,
        request: CreateCompletionRequest,
        responses: AsyncIterable,
        backend: str,
    ) -> AsyncIterator[str]:
        model = request.model
        include_usage = request.stream_options and request.stream_options.include_usage
        usage_accumulator = _StreamingUsageAccumulator(backend)
        current_offset = 0

        async for response in responses:
            if include_usage:
                usage_accumulator.update(response)

            text = _get_output(response)

            # Parse logprobs for this chunk if requested
            chunk_logprobs = None
            if request.logprobs is not None and request.logprobs > 0:
                chunk_logprobs = (
                    _get_openai_completion_format_logprobs_from_vllm_response(response)
                )
                # Adjust text offsets based on accumulated output
                if chunk_logprobs and chunk_logprobs.text_offset:
                    chunk_logprobs.text_offset = [
                        offset + current_offset for offset in chunk_logprobs.text_offset
                    ]

            current_offset += len(text)

            choice = Choice(
                finish_reason=FinishReason.stop if response.final else None,
                index=0,
                logprobs=chunk_logprobs,
                text=text,
            )
            chunk = CreateCompletionResponse(
                id=request_id,
                choices=[choice],
                system_fingerprint=None,
                object=ObjectType.text_completion,
                created=created,
                model=model,
                usage=None,
            )

            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        # Send the final usage chunk if requested via stream_options.
        if include_usage:
            usage_payload = usage_accumulator.get_final_usage()
            if usage_payload:
                final_usage_chunk = CreateCompletionResponse(
                    id=request_id,
                    choices=[],
                    system_fingerprint=None,
                    object=ObjectType.text_completion,
                    created=created,
                    model=model,
                    usage=usage_payload,
                )
                yield f"data: {final_usage_chunk.model_dump_json(exclude_unset=True)}\n\n"

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
            raise ClientError(f"Unknown model: {request.model}")

        if not metadata.backend:
            raise ServerError("Unknown backend")

        if not metadata.inference_request_converter:
            raise ServerError(
                f"Unknown inference request format for model: {request.model}"
            )

        if not metadata.embedding_request_converter:
            raise ServerError(
                f"Unknown embedding request format for model: {request.model}"
            )

        if (
            metadata.lora_configs is not None
            and lora_name is not None
            and lora_name
            not in [lora_config.name for lora_config in metadata.lora_configs]
        ):
            raise ClientError(f"Unknown LoRA: {lora_name}; for model: {request.model}")

        # Reject unsupported features if requested
        if request.suffix is not None:
            raise ClientError("suffix is not currently supported")

        if not request.prompt:
            raise ClientError("prompt must be non-empty")

        # Currently only support single string as input
        if not isinstance(request.prompt, str):
            raise ClientError("only single string input is supported")

        if "best_of" in request.model_fields_set and metadata.backend == "vllm":
            raise ClientError(
                "best_of is no longer supported in vLLM backend, removed from vLLM V1 engine"
            )

        if request.n and request.n > 1:
            raise ClientError(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.best_of and request.best_of > 1:
            raise ClientError(
                f"Received best_of={request.best_of}, but only single choice (best_of=1) is currently supported"
            )

        if request.logit_bias is not None:
            raise ClientError("logit bias is not supported")

        # Logprobs are only supported for vLLM backend currently
        if (
            request.logprobs is not None
            and request.logprobs > 0
            and metadata.backend != "vllm"
        ):
            raise ClientError(
                "logprobs are currently available only for the vLLM backend"
            )

        if request.stream_options and not request.stream:
            raise ClientError("`stream_options` can only be used when `stream` is True")

    def _validate_embedding_request(
        self,
        request: CreateEmbeddingRequest,
        metadata: TritonModelMetadata,
    ):
        """
        Validates an embedding request to align with currently supported features.
        """

        # Reject missing internal information needed to do inference
        if not metadata:
            raise ClientError(f"Unknown model: {request.model}")

        if not metadata.backend:
            raise ServerError("Unknown backend")

        if not metadata.inference_request_converter:
            raise ServerError(
                f"Unknown inference request format for model: {request.model}"
            )

        if not metadata.embedding_request_converter:
            raise ServerError(
                f"Unknown embedding request format for model: {request.model}"
            )

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

    def _get_lora_config(
        self, model_name: str, lora_name: Optional[str]
    ) -> TritonLoraConfig:
        model_metadata = self.model_metadata.get(model_name)
        if lora_name is None or model_metadata.lora_configs is None:
            return None
        for lora_config in model_metadata.lora_configs:
            if lora_config.name == lora_name:
                return lora_config
        raise ClientError(f"Unknown LoRA: {lora_name}; for model: {model_name}")
