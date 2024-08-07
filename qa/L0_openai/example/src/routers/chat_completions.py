import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ObjectType,
)
from src.utils.triton import create_vllm_inference_request, get_output

router = APIRouter()


def streaming_chat_completion_response(request_id, created, model, role, responses):
    # first chunk
    choice = ChatCompletionStreamingResponseChoice(
        index=0,
        delta=ChatCompletionStreamResponseDelta(
            role=role, content=None, function_call=None
        ),
        logprobs=None,
        finish_reason=None,
    )
    chunk = CreateChatCompletionStreamResponse(
        id=request_id,
        choices=[choice],
        created=created,
        model=model,
        system_fingerprint=None,
        object=ObjectType.chat_completion_chunk,
    )
    yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    for response in responses:
        text = get_output(response)

        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            ),
            logprobs=None,
            finish_reason=ChatCompletionFinishReason.stop if response.final else None,
        )

        chunk = CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
        )

        yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    yield "data: [DONE]\n\n"


@router.post(
    "/v1/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
def create_chat_completion(
    request: CreateChatCompletionRequest,
    raw_request: Request,
) -> CreateChatCompletionResponse | StreamingResponse:
    """
    Creates a model response for the given chat conversation.
    """

    # TODO: Cleanup
    model_metadatas = raw_request.app.models
    if not model_metadatas:
        raise HTTPException(status_code=400, detail="No known models")

    metadata = model_metadatas.get(request.model)
    if not metadata:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    # TODO: python models? default tokenizer? no tokenization OK?
    if not metadata.tokenizer:
        raise HTTPException(status_code=400, detail="No known tokenizer")

    if not metadata.backend:
        raise HTTPException(status_code=400, detail="No known backend")

    add_generation_prompt_default = True
    default_role = "assistant"

    model = raw_request.app.server.model(request.model)
    if request.model != model.name and request.model != metadata.source_name:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    if request.n and request.n > 1:
        raise HTTPException(status_code=400, detail=f"Only single choice is supported")

    conversation = [
        {"role": str(message.role), "content": str(message.content)}
        for message in request.messages
    ]

    # TODO: Use HF tokenizer or use Jinja/templater directly?
    # TODO: Function Calling / tools related to this?
    prompt = metadata.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt_default,
    )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())

    # TODO: Associate request function / backend with model metadata
    # responses = model.infer(create_inference_request(model, prompt, request))
    print(f"[DEBUG] {model=}")
    print(f"[DEBUG] {metadata=}")
    responses = model.infer(create_vllm_inference_request(model, prompt, request))

    if request.stream:
        return StreamingResponse(
            streaming_chat_completion_response(
                request_id, created, request.model, conversation[-1]["role"], responses
            )
        )

    response = list(responses)[0]

    text = get_output(response)

    return CreateChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=text, role=default_role, function_call=None
                ),
                logprobs=None,
                finish_reason=ChatCompletionFinishReason.stop,
            )
        ],
        created=created,
        model=request.model,
        system_fingerprint=None,
        object=ObjectType.chat_completion,
    )
