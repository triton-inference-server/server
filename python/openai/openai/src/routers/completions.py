import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.schemas.openai import (
    Choice,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    ObjectType,
)
from src.utils.triton import get_output, validate_triton_responses

router = APIRouter()


def streaming_completion_response(request_id, created, model, responses):
    for response in responses:
        text = get_output(response)

        choice = Choice(
            finish_reason=FinishReason.stop if response.final else None,
            index=0,
            logprobs=None,
            text=text,
        )
        response = CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=model,
        )

        yield f"data: {response.json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/v1/completions", response_model=CreateCompletionResponse, tags=["Completions"]
)
def create_completion(
    request: CreateCompletionRequest, raw_request: Request
) -> CreateCompletionResponse | StreamingResponse:
    """
    Creates a completion for the provided prompt and parameters.
    """

    if not request.model:
        raise Exception("Request must provide a valid 'model'")

    print(f"[DEBUG] Available model metadata: {raw_request.app.models.keys()=}")
    print(f"[DEBUG] Fetching model metadata for {request.model=}")
    metadata = raw_request.app.models.get(request.model)

    if not metadata:
        raise HTTPException(
            status_code=400, detail=f"Unknown model metadata for model: {request.model}"
        )

    if not metadata.request_convert_fn:
        raise HTTPException(
            status_code=400, detail=f"Unknown request format for model: {request.model}"
        )

    if request.suffix is not None:
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    if request.model != metadata.name:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    # Currently only support single string as input
    if not isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400, detail="only single string input is supported"
        )

    if request.n and request.n > 1:
        raise HTTPException(status_code=400, detail="Only single choice is supported")

    if request.logit_bias is not None or request.logprobs is not None:
        raise HTTPException(
            status_code=400, detail="logit bias and log probs not supported"
        )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())

    triton_model = raw_request.app.server.model(request.model)
    responses = triton_model.infer(
        metadata.request_convert_fn(triton_model, request.prompt, request)
    )
    if request.stream:
        return StreamingResponse(
            streaming_completion_response(
                request_id, created, metadata.name, responses
            ),
            media_type="text/event-stream",
        )

    # Response validation with decoupled models in mind
    responses = list(responses)
    validate_triton_responses(responses)
    response = responses[0]
    text = get_output(response)

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
        model=metadata.name,
    )
