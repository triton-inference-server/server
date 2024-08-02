from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from src.schemas.openai import (
    Choice,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    ObjectType,
)
from src.utils.triton import get_output

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

    if not model or not tokenizer or not create_inference_request:
        raise Exception("Unknown Model")

    if request.suffix is not None:
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    if request.model != model.name and request.model != model_source_name:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    if request.prompt is None:
        request.prompt = "<|endoftext|>"

    # Currently only support single string as input
    if not isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400, detail="only single string input is supported"
        )

    if request.logit_bias is not None or request.logprobs is not None:
        raise HTTPException(
            status_code=400, detail="logit bias and log probs not supported"
        )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())

    responses = model.infer(create_inference_request(model, request.prompt, request))
    if request.stream:
        return StreamingResponse(
            streaming_completion_response(request_id, created, model.name, responses)
        )
    response = list(responses)[0]
    text = get_output(response)

    choice = Choice(
        finish_reason=FinishReason.stop if response.final else None,
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
        model=model.name,
    )
