from fastapi import APIRouter, HTTPException, Request
from src.schemas.openai import ListModelsResponse, Model, ObjectType

router = APIRouter()

# TODO: What is this for?
OWNED_BY = "ACME"


@router.get("/v1/models", response_model=ListModelsResponse, tags=["Models"])
def list_models(request: Request) -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """
    models = request.app.models
    if not models:
        raise HTTPException(status_code=400, detail="No known models")

    model_list = []
    for model in models:
        metadata = models[model]
        if not metadata:
            raise HTTPException(
                status_code=400, detail=f"No metadata for model: {model}"
            )

        model_list.append(
            Model(
                id=metadata.name,
                created=metadata.create_time,
                object=ObjectType.model,
                owned_by=OWNED_BY,
            ),
        )

    return ListModelsResponse(object=ObjectType.list, data=model_list)


@router.get("/v1/models/{model_name}", response_model=Model, tags=["Models"])
def retrieve_model(request: Request, model_name: str) -> Model:
    """
    Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    """
    models = request.app.models
    if not models:
        raise HTTPException(status_code=400, detail="No known models")

    model = models.get(model_name)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    # TODO: Do we want to accept both triton name or source name interchangeably?
    if model_name == model.name:
        return Model(
            id=model.name,
            created=model.create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        )

    if model_name == model.source_name:
        return Model(
            id=model.source_name,
            created=model.create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        )

    raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
