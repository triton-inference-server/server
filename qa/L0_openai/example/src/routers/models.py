from fastapi import APIRouter
from src.schemas.openai import ListModelsResponse, Model, ObjectType

router = APIRouter()

# TODO: What is this for?
OWNED_BY = "ACME"


@router.get("/v1/models", response_model=ListModelsResponse, tags=["Models"])
def list_models() -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """

    model_list = [
        Model(
            id=model.name,
            created=model_create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        ),
        Model(
            id=model_source_name,
            created=model_create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        ),
    ]

    return ListModelsResponse(object=ObjectType.list, data=model_list)


@router.get("/v1/models/{model_name}", response_model=Model, tags=["Models"])
def retrieve_model(model_name: str) -> Model:
    """
    Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    """

    if model_name == model.name:
        return Model(
            id=model.name,
            created=model_create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        )

    if model_name == model_source_name:
        return Model(
            id=model_source_name,
            created=model_create_time,
            object=ObjectType.model,
            owned_by=OWNED_BY,
        )

    raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
