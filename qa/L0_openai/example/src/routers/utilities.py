from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/metrics", tags=["Utilities"])
def metrics(request: Request) -> str:
    if not request.app.server or not request.app.server.live():
        raise HTTPException(
            status_code=400, detail="Triton Inference Server is not live."
        )

    return request.app.server.metrics()


@router.get("/health", tags=["Utilities"])
def health(request: Request) -> Response:
    if not request.app.server or not request.app.server.live():
        raise HTTPException(
            status_code=400, detail="Triton Inference Server is not live."
        )

    return Response(status_code=200)
