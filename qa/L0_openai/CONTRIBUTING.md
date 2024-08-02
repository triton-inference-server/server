# Triton Inference Server OpenAI Example

## Development

For simplicity, a `Dockerfile` containing the necessary
dependencies is included, which can be modified and built
for your needs.

```
docker build -t fastapi_triton .
# TODO: minimal args
docker run ... fastapi_triton
# TODO: cd to location
fastapi dev
```

## Testing

The testing for this example is all done through `pytest`, which
is well integrated with `FastAPI`.

```
cd src/tests
pytest
```

## Adding New Routes

First define your own router in `src/routers`, referring
to the existing routers as examples.

Then, add your router to the application in `api_server.py`
with `app.include_router(my_router)`.

