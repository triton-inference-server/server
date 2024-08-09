import argparse
import os

import uvicorn
from src.api_server import app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton OpenAI Compatible RESTful API server."
    )
    # Uvicorn
    uvicorn_group = parser.add_argument_group("Uvicorn")
    uvicorn_group.add_argument("--host", type=str, default=None, help="host name")
    uvicorn_group.add_argument("--port", type=int, default=8000, help="port number")
    uvicorn_group.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )

    # Triton
    triton_group = parser.add_argument_group("Triton Inference Server")
    triton_group.add_argument(
        "--tritonserver-log-level",
        type=int,
        default=0,
        help="The tritonserver log verbosity level",
    )

    triton_group.add_argument(
        "--model-repository",
        type=str,
        default=None,
        help="model repository",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # NOTE: Think about other ways to pass triton args to fastapi app,
    # but use env vars for simplicity for now.
    if args.model_repository:
        os.environ["TRITON_MODEL_REPOSITORY"] = args.model_repository
    os.environ["TRITON_LOG_VERBOSE_LEVEL"] = str(args.tritonserver_log_level)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=5,
    )
