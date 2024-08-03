import argparse
import os

import uvicorn
from src.api_server import app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton OpenAI Compatible RESTful API server."
    )
    # Uvicorn
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )

    # Triton
    parser.add_argument(
        "--tritonserver-log-level",
        type=int,
        default=0,
        help="The tritonserver log verbosity level",
    )

    parser.add_argument(
        "--model-repository",
        type=str,
        default="/opt/tritonserver/models",
        help="model repository",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # NOTE: Think about other ways to pass triton args to fastapi app,
    # but use env vars for simplicity for now.
    os.environ["TRITON_MODEL_REPOSITORY"] = args.model_repository
    os.environ["TRITON_LOG_VERBOSE_LEVEL"] = args.tritonserver_log_level

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=5,
    )
