import argparse

import uvicorn
from src.api_server import app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton OpenAI Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )
    parser.add_argument(
        "--response-role", type=str, default="assistant", help="The role name to return"
    )

    parser.add_argument(
        "--tritonserver-log-level",
        type=int,
        default=0,
        help="The tritonserver log level",
    )

    parser.add_argument(
        "--model-repository",
        type=str,
        default="/workspace/llm-models",
        help="model repository",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # TODO: Cleanup
    args = parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=5,
    )
