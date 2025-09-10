#!/usr/bin/env python3

# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import signal
import sys
from functools import partial

import tritonserver
from engine.triton_engine import TritonLLMEngine
from frontend.fastapi_frontend import FastApiFrontend


def signal_handler(
    server, openai_frontend, kserve_http_frontend, kserve_grpc_frontend, signal, frame
):
    print(f"Received {signal=}, {frame=}")
    # Graceful Shutdown
    shutdown(server, openai_frontend, kserve_http_frontend, kserve_grpc_frontend)


def shutdown(server, openai_frontend, kserve_http, kserve_grpc):
    print("Shutting down Triton OpenAI-Compatible Frontend...")
    openai_frontend.stop()

    if kserve_http:
        print("Shutting down Triton KServe HTTP Frontend...")
        kserve_http.stop()

    if kserve_grpc:
        print("Shutting down Triton KServe GRPC Frontend...")
        kserve_grpc.stop()

    print("Shutting down Triton Inference Server...")
    server.stop()


def start_kserve_frontends(server, args):
    http_service, grpc_service = None, None
    try:
        from tritonfrontend import KServeGrpc, KServeHttp

        http_options = KServeHttp.Options(address=args.host, port=args.kserve_http_port)
        http_service = KServeHttp(server, http_options)
        http_service.start()

        grpc_options = KServeGrpc.Options(address=args.host, port=args.kserve_grpc_port)
        grpc_service = KServeGrpc(server, grpc_options)
        grpc_service.start()

    except ModuleNotFoundError:
        # FIXME: Raise error instead of warning if kserve frontends are opt-in
        print(
            "[WARNING] The 'tritonfrontend' package was not found. "
            "KServe frontends won't be available through this application without it. "
            "Check /opt/tritonserver/python for tritonfrontend*.whl and pip install it if present."
        )
    return http_service, grpc_service


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton Inference Server with OpenAI-Compatible RESTful API server."
    )

    # Triton Inference Server
    triton_group = parser.add_argument_group("Triton Inference Server")
    triton_group.add_argument(
        "--model-repository",
        type=str,
        required=True,
        help="Path to the Triton model repository holding the models to be served",
    )
    triton_group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace ID or local folder path of the Tokenizer to use for chat templates",
    )
    triton_group.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["vllm", "tensorrtllm"],
        help="Manual override of Triton backend request format (inputs/output names) to use for inference",
    )
    triton_group.add_argument(
        "--lora-separator",
        type=str,
        default=None,
        help="LoRA name selection may be appended to the model name following this separator if the separator is provided",
    )
    triton_group.add_argument(
        "--tritonserver-log-verbose-level",
        type=int,
        default=0,
        help="The tritonserver log verbosity level",
    )
    triton_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address/host of frontends (default: '0.0.0.0')",
    )
    triton_group.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        help="Specify the parser for handling tool calling related response text. Options include: 'llama3' and 'mistral'.",
    )
    # Allows the user to try a different chat template to craft better prompts and receive more targeted tool-calling responses from the model.
    # Some Mistral models have a separate chat template file, in addition to the tokenizer_config.json,
    # such as mistralai/Mistral-Small-3.1-24B-Instruct-2503.
    # This can serve as a workaround for those models.
    triton_group.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The path to the custom Jinja chat template file. This is useful if you'd like to use a different chat template than the one provided by the model.",
    )

    triton_group.add_argument(
        "--default-max-tokens",
        type=int,
        default=16,
        help="The default maximum number of tokens to generate if not specified in the request. The default is 16.",
    )

    # OpenAI-Compatible Frontend (FastAPI)
    openai_group = parser.add_argument_group("Triton OpenAI-Compatible Frontend")
    openai_group.add_argument(
        "--openai-port", type=int, default=9000, help="OpenAI HTTP port (default: 9000)"
    )
    openai_group.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )
    openai_group.add_argument(
        "--openai-restricted-api",
        type=str,
        default=None,
        nargs=3,
        metavar=("APIs", "Restricted Key", "Restricted Value"),
        action="append",
        help="Restrict access to specific OpenAI API endpoints. Format: '<API_1>,<API_2>,... <restricted-key> <restricted-value>' (e.g., 'inference,model-repository admin-key admin-value'). If not specified, all endpoints are allowed.",
    )

    # KServe Predict v2 Frontend
    kserve_group = parser.add_argument_group("Triton KServe Frontend")
    kserve_group.add_argument(
        "--enable-kserve-frontends",
        action="store_true",
        help="Enable KServe Predict v2 HTTP/GRPC frontends (disabled by default)",
    )
    kserve_group.add_argument(
        "--kserve-http-port",
        type=int,
        default=8000,
        help="KServe Predict v2 HTTP port (default: 8000)",
    )
    kserve_group.add_argument(
        "--kserve-grpc-port",
        type=int,
        default=8001,
        help="KServe Predict v2 GRPC port (default: 8001)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize a Triton Inference Server pointing at LLM models
    server: tritonserver.Server = tritonserver.Server(
        model_repository=args.model_repository,
        log_verbose=args.tritonserver_log_verbose_level,
        log_info=True,
        log_warn=True,
        log_error=True,
    ).start(wait_until_ready=True)

    # Wrap Triton Inference Server in an interface-conforming "LLMEngine"
    engine: TritonLLMEngine = TritonLLMEngine(
        server=server,
        tokenizer=args.tokenizer,
        backend=args.backend,
        lora_separator=args.lora_separator,
        tool_call_parser=args.tool_call_parser,
        chat_template=args.chat_template,
        default_max_tokens=args.default_max_tokens,
    )

    # Attach TritonLLMEngine as the backbone for inference and model management
    try:
        openai_frontend: FastApiFrontend = FastApiFrontend(
            engine=engine,
            host=args.host,
            port=args.openai_port,
            log_level=args.uvicorn_log_level,
            restricted_apis=args.openai_restricted_api,
        )
    except ValueError as e:
        print(
            f"[ERROR] Failed to initialize FastAPI frontend: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optionally expose Triton KServe HTTP/GRPC Frontends
    kserve_http, kserve_grpc = None, None
    if args.enable_kserve_frontends:
        kserve_http, kserve_grpc = start_kserve_frontends(server, args)

    # Gracefully shutdown when receiving signals for testing and interactive use
    signal.signal(
        signal.SIGINT,
        partial(signal_handler, server, openai_frontend, kserve_http, kserve_grpc),
    )
    signal.signal(
        signal.SIGTERM,
        partial(signal_handler, server, openai_frontend, kserve_http, kserve_grpc),
    )

    # Blocking call until killed or interrupted with SIGINT
    openai_frontend.start()


if __name__ == "__main__":
    main()
