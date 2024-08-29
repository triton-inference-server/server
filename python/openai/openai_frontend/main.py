#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import signal
from functools import partial

import tritonserver
from engine.triton_engine import TritonLLMEngine
from frontend.fastapi_frontend import FastApiFrontend


def signal_handler(server, frontend, signal, frame):
    print(f"Received {signal=}, {frame=}")

    # Graceful Shutdown
    print("Shutting down OpenAI Frontend...")
    frontend.stop()
    print("Shutting down Triton Inference Server...")
    server.stop()


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
        "--tritonserver-log-verbose-level",
        type=int,
        default=0,
        help="The tritonserver log verbosity level",
    )
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
        help="HuggingFace ID of the Tokenizer to use for chat templates",
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
    engine: TritonLLMEngine = TritonLLMEngine(server=server, tokenizer=args.tokenizer)

    # Attach TritonLLMEngine as the backbone for inference and model management
    frontend: FastApiFrontend = FastApiFrontend(
        engine=engine, host=args.host, port=args.port, log_level=args.uvicorn_log_level
    )

    # Gracefully shutdown when receiving signals for testing and interactive use
    signal.signal(signal.SIGINT, partial(signal_handler, server, frontend))
    signal.signal(signal.SIGTERM, partial(signal_handler, server, frontend))

    # Blocking call until killed or interrupted with SIGINT
    frontend.start()


if __name__ == "__main__":
    main()
