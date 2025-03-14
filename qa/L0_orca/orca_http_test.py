#!/usr/bin/python3
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import sys

import requests


# To run the test, have tritonserver running and run this script with the endpoint as a flag.
#
# Example:
# ```
# python3 orca_header_test.py http://localhost:8000/v2/models/ensemble/generate
# ```
def get_endpoint_header(url, data, request_header=None):
    """
    Sends a POST request to the given URL with the provided data and returns the value of the "endpoint-load-metrics" header,
    or None if the request fails.
    """
    HEADER_KEY = "endpoint-load-metrics"
    try:
        response = None
        if request_header:
            response = requests.post(url, json=data, headers=request_header)
        else:
            response = requests.post(url, json=data)
        response.raise_for_status()
        return response.headers.get(HEADER_KEY, "")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


def parse_header_data(header, orca_format):
    """
    Parses the header data into a dictionary based on the given format.
    """
    METRIC_KEY = "named_metrics"
    try:
        if orca_format == "json":
            # Parse the header in JSON format
            data = json.loads(header.replace("JSON ", ""))
            if METRIC_KEY in data:
                return data[METRIC_KEY]
            else:
                print(f"No key '{METRIC_KEY}' in header data: {data}")
                return None
        elif orca_format == "text":
            # Parse the header in TEXT format
            data = {}
            for key_value_pair in header.replace("TEXT ", "").split(", "):
                key, value = key_value_pair.split("=")
                if "." in key:
                    prefix, nested_key = key.split(".", 1)
                    if prefix == METRIC_KEY:
                        data[nested_key] = float(value)
            if not data:
                print(f"Could not parse any keys from header: {header}")
                return None
            return data
        else:
            print(f"Invalid ORCA format: {orca_format}")
            return None
    except (json.JSONDecodeError, ValueError, KeyError):
        print("Error: Invalid data in the header.")
        return None


def check_for_keys(data, desired_keys, orca_format):
    """
    Checks if all desired keys are present in the given data dictionary.
    """
    if all(key in data for key in desired_keys):
        print(
            f"ORCA header present in {orca_format} format with kv_cache_utilization: {[f'{k}: {data[k]}' for k in desired_keys]}"
        )
        return True
    else:
        print(f"Missing keys in header: {', '.join(set(desired_keys) - set(data))}")
        return False


def request_header(orca_format):
    return {"endpoint-load-metrics-format": orca_format} if orca_format else None


def test_header_type(url, data, orca_format):
    req_header = request_header(orca_format)
    response_header = get_endpoint_header(args.url, TEST_DATA, req_header)

    desired_keys = {
        "kv_cache_utilization",
        "max_token_capacity",
    }  # Just the keys, no need to initialize with None

    if response_header is None:
        print(f"Request to endpoint: '{args.url}' failed.")
        return False
    elif response_header == "":
        if orca_format:
            print(
                f"response header empty, endpoint-load-metrics-format={orca_format} is not a valid ORCA metric format"
            )
            return False
        else:
            # No request header set <=> no response header. Intended behavior.
            print(f"response header empty, endpoint-load-metrics-format is not set")
            return True

    data = parse_header_data(response_header, orca_format)
    if data:
        return check_for_keys(data, desired_keys, orca_format)
    else:
        print(f"Unexpected response header value: {response_header}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a POST request to generate endpoint to test the ORCA metrics header."
    )
    parser.add_argument("url", help="The model URL to send the request to.")
    args = parser.parse_args()
    TEST_DATA = json.loads(
        '{"text_input": "hello world", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
    )
    passed = True

    for format in ["json", "text", None]:
        print("Checking response header for ORCA format:", format)
        if not test_header_type(args.url, TEST_DATA, format):
            print("FAIL on format:", format)
            passed = False

    sys.exit(0 if passed else 1)
