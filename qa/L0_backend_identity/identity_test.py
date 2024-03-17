#!/usr/bin/python

# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from builtins import range

import numpy as np
import requests as httpreq
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Inference server URL."
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="http",
        help='Protocol ("http"/"grpc") used to '
        + 'communicate with inference service. Default is "http".',
    )

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            'unexpected protocol "{}", expects "http" or "grpc"'.format(FLAGS.protocol)
        )
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Run async requests to make sure backend handles request batches
    # correctly. We use just HTTP for this since we are not testing the
    # protocol anyway.
    if FLAGS.protocol == "http":
        model_name = "identity_uint32"
        request_parallelism = 4
        shape = [2, 2]
        with client_util.InferenceServerClient(
            FLAGS.url, concurrency=request_parallelism, verbose=FLAGS.verbose
        ) as client:
            input_datas = []
            requests = []
            for i in range(request_parallelism):
                input_data = (16384 * np.random.randn(*shape)).astype(np.uint32)
                input_datas.append(input_data)
                inputs = [
                    client_util.InferInput(
                        "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                requests.append(client.async_infer(model_name, inputs))

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                results = requests[i].get_result()
                print(results)

                output_data = results.as_numpy("OUTPUT0")
                if output_data is None:
                    print("error: expected 'OUTPUT0'")
                    sys.exit(1)

                if not np.array_equal(output_data, input_datas[i]):
                    print(
                        "error: expected output {} to match input {}".format(
                            output_data, input_datas[i]
                        )
                    )
                    sys.exit(1)

            # Make sure the requests ran in parallel.
            stats = client.get_inference_statistics(model_name)
            if (len(stats["model_stats"]) != 1) or (
                stats["model_stats"][0]["name"] != model_name
            ):
                print("error: expected statistics for {}".format(model_name))
                sys.exit(1)

            stat = stats["model_stats"][0]
            if (stat["inference_count"] != 8) or (stat["execution_count"] != 1):
                print(
                    "error: expected execution_count == 1 and inference_count == 8, got {} and {}".format(
                        stat["execution_count"], stat["inference_count"]
                    )
                )
                sys.exit(1)

            # Check metrics to make sure they are reported correctly
            metrics = httpreq.get("http://localhost:8002/metrics")
            print(metrics.text)

            success_str = (
                'nv_inference_request_success{model="identity_uint32",version="1"}'
            )
            infer_count_str = 'nv_inference_count{model="identity_uint32",version="1"}'
            infer_exec_str = (
                'nv_inference_exec_count{model="identity_uint32",version="1"}'
            )
            custom_metric_str = (
                'input_byte_size_counter{model="identity_uint32",version="1"}'
            )

            success_val = None
            infer_count_val = None
            infer_exec_val = None
            custom_metric_val = None
            for line in metrics.text.splitlines():
                if line.startswith(success_str):
                    success_val = float(line[len(success_str) :])
                if line.startswith(infer_count_str):
                    infer_count_val = float(line[len(infer_count_str) :])
                if line.startswith(infer_exec_str):
                    infer_exec_val = float(line[len(infer_exec_str) :])
                if line.startswith(custom_metric_str):
                    custom_metric_val = float(line[len(custom_metric_str) :])

            if success_val != 4:
                print(
                    "error: expected metric {} == 4, got {}".format(
                        success_str, success_val
                    )
                )
                sys.exit(1)
            if infer_count_val != 8:
                print(
                    "error: expected metric {} == 8, got {}".format(
                        infer_count_str, infer_count_val
                    )
                )
                sys.exit(1)
            if infer_exec_val != 1:
                print(
                    "error: expected metric {} == 1, got {}".format(
                        infer_exec_str, infer_exec_val
                    )
                )
                sys.exit(1)
            if custom_metric_val != 64:
                print(
                    "error: expected metric {} == 64, got {}".format(
                        custom_metric_str, custom_metric_val
                    )
                )
                sys.exit(1)

    # Reuse a single client for all sync tests
    with client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose) as client:
        for model_name, np_dtype, shape in (
            # yapf: disable
            ("identity_fp32", np.float32, [1, 0]),
            ("identity_fp32", np.float32, [1, 5]),
            ("identity_uint32", np.uint32, [4, 0]),
            ("identity_uint32", np.uint32, [8, 5]),
            ("identity_nobatch_int8", np.int8, [0]),
            ("identity_nobatch_int8", np.int8, [7]),
            ("identity_bytes", object, [1, 1]),
            ("identity_bf16", np.float32, [1, 0]),
            ("identity_bf16", np.float32, [1, 5])
        ):
            # yapf: enable
            if np_dtype != object:
                input_data = (16384 * np.random.randn(*shape)).astype(np_dtype)
            else:
                in0 = 16384 * np.ones(shape, dtype="int")
                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                input_data = in0n.reshape(in0.shape)
            if model_name != "identity_bf16":
                triton_type = np_to_triton_dtype(input_data.dtype)
            else:
                triton_type = "BF16"
            inputs = [client_util.InferInput("INPUT0", input_data.shape, triton_type)]
            inputs[0].set_data_from_numpy(input_data)

            results = client.infer(model_name, inputs)
            print(results)

            # Make sure outputs are expected value
            output_data = results.as_numpy("OUTPUT0")

            if np_dtype == object:
                output_data = np.array(
                    [str(x, encoding="utf-8") for x in output_data.flatten()],
                    dtype=object,
                ).reshape(output_data.shape)

            if output_data is None:
                print("error: expected 'OUTPUT0'")
                sys.exit(1)

            if model_name == "identity_bf16":
                if input_data.shape != output_data.shape:
                    print(
                        "error: expected output shape {} to match input shape {}".format(
                            output_data.shape, input_data.shape
                        )
                    )
                    sys.exit(1)
                for input, output in zip(
                    np.nditer(input_data, flags=["refs_ok", "zerosize_ok"], order="C"),
                    np.nditer(output_data, flags=["refs_ok", "zerosize_ok"], order="C"),
                ):
                    if input.tobytes()[2:4] != output.tobytes()[2:4]:
                        print(
                            "error: expected low-order bits of output {} to match low-order bits of input {}".format(
                                output, input
                            )
                        )
                        sys.exit(1)
                    if output.tobytes()[0:2] != b"\x00\x00":
                        print(
                            "error: expected output {} to have all-zero high-order bits, got {}".format(
                                output, output.tobytes()[0:2]
                            )
                        )
                        sys.exit(1)
            else:
                if not np.array_equal(output_data, input_data):
                    print(
                        "error: expected output {} to match input {}".format(
                            output_data, input_data
                        )
                    )
                    sys.exit(1)

            # Make sure response parameters are correct
            response = results.get_response()
            if FLAGS.protocol == "http":
                params = response["parameters"]
                param0 = params["param0"]
                param1 = params["param1"]
                param2 = params["param2"]
                param3 = params["param3"]
            else:
                params = response.parameters
                param0 = params["param0"].string_param
                param1 = params["param1"].int64_param
                param2 = params["param2"].bool_param
                param3 = params["param3"].double_param

            if param0 != "an example string parameter":
                print("error: expected 'param0' == 'an example string parameter'")
                sys.exit(1)
            if param1 != 42:
                print("error: expected 'param1' == 42")
                sys.exit(1)
            if param2 != False:
                print("error: expected 'param2' == False")
                sys.exit(1)
            if param3 != 123.123:
                print("error: expected 'param3' == 123.123")
                sys.exit(1)
