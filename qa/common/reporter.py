#!/usr/bin/python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
import csv
import json
import os
import socket
from itertools import pairwise

import numpy as np
import requests

FLAGS = None

ENVS = [
    "CUDA_DRIVER_VERSION",
    "CUDA_VERSION",
    "TRITON_SERVER_VERSION",
    "NVIDIA_TRITON_SERVER_VERSION",
    "TRT_VERSION",
    "CUDNN_VERSION",
    "CUBLAS_VERSION",
    "BENCHMARK_PIPELINE",
    "BENCHMARK_REPO_BRANCH",
    "BENCHMARK_REPO_COMMIT",
    "BENCHMARK_CLUSTER",
    "BENCHMARK_GPU_COUNT",
]


def collect_gpu_metrics(data):
    import pynvml

    pynvml.nvmlInit()
    unique_gpu_models = set()
    total_memory = 0
    total_free_memory = 0

    # Get the number of available GPUs
    device_count = pynvml.nvmlDeviceGetCount()

    # Iterate through each GPU
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        # Get GPU name
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        unique_gpu_models.add(gpu_name)

        # Get GPU memory information
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory += memory_info.total
        total_free_memory += memory_info.free

    data["l_gpus_count"] = device_count
    data["s_gpu_model"] = ", ".join(unique_gpu_models)
    data["d_total_gpu_memory_mb"] = total_memory / (1024**2)
    data["d_total_free_gpu_memory_mb"] = total_free_memory / (1024**2)

    pynvml.nvmlShutdown()


def collect_token_latencies(export_data, data):
    first_token_latencies = []
    token_to_token_latencies = []
    requests = export_data["experiments"][0]["requests"]

    for r in requests:
        init_request, responses = r["timestamp"], r["response_timestamps"]
        first_token_latency = (responses[0] - init_request) / 1_000_000
        first_token_latencies.append(first_token_latency)
        for prev_res, res in pairwise(responses):
            token_to_token_latencies.append((res - prev_res) / 1_000_000)

    data["d_avg_token_to_token_latency_ms"] = np.mean(token_to_token_latencies)  # msec
    data["d_avg_first_token_latency_ms"] = np.mean(first_token_latencies)  # msec


def annotate(data):
    # Add all interesting envvar values
    for data in data:
        for env in ENVS:
            if env in os.environ:
                val = os.environ[env]
                data["s_" + env.lower()] = val

        # Add this system's name. If running within slurm use
        # SLURM_JOB_NODELIST as the name (this assumes that the slurm
        # job was scheduled on a single node, otherwise
        # SLURM_JOB_NODELIST will list multiple nodes).
        if "SLURM_JOB_NODELIST" in os.environ:
            data["s_benchmark_system"] = os.environ["SLURM_JOB_NODELIST"]
        else:
            data["s_benchmark_system"] = socket.gethostname()


def annotate_csv(data, csv_file):
    csv_reader = csv.reader(csv_file, delimiter=",")
    linenum = 0
    header_row = None
    concurrency_row = None
    for row in csv_reader:
        if linenum == 0:
            header_row = row
        else:
            concurrency_row = row
            break
        linenum += 1

    if (header_row is not None) and (concurrency_row is not None):
        avg_latency_us = 0
        for header, result in zip(header_row, concurrency_row):
            if header == "Inferences/Second":
                data["d_infer_per_sec"] = float(result)
            elif (
                (header == "Client Send")
                or (header == "Network+Server Send/Recv")
                or (header == "Server Queue")
                or (header == "Server Compute Input")
                or (header == "Server Compute Output")
                or (header == "Server Compute Infer")
                or (header == "Client Recv")
            ):
                avg_latency_us += float(result)
            elif header == "p50 latency":
                data["d_latency_p50_ms"] = float(result) / 1000.0
            elif header == "p90 latency":
                data["d_latency_p90_ms"] = float(result) / 1000.0
            elif header == "p95 latency":
                data["d_latency_p95_ms"] = float(result) / 1000.0
            elif header == "p99 latency":
                data["d_latency_p99_ms"] = float(result) / 1000.0

        data["d_latency_avg_ms"] = avg_latency_us / 1000.0


def post_to_url(url, data):
    headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
    r = requests.post(url, data=data, headers=headers)
    r.raise_for_status()


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
        "--gpu-metrics",
        action="store_true",
        required=False,
        default=False,
        help="Collect GPU details",
    )
    parser.add_argument(
        "-e",
        "--profile-export-file",
        type=argparse.FileType("r"),
        required=False,
        help="Profile file exported by perf_analyzer",
    )
    parser.add_argument(
        "--token-latency",
        action="store_true",
        required=False,
        default=False,
        help="Collect token latency data",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=False, help="Output filename"
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Post results to a URL"
    )
    parser.add_argument(
        "--csv",
        type=argparse.FileType("r"),
        required=False,
        help="perf_analyzer generated CSV",
    )
    parser.add_argument("file", type=argparse.FileType("r"))
    FLAGS = parser.parse_args()

    data = json.loads(FLAGS.file.read())

    if FLAGS.verbose:
        print("*** Load json ***")
        print(json.dumps(data, sort_keys=True, indent=2))

    if FLAGS.gpu_metrics:
        collect_gpu_metrics(data[0])

    if FLAGS.token_latency:
        if not FLAGS.profile_export_file:
            raise Exception(
                "Please provide a profile export file to collect token latencies."
            )
        export_data = json.loads(FLAGS.profile_export_file.read())
        collect_token_latencies(export_data, data[0])

    if FLAGS.csv is not None:
        if len(data) != 1:
            raise Exception("--csv requires that json data have a single array entry")
        annotate_csv(data[0], FLAGS.csv)
        if FLAGS.verbose:
            print("*** Annotate CSV ***")
            print(json.dumps(data, sort_keys=True, indent=2))

    annotate(data)

    if FLAGS.verbose:
        print("*** Post Annotate ***")
        print(json.dumps(data, sort_keys=True, indent=2))

    if FLAGS.output is not None:
        with open(FLAGS.output, "w") as f:
            f.write(json.dumps(data))
            f.write("\n")

    if FLAGS.url is not None:
        post_to_url(FLAGS.url, json.dumps(data))
