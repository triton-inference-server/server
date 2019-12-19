#!/usr/bin/python
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

FLAGS = None

def verify_timestamps(traces, preserve):
    # Order traces by id
    traces = sorted(traces, key=lambda t: t.get('id', -1))

    # Filter the trace that is not meaningful
    filtered_traces = []
    for trace in traces:
        if "id" not in trace:
            continue
        filtered_traces.append(trace)

    # First find the latest send end timestamp for the batch with large delay
    large_delay_send_end = 0
    small_delay_traces = []
    for trace in filtered_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        # Hardcoded delay value here (knowing large delay is 400ms)
        compute_span = timestamps["compute end"] - timestamps["compute start"]
        # If the 3rd batch is also processed by large delay instance, we don't
        # want to use its responses as baseline
        if trace["id"] <= 7 and compute_span >= 400 * 1000 * 1000:
            if "grpc send end" in timestamps:
                send_end = timestamps["grpc send end"]
            else:
                send_end = timestamps["http send end"]
            large_delay_send_end = max(large_delay_send_end, send_end)
        else:
            small_delay_traces.append(trace)

    response_send_after_large_delay_count = 0
    for trace in small_delay_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        if "grpc send start" in timestamps:
            send_start = timestamps["grpc send start"]
        else:
            send_start = timestamps["http send start"]
        if send_start > large_delay_send_end:
            response_send_after_large_delay_count += 1
    
    # Hardcoded expected count here
    if preserve:
        # If preserve ordering, there must be large delay batch followed by
        # small delay batch and thus at least 4 responses are sent after
        return 0 if response_send_after_large_delay_count >= 4 else 1
    else:
        # If not preserve ordering, the small delay batches should all be done
        # before large delay batch regardless of the ordering in scheduler
        return 0 if response_send_after_large_delay_count == 0 else 1

    return 0

def summarize(protocol, traces):
    for trace in filtered_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]

        if ("request handler start" in timestamps) and ("request handler end" in timestamps):
            key = (trace["model_name"], trace["model_version"])
            if key not in model_count_map:
                model_count_map[key] = 0
                model_span_map[key] = dict()

            model_count_map[key] += 1
            if ("http recv start" in timestamps) and ("http send end" in timestamps):
                add_span(model_span_map[key], timestamps,
                         "http infer", "http recv start", "http send end")
                add_span(model_span_map[key], timestamps,
                         "http recv", "http recv start", "http recv end")
                add_span(model_span_map[key], timestamps,
                         "http send", "http send start", "http send end")
            elif ("grpc wait/read start" in timestamps) and ("grpc send end" in timestamps):
                add_span(model_span_map[key], timestamps,
                         "grpc infer", "grpc wait/read start", "grpc send end")
                add_span(model_span_map[key], timestamps,
                         "grpc wait/read", "grpc wait/read start", "grpc wait/read end")
                add_span(model_span_map[key], timestamps,
                         "grpc send", "grpc send start", "grpc send end")

            add_span(model_span_map[key], timestamps,
                     "request handler", "request handler start", "request handler end")
            
            # The tags below will be missing for ensemble model
            if ("queue start" in timestamps) and ("compute start" in timestamps):
                add_span(model_span_map[key], timestamps,
                        "queue", "queue start", "compute start")
            if ("compute start" in timestamps) and ("compute end" in timestamps):
                add_span(model_span_map[key], timestamps,
                        "compute", "compute start", "compute end")
            if ("compute input end" in timestamps) and ("compute output start" in timestamps):
                add_span(model_span_map[key], timestamps,
                         "compute input", "compute start", "compute input end")
                add_span(model_span_map[key], timestamps,
                         "compute infer", "compute input end", "compute output start")
                add_span(model_span_map[key], timestamps,
                         "compute output", "compute output start", "compute end")

            if FLAGS.show_trace:
                print("{} ({}):".format(trace["model_name"], trace["model_version"]))
                print("\tid: {}".format(trace["id"]))
                if "parent_id" in trace:
                    print("\tparent id: {}".format(trace["parent_id"]))
                ordered_timestamps = list()
                for ts in trace["timestamps"]:
                    ordered_timestamps.append((ts["name"], ts["ns"]))
                ordered_timestamps.sort(key=lambda tup: tup[1])

                now = None
                for ts in ordered_timestamps:
                    if now is not None:
                        print("\t\t{}us".format((ts[1] - now) / 1000))
                    print("\t{}".format(ts[0]))
                    now = ts[1]

    for key, cnt in model_count_map.items():
        model_name, model_value = key
        print("Summary for {} ({}): trace count = {}".format(model_name, model_value, cnt))

        if "http infer" in model_span_map[key]:
            print("HTTP infer request (avg): {}us".format(
                model_span_map[key]["http infer"] / (cnt * 1000)))
            print("\tReceive (avg): {}us".format(
                model_span_map[key]["http recv"] / (cnt * 1000)))
            print("\tSend (avg): {}us".format(
                model_span_map[key]["http send"] / (cnt * 1000)))
            print("\tOverhead (avg): {}us".format(
                (model_span_map[key]["http infer"] -
                 model_span_map[key]["request handler"] -
                 model_span_map[key]["http recv"] -
                 model_span_map[key]["http send"]) / (cnt * 1000)))
        elif "grpc infer" in model_span_map[key]:
            print("GRPC infer request (avg): {}us".format(
                model_span_map[key]["grpc infer"] / (cnt * 1000)))
            print("\tWait/Read (avg): {}us".format(
                model_span_map[key]["grpc wait/read"] / (cnt * 1000)))
            print("\tSend (avg): {}us".format(
                model_span_map[key]["grpc send"] / (cnt * 1000)))
            print("\tOverhead (avg): {}us".format(
                (model_span_map[key]["grpc infer"] -
                 model_span_map[key]["request handler"] -
                 model_span_map[key]["grpc wait/read"] -
                 model_span_map[key]["grpc send"]) / (cnt * 1000)))

        print("\tHandler (avg): {}us".format(
            model_span_map[key]["request handler"] / (cnt * 1000)))
        if ("queue" in model_span_map[key]) and "compute" in model_span_map[key]:
            print("\t\tOverhead (avg): {}us".format(
                (model_span_map[key]["request handler"] -
                model_span_map[key]["queue"] -
                model_span_map[key]["compute"]) / (cnt * 1000)))
            print("\t\tQueue (avg): {}us".format(
                model_span_map[key]["queue"] / (cnt * 1000)))
            print("\t\tCompute (avg): {}us".format(
                model_span_map[key]["compute"] / (cnt * 1000)))
        if ("compute input" in model_span_map[key]) and "compute output" in model_span_map[key]:
            print("\t\t\tInput (avg): {}us".format(
                model_span_map[key]["compute input"] / (cnt * 1000)))
            print("\t\t\tInfer (avg): {}us".format(
                model_span_map[key]["compute infer"] / (cnt * 1000)))
            print("\t\t\tOutput (avg): {}us".format(
                model_span_map[key]["compute output"] / (cnt * 1000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preserve', action="store_true", required=False, default=False,
                        help='Timestamps is collected with preserve ordering')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    FLAGS = parser.parse_args()

    for f in FLAGS.file:
        trace_data = json.loads(f.read())
        exit(verify_timestamps(trace_data, FLAGS.preserve))
