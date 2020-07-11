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
import json
import sys

FLAGS = None


def verify_timestamps(traces, preserve):
    # Order traces by id
    traces = sorted(traces, key=lambda t: t.get('id', -1))

    # Filter the trace that is not meaningful and group them by 'id'
    filtered_traces = dict()
    grpc_id_offset = 0
    for trace in traces:
        if "id" not in trace:
            continue
        # Skip GRPC traces as actual traces are not genarated via GRPC,
        # thus GRPC traces are ill-formed
        if "timestamps" in trace:
            is_grpc = False
            for ts in trace["timestamps"]:
                if "GRPC" in ts["name"]:
                    is_grpc = True
                    break
            if is_grpc:
                grpc_id_offset += 1
                continue

        if (trace['id'] in filtered_traces.keys()):
            rep_trace = filtered_traces[trace['id']]
            # Apend the timestamp to the trace representing this 'id'
            if "timestamps" in trace:
                rep_trace["timestamps"] += trace["timestamps"]
        else:
            # Use this trace to represent this 'id'
            if "timestamps" not in trace:
                trace["timestamps"] = []
            filtered_traces[trace['id']] = trace

    # First find the latest send end timestamp for the batch with large delay
    large_delay_send_end = 0
    small_delay_traces = []
    for trace_id, trace in filtered_traces.items():
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        # Hardcoded delay value here (knowing large delay is 400ms)
        compute_span = timestamps["COMPUTE_END"] - timestamps["COMPUTE_START"]
        # If the 3rd batch is also processed by large delay instance, we don't
        # want to use its responses as baseline
        if trace["id"] <= (
                8 + grpc_id_offset) and compute_span >= 400 * 1000 * 1000:
            send_end = timestamps["HTTP_SEND_END"]
            large_delay_send_end = max(large_delay_send_end, send_end)
        else:
            small_delay_traces.append(trace)

    response_request_after_large_delay_count = 0
    for trace in small_delay_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        send_end = timestamps["HTTP_SEND_END"]
        if send_end > large_delay_send_end:
            response_request_after_large_delay_count += 1

    # Hardcoded expected count here
    print(response_request_after_large_delay_count)
    if preserve:
        # If preserve ordering, there must be large delay batch followed by
        # small delay batch and thus at least 4 responses are sent after
        return 0 if response_request_after_large_delay_count >= 4 else 1
    else:
        # If not preserve ordering, the small delay batches should all be done
        # before large delay batch regardless of the ordering in scheduler
        return 0 if response_request_after_large_delay_count == 0 else 1

    return 0


def summarize(protocol, traces):
    for trace in filtered_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]

        if ("REQUEST_START" in timestamps) and ("REQUEST_END" in timestamps):
            key = (trace["model_name"], trace["model_version"])
            if key not in model_count_map:
                model_count_map[key] = 0
                model_span_map[key] = dict()

            model_count_map[key] += 1
            if ("HTTP_RECV_START" in timestamps) and ("HTTP_SEND_END"
                                                      in timestamps):
                add_span(model_span_map[key], timestamps, "http infer",
                         "HTTP_RECV_START", "HTTP_SEND_END")
                add_span(model_span_map[key], timestamps, "http recv",
                         "HTTP_RECV_START", "HTTP_RECV_END")
                add_span(model_span_map[key], timestamps, "http send",
                         "HTTP_SEND_START", "HTTP_SEND_END")
            elif ("GRPC_WAITREAD_START" in timestamps) and ("GRPC_SEND_END"
                                                            in timestamps):
                add_span(model_span_map[key], timestamps, "grpc infer",
                         "GRPC_WAITREAD_START", "GRPC_SEND_END")
                add_span(model_span_map[key], timestamps, "grpc wait/read",
                         "GRPC_WAITREAD_START", "GRPC_WAITREAD_END")
                add_span(model_span_map[key], timestamps, "grpc send",
                         "GRPC_SEND_START", "GRPC_SEND_END")

            add_span(model_span_map[key], timestamps, "request handler",
                     "REQUEST_START", "REQUEST_END")

            # The tags below will be missing for ensemble model
            if ("QUEUE_START" in timestamps) and ("COMPUTE_START"
                                                  in timestamps):
                add_span(model_span_map[key], timestamps, "queue",
                         "QUEUE_START", "COMPUTE_START")
            if ("COMPUTE_START" in timestamps) and ("COMPUTE_END"
                                                    in timestamps):
                add_span(model_span_map[key], timestamps, "compute",
                         "COMPUTE_START", "COMPUTE_END")
            if ("COMPUTE_INPUT_END" in timestamps) and ("COMPUTE_OUTPUT_START"
                                                        in timestamps):
                add_span(model_span_map[key], timestamps, "compute input",
                         "COMPUTE_START", "COMPUTE_INPUT_END")
                add_span(model_span_map[key], timestamps, "compute infer",
                         "COMPUTE_INPUT_END", "COMPUTE_OUTPUT_START")
                add_span(model_span_map[key], timestamps, "compute output",
                         "COMPUTE_OUTPUT_START", "COMPUTE_END")

            if FLAGS.show_trace:
                print("{} ({}):".format(trace["model_name"],
                                        trace["model_version"]))
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
        print("Summary for {} ({}): trace count = {}".format(
            model_name, model_value, cnt))

        if "http infer" in model_span_map[key]:
            print("HTTP infer request (avg): {}us".format(
                model_span_map[key]["http infer"] / (cnt * 1000)))
            print("\tReceive (avg): {}us".format(
                model_span_map[key]["http recv"] / (cnt * 1000)))
            print("\tSend (avg): {}us".format(model_span_map[key]["http send"] /
                                              (cnt * 1000)))
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
            print("\tSend (avg): {}us".format(model_span_map[key]["grpc send"] /
                                              (cnt * 1000)))
            print("\tOverhead (avg): {}us".format(
                (model_span_map[key]["grpc infer"] -
                 model_span_map[key]["request handler"] -
                 model_span_map[key]["grpc wait/read"] -
                 model_span_map[key]["grpc send"]) / (cnt * 1000)))

        print("\tHandler (avg): {}us".format(
            model_span_map[key]["request handler"] / (cnt * 1000)))
        if ("queue"
                in model_span_map[key]) and "compute" in model_span_map[key]:
            print("\t\tOverhead (avg): {}us".format(
                (model_span_map[key]["request handler"] -
                 model_span_map[key]["queue"] - model_span_map[key]["compute"])
                / (cnt * 1000)))
            print("\t\tQueue (avg): {}us".format(model_span_map[key]["queue"] /
                                                 (cnt * 1000)))
            print("\t\tCompute (avg): {}us".format(
                model_span_map[key]["compute"] / (cnt * 1000)))
        if ("compute input" in model_span_map[key]
           ) and "compute output" in model_span_map[key]:
            print("\t\t\tInput (avg): {}us".format(
                model_span_map[key]["compute input"] / (cnt * 1000)))
            print("\t\t\tInfer (avg): {}us".format(
                model_span_map[key]["compute infer"] / (cnt * 1000)))
            print("\t\t\tOutput (avg): {}us".format(
                model_span_map[key]["compute output"] / (cnt * 1000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--preserve',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Timestamps is collected with preserve ordering')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    FLAGS = parser.parse_args()

    for f in FLAGS.file:
        trace_data = json.loads(f.read())
        exit(verify_timestamps(trace_data, FLAGS.preserve))
